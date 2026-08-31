#!/usr/bin/env python3
"""
checkm_lite.py — fast, lightweight completeness/redundancy estimation for
metagenomic bins using a set of single-copy marker gene HMMs (e.g. the
71-gene bacterial marker set), Prodigal, and HMMER.

Pipeline per bin:
    1. Prodigal (meta mode) predicts proteins from the bin FASTA.
    2. hmmsearch scans predicted proteins against the marker HMM database.
    3. Each protein is assigned to at most one marker (its best hit by
       e-value/score, above a bit-score/e-value threshold).
    4. completeness = (# distinct markers with >=1 hit) / total_markers * 100
       redundancy    = (# extra hits beyond one per marker) / total_markers * 100

Requires `prodigal` and `hmmsearch` (HMMER3) on PATH.

Usage:
    python checkm_lite.py --hmm_db markers_71.hmm --bin_dir bins/ --out results.csv

Runs fully sequentially by default (one bin at a time, one CPU per hmmsearch) to keep
RAM/CPU usage low. Raise --threads / --hmm_threads only if you have resources to spare.

Optional:
    --extension fa,fasta,fna     comma-separated bin file extensions (default: fa,fasta,fna)
    --threads N                  parallel workers across bins (default: 1, sequential)
    --hmm_threads N              CPU threads given to each hmmsearch call (default: 1)
    --evalue 1e-10               hmmsearch reporting e-value threshold
    --keep_tmp                   keep intermediate prodigal/hmmsearch files
    --tmp_dir DIR                where intermediate files are written (default: <out>_tmp)
"""

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FASTA_EXTS_DEFAULT = "fa,fasta,fna"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_bin_files(bin_dir: Path, extensions):
    exts = {e.strip().lstrip(".").lower() for e in extensions.split(",") if e.strip()}
    bins = []
    for p in sorted(bin_dir.iterdir()):
        if p.is_file() and p.suffix.lstrip(".").lower() in exts:
            bins.append(p)
    return bins


def fasta_length(path: Path) -> int:
    """Total number of bases (non-header, non-whitespace characters) in a FASTA file."""
    total = 0
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            total += len(line.strip())
    return total


def count_markers(hmm_db: Path) -> int:
    """Count NAME lines in the HMM database to report how many markers are expected."""
    n = 0
    try:
        with open(hmm_db, "r", errors="ignore") as fh:
            for line in fh:
                if line.startswith("NAME"):
                    n += 1
    except OSError:
        pass
    return n


def run_prodigal(bin_fasta: Path, out_faa: Path, log_fh) -> bool:
    cmd = ["prodigal", "-i", str(bin_fasta), "-a", str(out_faa), "-p", "meta", "-q"]
    try:
        subprocess.run(cmd, stdout=log_fh, stderr=log_fh, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        eprint(f"[prodigal failed] {bin_fasta.name}: {e}")
        return False
    return out_faa.exists() and out_faa.stat().st_size > 0


def run_hmmsearch(faa: Path, hmm_db: Path, out_tbl: Path, evalue: float, cpu: int, log_fh) -> bool:
    cmd = [
        "hmmsearch",
        "--tblout", str(out_tbl),
        "-E", str(evalue),
        "--cpu", str(cpu),
        "--noali",
        str(hmm_db),
        str(faa),
    ]
    try:
        subprocess.run(cmd, stdout=log_fh, stderr=log_fh, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        eprint(f"[hmmsearch failed] {faa.name}: {e}")
        return False
    return out_tbl.exists()


def parse_hmmsearch_tblout(tbl_path: Path):
    """
    Parse HMMER --tblout output. Returns a dict: protein_query -> (marker_name, evalue, score)
    keeping only the best (lowest e-value, then highest score) hit per protein query,
    so each predicted protein is assigned to a single marker.
    """
    best_hit = {}
    if not tbl_path.exists():
        return best_hit
    with open(tbl_path, "r") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            # hmmsearch tblout columns: target_name target_acc query_name query_acc E-value score bias ...
            # Since the HMM db is the query file and the protein FASTA is the target file,
            # "target name" = predicted protein, "query name" = marker HMM.
            protein_name = fields[0]
            marker_name = fields[2]
            try:
                evalue = float(fields[4])
                score = float(fields[5])
            except (ValueError, IndexError):
                continue
            prev = best_hit.get(protein_name)
            if prev is None or evalue < prev[1] or (evalue == prev[1] and score > prev[2]):
                best_hit[protein_name] = (marker_name, evalue, score)
    return best_hit


def process_bin(args):
    (bin_path, hmm_db, tmp_dir, evalue, hmm_threads, total_markers, keep_tmp) = args
    bin_name = bin_path.stem
    bin_tmp = tmp_dir / bin_name
    bin_tmp.mkdir(parents=True, exist_ok=True)

    log_path = bin_tmp / "log.txt"
    faa_path = bin_tmp / f"{bin_name}.faa"
    tbl_path = bin_tmp / f"{bin_name}.tblout"

    length = fasta_length(bin_path)

    with open(log_path, "w") as log_fh:
        ok = run_prodigal(bin_path, faa_path, log_fh)
        if not ok:
            result = {"bin": bin_name, "length": length, "completeness": 0.0,
                      "redundancy": 0.0, "markers_found": 0, "total_hits": 0,
                      "note": "prodigal_failed"}
            if not keep_tmp:
                shutil.rmtree(bin_tmp, ignore_errors=True)
            return result

        ok = run_hmmsearch(faa_path, hmm_db, tbl_path, evalue, hmm_threads, log_fh)
        if not ok:
            result = {"bin": bin_name, "length": length, "completeness": 0.0,
                      "redundancy": 0.0, "markers_found": 0, "total_hits": 0,
                      "note": "hmmsearch_failed"}
            if not keep_tmp:
                shutil.rmtree(bin_tmp, ignore_errors=True)
            return result

    best_hits = parse_hmmsearch_tblout(tbl_path)

    marker_counts = {}
    for query, (marker, ev, sc) in best_hits.items():
        marker_counts[marker] = marker_counts.get(marker, 0) + 1

    markers_found = len(marker_counts)
    total_hits = sum(marker_counts.values())
    extra_hits = sum(c - 1 for c in marker_counts.values() if c > 1)

    denom = total_markers if total_markers > 0 else max(markers_found, 1)
    completeness = 100.0 * markers_found / denom
    redundancy = 100.0 * extra_hits / denom

    if not keep_tmp:
        shutil.rmtree(bin_tmp, ignore_errors=True)

    return {
        "bin": bin_name,
        "length": length,
        "completeness": round(completeness, 2),
        "redundancy": round(redundancy, 2),
        "markers_found": markers_found,
        "total_hits": total_hits,
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hmm_db", required=True, type=Path, help="Path to single-copy marker gene HMM database (e.g. 71 bacterial SCGs)")
    ap.add_argument("--bin_dir", required=True, type=Path, help="Folder containing bin FASTA files")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--extension", default=FASTA_EXTS_DEFAULT, help=f"Comma-separated bin file extensions (default: {FASTA_EXTS_DEFAULT})")
    ap.add_argument("--threads", type=int, default=1, help="Number of bins to process in parallel (default: 1, i.e. sequential — increase only if you have RAM/CPU to spare)")
    ap.add_argument("--hmm_threads", type=int, default=1, help="CPU threads per hmmsearch call (default: 1)")
    ap.add_argument("--evalue", type=float, default=1e-10, help="hmmsearch e-value threshold (default: 1e-10)")
    ap.add_argument("--n_markers", type=int, default=None, help="Total number of markers in the HMM DB (default: auto-counted from NAME lines)")
    ap.add_argument("--tmp_dir", type=Path, default=None, help="Directory for intermediate files (default: <out>_tmp)")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep intermediate prodigal/hmmsearch files")
    args = ap.parse_args()

    for exe in ("prodigal", "hmmsearch"):
        if shutil.which(exe) is None:
            eprint(f"ERROR: '{exe}' not found on PATH. Please install it (e.g. `apt install prodigal hmmer` or `conda install -c bioconda {exe}`).")
            sys.exit(1)

    if not args.hmm_db.exists():
        eprint(f"ERROR: HMM database not found: {args.hmm_db}")
        sys.exit(1)
    if not args.bin_dir.is_dir():
        eprint(f"ERROR: bin_dir is not a directory: {args.bin_dir}")
        sys.exit(1)

    bins = find_bin_files(args.bin_dir, args.extension)
    if not bins:
        eprint(f"ERROR: no bin files found in {args.bin_dir} with extension(s) {args.extension}")
        sys.exit(1)

    total_markers = args.n_markers if args.n_markers else count_markers(args.hmm_db)
    if total_markers == 0:
        eprint("WARNING: could not determine marker count from HMM DB NAME lines; "
               "completeness/redundancy will be normalized by markers actually found instead. "
               "Pass --n_markers to fix this.")

    tmp_dir = args.tmp_dir if args.tmp_dir else Path(str(args.out) + "_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mode = "sequentially (1 bin at a time)" if args.threads <= 1 else f"with {args.threads} parallel workers"
    eprint(f"Found {len(bins)} bins. Marker set size: {total_markers or 'unknown'}. Running {mode}...")

    job_args = [(b, args.hmm_db, tmp_dir, args.evalue, args.hmm_threads, total_markers, args.keep_tmp) for b in bins]

    results = []
    n_workers = max(1, min(args.threads, len(bins)))
    if n_workers == 1:
        for ja in job_args:
            results.append(process_bin(ja))
            eprint(f"  done: {results[-1]['bin']}")
    else:
        with mp.Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_bin, job_args), 1):
                results.append(res)
                eprint(f"  [{i}/{len(bins)}] done: {res['bin']}")

    results.sort(key=lambda r: r["bin"])

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bin", "length", "completeness", "redundancy"])
        for r in results:
            writer.writerow([r["bin"], r["length"], r["completeness"], r["redundancy"]])

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    eprint(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
