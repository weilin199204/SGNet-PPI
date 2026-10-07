#!/usr/bin/env python
"""Build missing ordered partner PLY surfaces for one deterministic shard."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ACCEPTED = {"ok", "inferred", "inferred_symmetry", "review"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-pdb-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-prefix", default="surface",
                        help="Prefix for shard reports; useful for retries")
    parser.add_argument(
        "--masif-root", required=True,
        help="Path to a MaSIF checkout (tested with LPDI-EPFL/masif).")
    return parser.parse_args()


def group_name(value):
    return "".join(item.strip() for item in str(value).replace(";", ",").split(","))


def valid_ply(path):
    if not path.is_file() or path.stat().st_size < 256:
        return False
    with path.open("rb") as handle:
        return handle.read(3).lower() == b"ply"


def read_tasks(path, num_shards, shard_index):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard configuration")
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    tasks, seen = [], set()
    for row in rows:
        if row["status"] not in ACCEPTED:
            continue
        pdb_id = row["pdb_id"].upper()
        ligand = group_name(row["ligand_chains"])
        receptor = group_name(row["receptor_chains"])
        key = (pdb_id, ligand, receptor)
        if key in seen:
            continue
        seen.add(key)
        tasks.append(key)
    tasks.sort()
    return [task for index, task in enumerate(tasks)
            if index % num_shards == shard_index], len(tasks)


def run_partner(wrapper, pdb_id, group, context_chains, raw_dir, ply_dir, pdb_dir,
                masif_root):
    entry = "{}_{}".format(pdb_id, group)
    command = [
        sys.executable, str(wrapper), entry,
        "--raw-pdb-dir", str(raw_dir),
        "--output-dir", str(ply_dir),
        "--pdb-output-dir", str(pdb_dir),
        "--context-chains", str(context_chains),
        "--masif-root", str(masif_root),
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, check=False)
    if completed.returncode:
        raise RuntimeError("{} failed:\n{}".format(entry, completed.stdout[-5000:]))


def main():
    args = parse_args()
    tasks, total = read_tasks(args.manifest, args.num_shards, args.shard_index)
    out = Path(args.out_dir)
    ply_dir, pdb_dir = out / "complete_ply", out / "partner_pdbs"
    report_dir = out / "shard_reports"
    for directory in (ply_dir, pdb_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_pdb_dir).resolve()
    wrapper = Path(__file__).with_name("run_masif_surface.py").resolve()
    rows = []
    for position, (pdb_id, ligand, receptor) in enumerate(tasks, 1):
        started = time.time()
        row = {"pdb_id": pdb_id, "ligand_chains": ligand,
               "receptor_chains": receptor, "status": "failed",
               "computed_partners": 0, "seconds": 0.0, "error": ""}
        try:
            raw = raw_dir / (pdb_id + ".pdb")
            if not raw.is_file():
                raise FileNotFoundError(raw)
            computed = 0
            for group in (ligand, receptor):
                target = ply_dir / "{}_{}.ply".format(pdb_id, group)
                if valid_ply(target) and not args.overwrite:
                    continue
                if target.exists():
                    target.unlink()
                run_partner(wrapper, pdb_id, group, ligand + receptor,
                            raw_dir.resolve(),
                            ply_dir.resolve(), pdb_dir.resolve(),
                            Path(args.masif_root).resolve())
                if not valid_ply(target):
                    raise FileNotFoundError("invalid/missing output {}".format(target))
                computed += 1
            row["computed_partners"] = computed
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = "{}: {}".format(type(exc).__name__, exc)
        row["seconds"] = round(time.time() - started, 3)
        rows.append(row)
        print("shard {}/{} [{}/{}] {} {} computed={} {}".format(
            args.shard_index, args.num_shards, position, len(tasks), pdb_id,
            row["status"], row["computed_partners"], row["error"]), flush=True)
    report_path = report_dir / "{}_shard_{:03d}.tsv".format(
        args.report_prefix, args.shard_index)
    fields = ("pdb_id", "ligand_chains", "receptor_chains", "status",
              "computed_partners", "seconds", "error")
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(row["status"] != "ok" for row in rows)
    summary = {"total_manifest_complexes": total, "shard": args.shard_index,
               "num_shards": args.num_shards, "requested": len(rows),
               "ok": len(rows) - failed, "failed": failed,
               "report": str(report_path.resolve())}
    (report_dir / "{}_shard_{:03d}.json".format(
        args.report_prefix, args.shard_index)).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if args.strict and failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
