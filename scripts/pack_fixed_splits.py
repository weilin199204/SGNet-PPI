#!/usr/bin/env python
"""Pack exact official SSIF train/val/test splits into SGNet samples."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import pack_affinity_surface as base

ACCEPTED = {"ok", "inferred", "inferred_symmetry"}
EXPECTED = {"train": 2350, "val": 80, "test1": 79, "test2": 81}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_pickle(path, value, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(path))
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()
    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    invalid = [row for row in rows if row["status"] not in ACCEPTED]
    if invalid:
        raise ValueError("manifest contains unaccepted rows: {}".format(
            [(row["pdb_id"], row["status"]) for row in invalid[:20]]))
    observed = Counter(row["split"] for row in rows)
    if dict(observed) != EXPECTED:
        raise ValueError("split sizes {} do not equal {}".format(dict(observed), EXPECTED))

    graph_root = Path(args.graph_root)
    patch_root = Path(args.patch_root)
    packed = defaultdict(list)
    report = []
    failures = []
    for position, row in enumerate(rows, 1):
        pdb_id = row["pdb_id"].upper()
        try:
            sample, surface_status, message = base.pack_one(
                graph_root, patch_root, pdb_id)
            target = float(row["label_pk"])
            if not torch.isfinite(torch.tensor(target)):
                raise ValueError("non-finite target")
            sample["target"].y = torch.tensor([target], dtype=torch.float32)
            sample["split"] = row["split"]
            sample["label_source"] = row["label_source"]
            packed[row["split"]].append(sample)
            report.append({"pdb_id": pdb_id, "split": row["split"],
                           "status": "ok", "surface_status": surface_status,
                           "message": message})
        except Exception as exc:
            failures.append((pdb_id, str(exc)))
            report.append({"pdb_id": pdb_id, "split": row["split"],
                           "status": "invalid", "surface_status": "unavailable",
                           "message": str(exc)})
        if position % 50 == 0 or position == len(rows):
            print("processed {}/{} valid={}".format(
                position, len(rows), sum(len(value) for value in packed.values())),
                flush=True)
    if failures:
        raise RuntimeError("{} packing failures: {}".format(len(failures), failures[:20]))

    diagnostics = {}
    all_samples = []
    for split in ("train", "val", "test1", "test2"):
        if len(packed[split]) != EXPECTED[split]:
            raise ValueError("{} packed {} expected {}".format(
                split, len(packed[split]), EXPECTED[split]))
        diagnostics[split] = base.validate_affinity_samples(packed[split])
        all_samples.extend(packed[split])
    if len({sample["pdb_id"] for sample in all_samples}) != len(all_samples):
        raise ValueError("PDB overlap across official splits")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test1", "test2"):
        atomic_pickle(out / (split + "_dataset_save.pkl"), packed[split], args.overwrite)
    with (out / "pack_report.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pdb_id", "split", "status", "surface_status", "message"),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(report)
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "graph_root": str(graph_root.resolve()),
        "patch_root": str(patch_root.resolve()),
        "split_counts": {key: len(value) for key, value in packed.items()},
        "validation": diagnostics,
        "label_policy": "exact label_pk from frozen manifest",
        "surface_policy": ("observed patches plus marked placeholder; "
                           "materialized from training-only mean statistics"),
        "surface_available": sum(
            int(row["surface_status"] == "available") for row in report),
        "surface_imputed": sum(
            int(row["surface_status"] == "imputed") for row in report),
    }
    (out / "dataset_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
