#!/usr/bin/env python
"""Pack mutation-aware ESM tensors and shared WT surface patches for S166."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import pack_affinity_surface as base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pack_one(graph_root, patch_root, row):
    sample_id, pdb_id = row["sample_id"], row["pdb_id"].upper()
    graph_dir = graph_root / "graph"
    inter = base.load_pickle(graph_dir / "inter_graph" / sample_id)
    intra_a = base.load_pickle(graph_dir / "individual_graph" / (sample_id + "_1"))
    intra_b = base.load_pickle(graph_dir / "individual_graph" / (sample_id + "_2"))
    x_a = intra_a.x.detach().cpu().float().contiguous()
    x_b = intra_b.x.detach().cpu().float().contiguous()
    n_a, n_b = int(x_a.shape[0]), int(x_b.shape[0])
    if n_a < 1 or n_b < 1 or x_a.shape[1] != x_b.shape[1]:
        raise ValueError("invalid ESM shapes")
    graph_ascii, lengths_a, lengths_b = base.graph_sequences(
        graph_root, sample_id, n_a, n_b)
    target = Data(
        x=torch.empty((n_a + n_b, 0), dtype=torch.float32),
        y=torch.tensor([float(row["label_pk"])], dtype=torch.float32),
        aux_graph_aa_ascii=graph_ascii,
        chain_lengths_a=lengths_a,
        chain_lengths_b=lengths_b,
    )
    patch_path = patch_root / pdb_id
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    patch = base.compact_patch(base.load_pickle(patch_path), n_a, n_b)
    observed = float(inter.y.reshape(-1)[0])
    if not torch.isfinite(torch.tensor(observed)):
        raise ValueError("non-finite graph label")
    return {
        "pdb_id": sample_id,
        "sample_id": sample_id,
        "base_pdb_id": pdb_id,
        "mutation": row["mutation"],
        "sequence_a": Data(x=x_a),
        "sequence_b": Data(x=x_b),
        "target": target,
        "patch": patch,
    }


def main():
    args = parse_args()
    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 166:
        raise ValueError("S166 manifest has {} rows".format(len(rows)))
    graph_root, patch_root = Path(args.graph_root), Path(args.patch_root)
    samples, report, failed = [], [], []
    for index, row in enumerate(rows, 1):
        try:
            samples.append(pack_one(graph_root, patch_root, row))
            report.append({"sample_id": row["sample_id"], "pdb_id": row["pdb_id"],
                           "status": "ok", "message": ""})
        except Exception as exc:
            failed.append((row["sample_id"], str(exc)))
            report.append({"sample_id": row["sample_id"], "pdb_id": row["pdb_id"],
                           "status": "invalid", "message": str(exc)})
        if index % 25 == 0 or index == len(rows):
            print("processed {}/{} valid={}".format(index, len(rows), len(samples)),
                  flush=True)
    if failed:
        raise RuntimeError("{} invalid S166 samples: {}".format(len(failed), failed[:20]))
    diagnostics = base.validate_affinity_samples(samples)
    if diagnostics["surface_available"] != 166:
        raise ValueError("S166 must use observed WT patches")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(out))
    temporary = out.with_name(out.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("wb") as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(out)
    finally:
        if temporary.exists():
            temporary.unlink()
    with out.with_suffix(".report.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "pdb_id", "status", "message"),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(report)
    summary = dict(diagnostics)
    summary.update({"wild": sum(not sample["mutation"] for sample in samples),
                    "mutant": sum(bool(sample["mutation"]) for sample in samples),
                    "surface_policy": "mutants share base_pdb_id WT patch",
                    "output": str(out.resolve())})
    out.with_suffix(".diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
