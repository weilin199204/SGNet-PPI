#!/usr/bin/env python
"""Shared packing helpers for sequence tensors and v5 surface patch graphs.

The command-line entry point is retained for inspecting a custom dataset. The
paper protocol itself is packed by ``pack_fixed_splits.py``.
"""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgnet.data import validate_affinity_samples

PATCH_GRAPH_VERSION = 5
PATCH_NODE_DIM = 23
PATCH_EDGE_DIM = 42

PATCH_ATTRS = (
    "x", "edge_index", "edge_attr", "patch_area", "chain_id", "edge_type",
    "residue_ids", "residue_weights", "residue_counts", "patch_graph_version",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metadata", default=None,
                        help="Optional authoritative CSV/TSV used to enforce exact ID/label alignment.")
    parser.add_argument("--expected-size", type=int, default=1741)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def metadata_labels(path):
    with Path(path).open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        rows = list(csv.DictReader(handle, dialect=dialect))
    result = {}
    for row in rows:
        lower = {str(key).strip().lower(): value for key, value in row.items()}
        pdb_id = str(lower.get("pdb") or lower.get("pdb_id") or "").strip().upper()
        label = lower.get("label")
        if not pdb_id or label in (None, ""):
            raise ValueError("metadata contains an empty PDB ID or label")
        if pdb_id in result:
            raise ValueError("duplicate metadata PDB ID {}".format(pdb_id))
        result[pdb_id] = float(label)
    return result


def dump_pickle(path, value, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(path))
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def placeholder_patch(n_a, n_b, residue_slots=3):
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 2, 1, 3],
        [1, 0, 3, 2, 2, 0, 3, 1],
    ], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return Data(
        x=torch.zeros((4, PATCH_NODE_DIM), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.zeros((8, PATCH_EDGE_DIM), dtype=torch.float32),
        pos=torch.zeros((4, 3), dtype=torch.float32),
        normal=torch.zeros((4, 3), dtype=torch.float32),
        patch_area=torch.ones((4, 1), dtype=torch.float32),
        chain_id=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_type=edge_type,
        edge_vector=torch.zeros((8, 3), dtype=torch.float32),
        residue_ids=torch.full((4, residue_slots), -1, dtype=torch.long),
        residue_weights=torch.zeros((4, residue_slots), dtype=torch.float32),
        residue_counts=torch.tensor([[n_a, n_b]], dtype=torch.long),
        patch_graph_version=torch.tensor([PATCH_GRAPH_VERSION], dtype=torch.long),
        surface_available=torch.tensor([False], dtype=torch.bool),
    )


def compact_patch(patch, n_a, n_b):
    missing = [name for name in PATCH_ATTRS if not hasattr(patch, name)]
    if missing:
        raise ValueError("patch missing {}".format(missing))
    values = {name: getattr(patch, name).detach().cpu().clone()
              for name in PATCH_ATTRS}
    result = Data(**values)
    if result.x.ndim != 2 or result.x.shape[1] != PATCH_NODE_DIM:
        raise ValueError("patch node width is not {}".format(PATCH_NODE_DIM))
    if result.edge_attr.ndim != 2 or result.edge_attr.shape[1] != PATCH_EDGE_DIM:
        raise ValueError("patch edge width is not {}".format(PATCH_EDGE_DIM))
    if int(result.patch_graph_version.reshape(-1)[0]) != PATCH_GRAPH_VERSION:
        raise ValueError("patch version is not {}".format(PATCH_GRAPH_VERSION))
    if result.residue_counts.long().reshape(-1, 2)[0].tolist() != [n_a, n_b]:
        raise ValueError("patch/sequence residue count mismatch")
    if sorted(torch.unique(result.chain_id.long()).tolist()) != [0, 1]:
        raise ValueError("patch does not contain both partners")
    if int((result.edge_type.long() == 1).sum()) == 0:
        raise ValueError("patch has no cross edges")
    result.surface_available = torch.tensor([True], dtype=torch.bool)
    return result


def graph_sequences(root, pdb_id, n_a, n_b):
    directory = root / "graph_construct" / "individual_graph"
    chains_a = list(load_pickle(directory / (pdb_id + "_1"))[2])
    chains_b = list(load_pickle(directory / (pdb_id + "_2"))[2])
    sequence_a, sequence_b = "".join(chains_a), "".join(chains_b)
    if len(sequence_a) != n_a or len(sequence_b) != n_b:
        raise ValueError("graph-construct sequence/ESM length mismatch")
    values = torch.tensor([ord(value) for value in sequence_a + sequence_b],
                          dtype=torch.long)
    lengths_a = torch.tensor([len(value) for value in chains_a], dtype=torch.long)
    lengths_b = torch.tensor([len(value) for value in chains_b], dtype=torch.long)
    return values, lengths_a, lengths_b


def pack_one(graph_root, patch_root, pdb_id):
    graph_dir = graph_root / "graph"
    inter = load_pickle(graph_dir / "inter_graph" / pdb_id)
    intra_a = load_pickle(graph_dir / "individual_graph" / (pdb_id + "_1"))
    intra_b = load_pickle(graph_dir / "individual_graph" / (pdb_id + "_2"))
    x_a = intra_a.x.detach().cpu().float().contiguous()
    x_b = intra_b.x.detach().cpu().float().contiguous()
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[1] != x_b.shape[1]:
        raise ValueError("invalid ESM embedding shapes")
    n_a, n_b = int(x_a.shape[0]), int(x_b.shape[0])
    y = getattr(inter, "y", None)
    if y is None or y.numel() != 1 or not bool(torch.isfinite(y).all()):
        raise ValueError("missing/non-finite affinity")
    graph_ascii, chain_lengths_a, chain_lengths_b = graph_sequences(
        graph_root, pdb_id, n_a, n_b)
    target = Data(
        x=torch.empty((n_a + n_b, 0), dtype=torch.float32),
        y=y.detach().cpu().float().reshape(1),
        aux_graph_aa_ascii=graph_ascii,
        chain_lengths_a=chain_lengths_a,
        chain_lengths_b=chain_lengths_b,
    )
    patch_path = patch_root / pdb_id
    surface_status, message = "available", ""
    if patch_path.exists():
        try:
            patch = compact_patch(load_pickle(patch_path), n_a, n_b)
        except Exception as exc:
            patch = placeholder_patch(n_a, n_b)
            surface_status = "imputed"
            message = "invalid patch: {}".format(exc)
    else:
        patch = placeholder_patch(n_a, n_b)
        surface_status = "imputed"
        message = "missing patch graph"
    return ({
        "pdb_id": pdb_id,
        "sequence_a": Data(x=x_a),
        "sequence_b": Data(x=x_b),
        "target": target,
        "patch": patch,
    }, surface_status, message)


def main():
    args = parse_args()
    graph_root = Path(args.graph_root)
    patch_root = Path(args.patch_root)
    graph_ids = sorted(path.name.upper()
                       for path in (graph_root / "graph" / "inter_graph").iterdir()
                       if path.is_file())
    labels = metadata_labels(args.metadata) if args.metadata else None
    if labels is not None:
        ids = list(labels)
        missing = sorted(set(ids) - set(graph_ids))
        extra = sorted(set(graph_ids) - set(ids))
        if missing or extra:
            raise ValueError("graph/metadata ID mismatch: missing={} extra={}".format(
                missing, extra))
    else:
        ids = graph_ids
    if len(ids) != args.expected_size:
        raise ValueError("expected {} graph IDs, found {}".format(args.expected_size, len(ids)))
    packed, reports = [], []
    for position, pdb_id in enumerate(ids, 1):
        try:
            sample, surface_status, message = pack_one(
                graph_root, patch_root, pdb_id.upper())
            if labels is not None:
                observed = float(sample["target"].y.reshape(-1)[0])
                expected = round(float(labels[pdb_id]), 2)
                if abs(observed - expected) > 1e-4:
                    raise ValueError("graph/metadata label mismatch: {} != {}".format(
                        observed, expected))
            packed.append(sample)
            reports.append({"pdb_id": pdb_id.upper(), "status": "ok",
                            "surface_status": surface_status, "message": message})
        except Exception as exc:
            reports.append({"pdb_id": pdb_id.upper(), "status": "invalid",
                            "surface_status": "unavailable", "message": str(exc)})
        if position % 50 == 0 or position == len(ids):
            print("processed {}/{}; valid={}".format(position, len(ids), len(packed)),
                  flush=True)
    invalid = [row for row in reports if row["status"] != "ok"]
    if invalid:
        raise RuntimeError("{} affinity samples invalid: {}".format(
            len(invalid), invalid[:10]))
    validation = validate_affinity_samples(packed)
    order = list(range(len(packed)))
    random.Random(args.seed).shuffle(order)
    split_at = int(len(order) * args.train_ratio)
    train = [packed[index] for index in order[:split_at]]
    val = [packed[index] for index in order[split_at:]]
    split_by_id = {sample["pdb_id"]: "train" for sample in train}
    split_by_id.update({sample["pdb_id"]: "val" for sample in val})
    for row in reports:
        row["source_split"] = split_by_id.get(row["pdb_id"], "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pack_report.tsv").open("w", newline="") as handle:
        fields = ("pdb_id", "source_split", "status", "surface_status", "message")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(reports)
    diagnostics = dict(validation)
    diagnostics.update({"expected_size": args.expected_size,
                        "train": len(train), "val": len(val),
                        "graph_root": str(graph_root.resolve()),
                        "patch_root": str(patch_root.resolve()),
                        "metadata": (str(Path(args.metadata).resolve())
                                     if args.metadata else None)})
    (out / "dataset_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n")
    dump_pickle(out / "train_dataset_save.pkl", train, args.overwrite)
    dump_pickle(out / "val_dataset_save.pkl", val, args.overwrite)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
