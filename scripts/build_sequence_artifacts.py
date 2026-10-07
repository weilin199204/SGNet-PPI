#!/usr/bin/env python
"""Build sharded residue-sequence/ESM artifacts for the frozen SSIF manifest.

SGNet sequence-only and sequence+surface models consume residue ESM tensors but
not residue-graph edges. This builder deliberately avoids the quadratic legacy
contact construction while retaining the legacy file layout used by the reproducibility packer.
"""
from __future__ import absolute_import, print_function

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgnet_graph.constants import ATOM_PAIRS
from sgnet_graph.esm import EsmEmbedder
from sgnet_graph.metadata import read_metadata_csv
from sgnet_graph.pipeline import extract_ca_coords, find_pdb_file
from sgnet_graph.structure import concat_and_reindex, parse_residue_chains


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser.parse_args()


def artifact_paths(root, pdb_id):
    return (
        root / "graph" / "inter_graph" / pdb_id,
        root / "graph" / "individual_graph" / (pdb_id + "_1"),
        root / "graph" / "individual_graph" / (pdb_id + "_2"),
        root / "graph_construct" / "inter_graph" / pdb_id,
        root / "graph_construct" / "individual_graph" / (pdb_id + "_1"),
        root / "graph_construct" / "individual_graph" / (pdb_id + "_2"),
    )


def save_pickle(path, value, overwrite=False):
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_partner(path, chains):
    chain_residues = parse_residue_chains(path, chains)
    residues, sequences = concat_and_reindex(chain_residues)
    if not residues or not sequences:
        raise ValueError("no standard residues for chains {}".format(chains))
    return residues, sequences


def embed_sequences_chunked(embedder, sequences, chunk_size, overlap):
    if chunk_size < 2 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid ESM chunk configuration")
    outputs = []
    stride = chunk_size - overlap
    for sequence in sequences:
        if not sequence:
            continue
        if len(sequence) <= chunk_size:
            outputs.append(embedder.embed_sequences([sequence]))
            continue
        total = None
        counts = None
        for start in range(0, len(sequence), stride):
            stop = min(start + chunk_size, len(sequence))
            hidden = embedder.embed_sequences([sequence[start:stop]])
            if total is None:
                total = hidden.new_zeros((len(sequence), hidden.shape[-1]))
                counts = hidden.new_zeros((len(sequence), 1))
            total[start:stop] += hidden
            counts[start:stop] += 1
            if stop == len(sequence):
                break
        outputs.append(total / counts.clamp_min(1))
    if not outputs:
        raise ValueError("no non-empty sequences")
    return torch.cat(outputs, dim=0)


def build_entry(entry, pdb_file, root, embedder, overwrite, chunk_size, chunk_overlap):
    residues_a, sequences_a = parse_partner(pdb_file, entry.ligand_chains)
    residues_b, sequences_b = parse_partner(pdb_file, entry.receptor_chains)
    x_a = embed_sequences_chunked(embedder, list(sequences_a), chunk_size, chunk_overlap).float().contiguous()
    x_b = embed_sequences_chunked(embedder, list(sequences_b), chunk_size, chunk_overlap).float().contiguous()
    if x_a.shape[0] != len(residues_a) or x_b.shape[0] != len(residues_b):
        raise ValueError("ESM/residue length mismatch")

    empty_edges = torch.empty((2, 0), dtype=torch.long)
    edge_dim = len(ATOM_PAIRS) * 10
    empty_attr = torch.empty((0, edge_dim), dtype=torch.float32)
    inter = Data(
        x=torch.cat((x_a, x_b), dim=0),
        edge_index=empty_edges.clone(),
        edge_attr=empty_attr.clone(),
        pos=torch.cat((extract_ca_coords(residues_a),
                       extract_ca_coords(residues_b)), dim=0),
    )
    if entry.label is not None:
        inter.y = torch.tensor([round(float(entry.label), 2)], dtype=torch.float32)
    intra_a = Data(x=x_a, edge_index=empty_edges.clone(), edge_attr=empty_attr.clone(),
                   pos=extract_ca_coords(residues_a),
                   interface_mask=torch.zeros(len(residues_a), dtype=torch.bool))
    intra_b = Data(x=x_b, edge_index=empty_edges.clone(), edge_attr=empty_attr.clone(),
                   pos=extract_ca_coords(residues_b),
                   interface_mask=torch.zeros(len(residues_b), dtype=torch.bool))

    inter_construct = ([], empty_edges.clone(),
                       (sequences_a, sequences_b), (residues_a, residues_b))
    construct_a = ([], empty_edges.clone(), sequences_a, residues_a)
    construct_b = ([], empty_edges.clone(), sequences_b, residues_b)
    paths = artifact_paths(root, entry.pdb_id)
    for target, value in zip(paths, (
            inter, intra_a, intra_b, inter_construct, construct_a, construct_b)):
        save_pickle(target, value, overwrite=overwrite)
    return {"pdb_id": entry.pdb_id,
            "chains_a": list(entry.ligand_chains),
            "chains_b": list(entry.receptor_chains),
            "residues_a": len(residues_a), "residues_b": len(residues_b),
            "pdb_file": str(pdb_file)}


def main():
    args = parse_args()
    root = Path(args.out_dir)
    pdb_dir = Path(args.pdb_dir)
    entries = read_metadata_csv(Path(args.metadata))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    selected = [entry for index, entry in enumerate(entries)
                if index % args.num_shards == args.shard_index]
    missing = [entry for entry in selected if args.overwrite or
               not all(path.exists() for path in artifact_paths(root, entry.pdb_id))]
    print("metadata={} missing_sequence_artifacts={}".format(
        len(selected), len(missing)), flush=True)
    embedder = None
    built, failed = [], []
    for index, entry in enumerate(missing, 1):
        try:
            pdb_file = find_pdb_file(pdb_dir, entry.pdb_id)
            if pdb_file is None:
                raise FileNotFoundError("No PDB file found for {}".format(entry.pdb_id))
            if embedder is None:
                embedder = EsmEmbedder(model_name=args.esm_model, device=args.device)
            built.append(build_entry(entry, pdb_file, root, embedder, args.overwrite,
                                     args.chunk_size, args.chunk_overlap))
            print("built {}/{} {}".format(index, len(missing), entry.pdb_id), flush=True)
        except Exception as exc:
            failed.append({"pdb_id": entry.pdb_id,
                           "error": "{}: {}".format(type(exc).__name__, exc)})
    unresolved = [entry.pdb_id for entry in selected
                  if not all(path.exists() for path in artifact_paths(root, entry.pdb_id))]
    report = {"metadata": len(entries), "selected": len(selected),
              "num_shards": args.num_shards, "shard_index": args.shard_index,
              "requested": len(missing),
              "built": built, "failed": failed, "unresolved": unresolved,
              "edge_policy": "empty: SGNet5 consumes ESM tensors only"}
    report_dir = root / "sequence_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "sequence_shard_{:03d}.json".format(args.shard_index)).write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if failed or unresolved:
        raise RuntimeError("unresolved sequence artifacts; inspect report above")


if __name__ == "__main__":
    main()
