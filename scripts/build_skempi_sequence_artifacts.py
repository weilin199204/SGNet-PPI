#!/usr/bin/env python
"""Build mutation-aware ESM artifacts for S166 while retaining WT coordinates."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_sequence_artifacts import (artifact_paths, embed_sequences_chunked,
                                      save_pickle)
from build_skempi_manifest import ppb_mutations
from sgnet_graph.constants import ATOM_PAIRS
from sgnet_graph.esm import EsmEmbedder
from sgnet_graph.pipeline import extract_ca_coords
from sgnet_graph.structure import concat_and_reindex, parse_residue_chains


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="Mutation-specific ESM graph root, keyed by sample_id")
    parser.add_argument("--atom-root", required=True,
                        help="WT atom-construct root, keyed by base PDB ID")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def chains(value):
    return tuple(item.strip() for item in str(value).replace(";", ",").split(",")
                 if item.strip())


def parse_partner(path, requested):
    by_chain = parse_residue_chains(path, requested)
    residues, sequences = concat_and_reindex(by_chain)
    if not residues or not sequences:
        raise ValueError("no standard residues for chains {}".format(requested))
    return residues, tuple(sequences)


def residue_number(text):
    match = re.match(r"^(-?\d+)([A-Za-z]?)$", str(text))
    if match is None:
        raise ValueError("unsupported mutation residue ID {}".format(text))
    return int(match.group(1)), match.group(2).upper()


def mutate_sequences(residues_a, sequences_a, residues_b, sequences_b, mutations):
    merged = list(residues_a) + list(residues_b)
    sequence = list("".join(sequences_a) + "".join(sequences_b))
    offset_b = len(residues_a)
    applied = []
    for chain, wild_type, residue_id, mutant in mutations:
        resseq, icode = residue_number(residue_id)
        matches = []
        for global_index, residue in enumerate(merged):
            if (residue["chain"] == chain and int(residue["resseq"]) == resseq and
                    str(residue.get("icode", "")).upper() == icode):
                matches.append((global_index, residue))
        if len(matches) != 1:
            raise ValueError("mutation {}_{}{}{} maps to {} residues".format(
                chain, wild_type, residue_id, mutant, len(matches)))
        index, residue = matches[0]
        observed = str(residue["type"]).upper()
        if observed != wild_type:
            raise ValueError("mutation {} expects {}, PDB has {}".format(
                (chain, residue_id), wild_type, observed))
        sequence[index] = mutant
        applied.append({"chain": chain, "residue_id": residue_id,
                        "wild_type": wild_type, "mutant": mutant,
                        "partner": "A" if index < offset_b else "B"})
    lengths = [len(value) for value in sequences_a + sequences_b]
    pieces, cursor = [], 0
    for length in lengths:
        pieces.append("".join(sequence[cursor:cursor + length]))
        cursor += length
    return tuple(pieces[:len(sequences_a)]), tuple(pieces[len(sequences_a):]), applied


def embed_partner(embedder, sequences, cache, chunk_size, overlap):
    values = []
    for sequence in sequences:
        if sequence not in cache:
            cache[sequence] = embed_sequences_chunked(
                embedder, [sequence], chunk_size, overlap).float().contiguous()
        values.append(cache[sequence])
    return torch.cat(values, dim=0)


def save_wt_construct(root, pdb_id, residues_a, sequences_a,
                      residues_b, sequences_b, overwrite):
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    inter = ([], empty_edges.clone(), (sequences_a, sequences_b),
             (residues_a, residues_b))
    intra_a = ([], empty_edges.clone(), sequences_a, residues_a)
    intra_b = ([], empty_edges.clone(), sequences_b, residues_b)
    targets = (
        root / "graph_construct" / "inter_graph" / pdb_id,
        root / "graph_construct" / "individual_graph" / (pdb_id + "_1"),
        root / "graph_construct" / "individual_graph" / (pdb_id + "_2"),
    )
    for target, value in zip(targets, (inter, intra_a, intra_b)):
        save_pickle(target, value, overwrite=overwrite)


def build_sample(row, pdb_dir, out, atom_root, embedder, cache, args):
    sample_id, pdb_id = row["sample_id"], row["pdb_id"].upper()
    path = Path(pdb_dir) / (pdb_id + ".pdb")
    if not path.is_file():
        raise FileNotFoundError(path)
    residues_a, wt_a = parse_partner(path, chains(row["ligand_chains"]))
    residues_b, wt_b = parse_partner(path, chains(row["receptor_chains"]))
    mutations = ppb_mutations(row["mutation"])
    seq_a, seq_b, applied = mutate_sequences(
        residues_a, wt_a, residues_b, wt_b, mutations)
    if args.validate_only:
        return {"sample_id": sample_id, "pdb_id": pdb_id,
                "mutations": len(applied), "residues_a": len(residues_a),
                "residues_b": len(residues_b)}

    save_wt_construct(atom_root, pdb_id, residues_a, wt_a,
                      residues_b, wt_b, args.overwrite)
    x_a = embed_partner(embedder, seq_a, cache, args.chunk_size, args.chunk_overlap)
    x_b = embed_partner(embedder, seq_b, cache, args.chunk_size, args.chunk_overlap)
    if x_a.shape[0] != len(residues_a) or x_b.shape[0] != len(residues_b):
        raise ValueError("ESM/residue length mismatch")
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    empty_attr = torch.empty((0, len(ATOM_PAIRS) * 10), dtype=torch.float32)
    inter = Data(x=torch.cat((x_a, x_b), 0), edge_index=empty_edges.clone(),
                 edge_attr=empty_attr.clone(),
                 pos=torch.cat((extract_ca_coords(residues_a),
                                extract_ca_coords(residues_b)), 0),
                 y=torch.tensor([float(row["label_pk"])], dtype=torch.float32))
    intra_a = Data(x=x_a, edge_index=empty_edges.clone(), edge_attr=empty_attr.clone(),
                   pos=extract_ca_coords(residues_a),
                   interface_mask=torch.zeros(len(residues_a), dtype=torch.bool))
    intra_b = Data(x=x_b, edge_index=empty_edges.clone(), edge_attr=empty_attr.clone(),
                   pos=extract_ca_coords(residues_b),
                   interface_mask=torch.zeros(len(residues_b), dtype=torch.bool))
    inter_construct = ([], empty_edges.clone(), (seq_a, seq_b),
                       (residues_a, residues_b))
    construct_a = ([], empty_edges.clone(), seq_a, residues_a)
    construct_b = ([], empty_edges.clone(), seq_b, residues_b)
    for target, value in zip(artifact_paths(out, sample_id), (
            inter, intra_a, intra_b, inter_construct, construct_a, construct_b)):
        save_pickle(target, value, overwrite=args.overwrite)
    return {"sample_id": sample_id, "pdb_id": pdb_id,
            "mutations": len(applied), "residues_a": len(residues_a),
            "residues_b": len(residues_b)}


def main():
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for index, row in enumerate(rows)
                if index % args.num_shards == args.shard_index]
    out, atom_root = Path(args.out_dir), Path(args.atom_root)
    embedder = None if args.validate_only else EsmEmbedder(
        model_name=args.esm_model, device=args.device)
    cache, built, failed = {}, [], []
    for index, row in enumerate(selected, 1):
        paths = artifact_paths(out, row["sample_id"])
        if (not args.validate_only and not args.overwrite and
                all(path.exists() for path in paths)):
            built.append({"sample_id": row["sample_id"], "status": "existing"})
            continue
        try:
            built.append(build_sample(
                row, args.pdb_dir, out, atom_root, embedder, cache, args))
            print("[{}/{}] {} ok".format(index, len(selected), row["sample_id"]),
                  flush=True)
        except Exception as exc:
            failed.append({"sample_id": row["sample_id"],
                           "error": "{}: {}".format(type(exc).__name__, exc)})
            print("[{}/{}] {} failed {}".format(
                index, len(selected), row["sample_id"], failed[-1]["error"]),
                flush=True)
    report = {"manifest": len(rows), "selected": len(selected),
              "shard_index": args.shard_index, "num_shards": args.num_shards,
              "validate_only": args.validate_only, "built": len(built),
              "failed": failed, "unique_sequence_cache": len(cache)}
    report_dir = out / "sequence_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "validate" if args.validate_only else "build"
    (report_dir / "{}_shard_{:03d}.json".format(suffix, args.shard_index)).write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
