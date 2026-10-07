#!/usr/bin/env python
"""Build SGNet4 interface patches directly from complete PLY meshes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sgnet_graph.interface_patch import (  # noqa: E402
    PatchBuildConfig,
    build_interface_patch_graph_from_ply,
    write_patch_metadata,
)
from sgnet_graph.surface import find_surface_files  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", required=True,
                        help="Directory containing the two complete PLY meshes per complex")
    parser.add_argument("--surface-index",
                        help="Optional TSV mapping dataset entries to explicit PLY paths")
    parser.add_argument("--atom-root", required=True,
                        help="Root containing graph_construct/inter_graph")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--graph-name", default="interface_patch_full")
    parser.add_argument("--ids", help="Optional text/TSV/CSV file; first field is PDB ID")
    parser.add_argument("--pdb-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero when any entry fails")
    parser.add_argument("--report")
    parser.add_argument("--patch-radius", type=float, default=4.0)
    parser.add_argument("--surface-cross-cutoff", type=float, default=5.0)
    parser.add_argument("--surface-interface-cutoff", type=float, default=5.0)
    parser.add_argument("--atom-contact-cutoff", type=float, default=4.5)
    parser.add_argument("--delta-sasa-threshold", type=float, default=0.1)
    parser.add_argument("--context-hops", type=int, default=1)
    parser.add_argument("--sasa-points", type=int, default=192)
    parser.add_argument("--max-patches-per-chain", type=int, default=384)
    parser.add_argument("--atom-surface-k", type=int, default=16)
    parser.add_argument("--atom-surface-max-gap", type=float, default=1.75)
    parser.add_argument("--atom-surface-sigma", type=float, default=0.75)
    parser.add_argument("--min-delta-projection", type=float, default=0.8)
    return parser.parse_args()


def normalize_id(value):
    name = Path(str(value).strip()).name
    for suffix in (".ent.pdb", ".pdbqt", ".pdb"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name.upper()


def read_ids(path):
    ids = []
    with Path(path).open() as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            first = text.replace(",", "\t").split("\t", 1)[0].split()[0]
            pdb_id = normalize_id(first)
            if pdb_id.lower() in {"pdb", "pdb_id", "filename"}:
                continue
            ids.append(pdb_id)
    return ids


def discover_ids(surface_dir, atom_root, surface_index=None):
    atom_dir = Path(atom_root) / "graph_construct" / "inter_graph"
    atom_ids = sorted(path.name.upper() for path in atom_dir.iterdir() if path.is_file())
    matched = []
    for pdb_id in atom_ids:
        try:
            find_surface_files(surface_dir, pdb_id, surface_index=surface_index)
        except FileNotFoundError:
            continue
        matched.append(pdb_id)
    return matched


def make_config(args):
    return PatchBuildConfig(
        patch_radius=args.patch_radius,
        surface_cross_cutoff=args.surface_cross_cutoff,
        surface_interface_cutoff=args.surface_interface_cutoff,
        atom_contact_cutoff=args.atom_contact_cutoff,
        delta_sasa_threshold=args.delta_sasa_threshold,
        context_hops=args.context_hops,
        sasa_points=args.sasa_points,
        max_patches_per_chain=args.max_patches_per_chain,
        atom_surface_k=args.atom_surface_k,
        atom_surface_max_gap=args.atom_surface_max_gap,
        atom_surface_sigma=args.atom_surface_sigma,
        min_delta_projection=args.min_delta_projection,
    )


def build_one(task):
    (pdb_id, surface_dir, surface_index, atom_root, out_root,
     graph_name, config_dict, overwrite) = task
    started = time.time()
    atom_path = Path(atom_root) / "graph_construct" / "inter_graph" / pdb_id
    output_path = Path(out_root) / "surfgraph_4" / graph_name / pdb_id
    row = {
        "pdb_id": pdb_id,
        "status": "failed",
        "seconds": 0.0,
        "patches": "",
        "intra_edges": "",
        "cross_edges": "",
        "delta_sasa": "",
        "captured_delta_fraction": "",
        "full_vertices": "",
        "selected_vertices": "",
        "selected_fraction": "",
        "ply_iface_vertices": "",
        "delta_sasa_vertices": "",
        "atom_contact_vertices": "",
        "surface_proximity_vertices": "",
        "error": "",
    }
    try:
        if not atom_path.exists():
            raise FileNotFoundError("missing atom source: {}".format(atom_path))
        graph, status = build_interface_patch_graph_from_ply(
            surface_dir,
            pdb_id,
            atom_path,
            output_path,
            surface_index=surface_index,
            config=PatchBuildConfig(**config_dict),
            overwrite=overwrite,
        )
        row["status"] = status
        if graph is not None:
            edge_type = graph.edge_type.long()
            summary = graph.desolvation_summary.view(-1)
            selection = graph.interface_selection_summary.view(-1)
            full_vertices = int(selection[0] + selection[1])
            selected_vertices = int(selection[2] + selection[3])
            row.update({
                "patches": int(graph.num_nodes),
                "intra_edges": int((edge_type == 0).sum()),
                "cross_edges": int((edge_type == 1).sum()),
                "delta_sasa": float(summary[0] + summary[1]),
                "captured_delta_fraction": float(0.5 * (summary[4] + summary[5])),
                "full_vertices": full_vertices,
                "selected_vertices": selected_vertices,
                "selected_fraction": selected_vertices / max(full_vertices, 1),
                "ply_iface_vertices": int(selection[4] + selection[5]),
                "delta_sasa_vertices": int(selection[6] + selection[7]),
                "atom_contact_vertices": int(selection[8] + selection[9]),
                "surface_proximity_vertices": int(selection[10] + selection[11]),
            })
    except Exception as exc:
        row["error"] = "{}: {}".format(type(exc).__name__, exc)
        row["traceback"] = traceback.format_exc()
    row["seconds"] = round(time.time() - started, 3)
    return row


def write_report(path, rows):
    fields = (
        "pdb_id", "status", "seconds", "patches", "intra_edges",
        "cross_edges", "delta_sasa", "captured_delta_fraction",
        "full_vertices", "selected_vertices", "selected_fraction",
        "ply_iface_vertices", "delta_sasa_vertices", "atom_contact_vertices",
        "surface_proximity_vertices", "error",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main():
    args = parse_args()
    config = make_config(args)
    requested = [normalize_id(value) for value in args.pdb_id]
    if args.ids:
        requested.extend(read_ids(args.ids))
    ids = sorted(dict.fromkeys(requested)) if requested else discover_ids(
        args.surface_dir, args.atom_root, surface_index=args.surface_index
    )
    if args.limit is not None:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No matching complete PLY and atom graph IDs found")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    metadata_path = write_patch_metadata(
        args.out_root, config, graph_name=args.graph_name
    )
    tasks = [
        (pdb_id, args.surface_dir, args.surface_index, args.atom_root,
         args.out_root, args.graph_name, asdict(config), args.overwrite)
        for pdb_id in ids
    ]
    rows = []
    if args.workers == 1:
        for index, task in enumerate(tasks, start=1):
            row = build_one(task)
            rows.append(row)
            print("[{}/{}] {} {} {}".format(
                index, len(tasks), row["pdb_id"], row["status"], row["error"]
            ), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(build_one, task): task[0] for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print("[{}/{}] {} {} {}".format(
                    index, len(tasks), row["pdb_id"], row["status"], row["error"]
                ), flush=True)
    rows.sort(key=lambda row: row["pdb_id"])
    report = args.report or str(
        Path(args.out_root) / "surfgraph_4" / "{}_build_report.tsv".format(args.graph_name)
    )
    write_report(report, rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    result = {
        "requested": len(ids),
        "status_counts": counts,
        "report": str(Path(report).resolve()),
        "metadata": str(metadata_path.resolve()),
        "graph_name": args.graph_name,
        "config": asdict(config),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
