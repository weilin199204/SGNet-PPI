#!/usr/bin/env python
"""Validate all manifest partner PLYs and write the exact surface index."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
from pathlib import Path


ACCEPTED = {"ok", "inferred", "inferred_symmetry", "review"}


def group_name(value):
    return "".join(item.strip() for item in str(value).replace(";", ",").split(","))


def valid_ply(path):
    if not path.is_file() or path.stat().st_size < 256:
        return False
    with path.open("rb") as handle:
        return handle.read(3).lower() == b"ply"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--surface-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing", type=int, default=0,
                        help="Maximum number of complexes delegated to pack-time imputation")
    args = parser.parse_args()
    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    ply_dir = Path(args.surface_root) / "complete_ply"
    index_rows, missing, seen = [], [], set()
    for row in rows:
        if row["status"] not in ACCEPTED:
            continue
        pdb_id = row["pdb_id"].upper()
        ligand, receptor = group_name(row["ligand_chains"]), group_name(row["receptor_chains"])
        key = (pdb_id, ligand, receptor)
        if key in seen:
            continue
        seen.add(key)
        paths = [ply_dir / "{}_{}.ply".format(pdb_id, ligand),
                 ply_dir / "{}_{}.ply".format(pdb_id, receptor)]
        if not all(valid_ply(path) for path in paths):
            missing.append({"pdb_id": pdb_id, "ligand": str(paths[0]),
                            "receptor": str(paths[1])})
            continue
        index_rows.append({"dataset_entry": "{}_{}_{}".format(pdb_id, ligand, receptor),
                           "ligand_ply": str(paths[0].resolve()),
                           "receptor_ply": str(paths[1].resolve())})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset_entry", "ligand_ply",
                                                     "receptor_ply"), delimiter="\t")
        writer.writeheader(); writer.writerows(index_rows)
    summary = {"manifest_complexes": len(seen), "valid": len(index_rows),
               "missing": len(missing), "allow_missing": int(args.allow_missing),
               "imputed_at_pack_time": [item["pdb_id"] for item in missing],
               "missing_entries": missing,
               "surface_index": str(out.resolve())}
    (out.parent / "surface_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "missing_entries"}, indent=2), flush=True)
    if len(missing) > args.allow_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
