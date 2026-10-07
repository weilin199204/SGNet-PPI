#!/usr/bin/env python
"""Create a stable uppercase-PDB view for all manifest structures."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--copy", action="store_true",
                        help="Copy structures instead of using relative symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for row in rows:
        if row["status"] == "invalid":
            continue
        pdb_id = row["pdb_id"].upper()
        source = Path(row["structure_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        previous = sources.setdefault(pdb_id, source)
        if previous != source:
            raise ValueError("{} maps to two structure files".format(pdb_id))
    linked = copied = existing = 0
    for pdb_id, source in sorted(sources.items()):
        target = out / (pdb_id + ".pdb")
        if target.exists() or target.is_symlink():
            if target.resolve() == source and not args.copy:
                existing += 1
                continue
            if not args.overwrite:
                raise FileExistsError(target)
            target.unlink()
        if args.copy:
            shutil.copy2(str(source), str(target))
            copied += 1
        else:
            target.symlink_to(os.path.relpath(str(source), str(out)))
            linked += 1
    report = {"manifest_rows": len(rows), "unique_structures": len(sources),
              "linked": linked, "copied": copied, "existing": existing,
              "out_dir": str(out.resolve())}
    (out / "structure_view.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
