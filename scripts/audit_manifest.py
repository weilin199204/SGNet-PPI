#!/usr/bin/env python
"""Validate IDs, splits, labels, structures, chains, and inference provenance."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPECTED = {"train": 2350, "val": 80, "test1": 79, "test2": 81}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strict-inferred", action="store_true",
                        help="Reject both review and inferred chain assignments.")
    parser.add_argument("--report")
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    errors, warnings = [], []
    ids = [row["sample_id"].upper() for row in rows]
    duplicates = [key for key, value in Counter(ids).items() if value > 1]
    if duplicates:
        errors.append("duplicate sample IDs: {}".format(duplicates[:20]))
    observed = Counter(row["split"] for row in rows)
    if dict(observed) != EXPECTED:
        errors.append("split counts {} != {}".format(dict(observed), EXPECTED))
    split_sets = {split: {row["sample_id"].upper() for row in rows
                          if row["split"] == split} for split in EXPECTED}
    names = sorted(split_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                errors.append("{} / {} overlap: {}".format(
                    left, right, sorted(overlap)[:20]))
    for row in rows:
        sample = row["sample_id"]
        if row["status"] == "invalid":
            errors.append("{} invalid: {}".format(sample, row["notes"]))
        elif row["status"] == "review":
            errors.append("{} requires chain review: {}".format(sample, row["notes"]))
        elif row["status"] in {"inferred", "inferred_symmetry"}:
            message = "{} uses inferred chains {} / {} (confidence {})".format(
                sample, row["ligand_chains"], row["receptor_chains"],
                row["chain_confidence"])
            (errors if args.strict_inferred else warnings).append(message)
        if not Path(row["structure_path"]).is_file():
            errors.append("{} missing structure {}".format(sample, row["structure_path"]))
        try:
            label = float(row["label_pk"])
            molar = float(row["affinity_molar"])
            if not (0.0 < label < 20.0 and molar > 0.0):
                raise ValueError
        except ValueError:
            errors.append("{} invalid affinity fields".format(sample))
        if not row["ligand_chains"] or not row["receptor_chains"]:
            errors.append("{} missing partner chains".format(sample))
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(rows),
        "split_counts": dict(observed),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "chain_source_counts": dict(Counter(row["chain_source"] for row in rows)),
        "label_source_counts": dict(Counter(row["label_source"] for row in rows)),
        "errors": errors,
        "warnings": warnings,
    }
    path = Path(args.report or (str(args.manifest) + ".audit.json"))
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
