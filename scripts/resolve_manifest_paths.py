#!/usr/bin/env python
"""Resolve portable manifest path tokens to local structure directories."""
from __future__ import absolute_import, print_function

import argparse
import csv
from pathlib import Path


TOKENS = {
    "PDBBIND_ROOT": "pdbbind_dir",
    "AFFINITY_BENCHMARK_ROOT": "affinity_benchmark_dir",
    "SKEMPI_ROOT": "skempi_dir",
    "REPLACEMENT_ROOT": "replacement_dir",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pdbbind-dir")
    parser.add_argument("--affinity-benchmark-dir")
    parser.add_argument("--skempi-dir")
    parser.add_argument("--replacement-dir")
    parser.add_argument(
        "--check", action="store_true",
        help="Fail if a resolved structure file is absent.")
    return parser.parse_args()


def resolve_path(value, roots):
    text = str(value).strip()
    for token, root in roots.items():
        prefix = token + "/"
        if text.startswith(prefix):
            if root is None:
                raise ValueError(
                    "{} requires the corresponding directory argument".format(text))
            return root / text[len(prefix):]
    return Path(text)


def main():
    args = parse_args()
    roots = {
        token: (Path(getattr(args, argument)).resolve()
                if getattr(args, argument) else None)
        for token, argument in TOKENS.items()
    }
    source = Path(args.manifest)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames
    missing = []
    for row in rows:
        path = resolve_path(row["structure_path"], roots).resolve()
        row["structure_path"] = str(path)
        if args.check and not path.is_file():
            missing.append((row.get("sample_id", ""), str(path)))
    if missing:
        raise FileNotFoundError(
            "{} structure files are missing; first entries: {}".format(
                len(missing), missing[:10]))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} rows to {}".format(len(rows), target.resolve()))


if __name__ == "__main__":
    main()
