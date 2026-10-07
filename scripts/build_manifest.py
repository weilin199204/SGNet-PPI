#!/usr/bin/env python
"""Build auditable SSIF train/val/test manifests from local PPI sources."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aligned_io import (chains_text, concentration_to_molar, normalize_pdb_id,
                        parse_chain_group, parse_index_affinities,
                        parse_pairwise_composition, pk_from_molar,
                        read_nonempty_lines, xlsx_rows)
from pdb_contacts import (confidence, groups_sequence_equivalent,
                          infer_chain_pair, rank_candidates, read_chain_atoms)


FIELDS = (
    "sample_id", "pdb_id", "source_pdb_id", "split", "mutation",
    "ligand_chains", "receptor_chains", "label", "label_pk",
    "affinity_molar", "affinity_metric", "label_source", "chain_source",
    "structure_path", "structure_source", "chain_contact_pairs",
    "chain_contact_residues", "minimum_interatomic_distance",
    "runner_up_contact_pairs", "chain_confidence", "status", "notes",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default=str(ROOT / "protocol" / "splits"))
    parser.add_argument("--corrected-xlsx", required=True)
    parser.add_argument("--ppb-csv", required=True)
    parser.add_argument("--pdbbind-dir", required=True)
    parser.add_argument("--pdbbind-index", required=True)
    parser.add_argument("--affinity-benchmark-dir", required=True)
    parser.add_argument("--affinity-benchmark-xlsx", required=True)
    parser.add_argument("--replacement-dir", default=str(ROOT / "work" / "raw_pdb"))
    parser.add_argument("--chain-overrides", default=str(ROOT / "chain_overrides.tsv"))
    parser.add_argument("--out-dir", default=str(ROOT / "work" / "manifests"))
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--low-confidence", type=float, default=0.10)
    return parser.parse_args()


def chain_overrides(path):
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pdb_id = normalize_pdb_id(row.get("pdb_id", ""))
            group_a = parse_chain_group(row.get("ligand_chains", ""))
            group_b = parse_chain_group(row.get("receptor_chains", ""))
            if not pdb_id or not group_a or not group_b or set(group_a) & set(group_b):
                raise ValueError("invalid chain override: {}".format(row))
            result[pdb_id] = {
                "group_a": group_a,
                "group_b": group_b,
                "reason": str(row.get("reason", "")).strip(),
            }
    return result


def corrected_records(path):
    records, replacements = {}, {}
    for row in xlsx_rows(path):
        pdb_id = normalize_pdb_id(row.get("PDB ID", ""))
        if not pdb_id:
            continue
        group_a, group_b = parse_pairwise_composition(row.get("pairwise composition", ""))
        molar = concentration_to_molar(row.get("value"), row.get("unit"))
        record = {"pdb_id": pdb_id, "group_a": group_a, "group_b": group_b,
                  "molar": molar, "metric": row.get("metric", ""), "row": row}
        records[pdb_id] = record
        note = str(row.get("", "") or row.get("notes", "") or "")
        # The correction workbook stores comments in an unnamed seventh column.
        for value in row.values():
            if str(value).lower().startswith("new pdb:"):
                replacement = normalize_pdb_id(str(value).split(":", 1)[1])
                replacements[replacement] = pdb_id
                break
    return records, replacements


def ppb_records(path):
    result = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = normalize_pdb_id(row.get("PDB", ""))
            if not pdb_id:
                continue
            group_a = parse_chain_group(row.get("Ligand Chains", ""))
            group_b = parse_chain_group(row.get("Receptor Chains", ""))
            if not group_a or not group_b or set(group_a) & set(group_b):
                continue
            result[pdb_id].append({"group_a": group_a, "group_b": group_b,
                                   "source": row.get("Source Data Set", ""), "row": row})
    return result


def benchmark_records(path):
    result = {}
    for row in xlsx_rows(path):
        pdb_id = normalize_pdb_id(row.get("PDB", ""))
        if not pdb_id:
            continue
        group_a = parse_chain_group(row.get("Ligand Chains", ""))
        group_b = parse_chain_group(row.get("Receptor Chains", ""))
        affinity_text = str(row.get("KD(M)", "")).strip()
        qualifier = "upper_bound" if affinity_text.startswith("<") else ""
        molar = float(affinity_text.lstrip("<>~"))
        result[pdb_id] = {"group_a": group_a, "group_b": group_b,
                          "molar": molar, "metric": "Kd",
                          "qualifier": qualifier}
    return result


def structure_file(pdb_id, split, args):
    if pdb_id in {"5BXQ", "7SQ2"}:
        path = Path(args.replacement_dir) / (pdb_id + ".pdb")
        return path, "RCSB replacement"
    if split == "test1":
        path = Path(args.affinity_benchmark_dir) / (pdb_id + ".pdb")
        return path, "Affinity Benchmark v5.5"
    root = Path(args.pdbbind_dir)
    for name in (pdb_id + ".pdb", pdb_id.lower() + ".ent.pdb",
                 pdb_id + ".ent.pdb"):
        path = root / name
        if path.exists():
            return path, "PDBbind v2020 PP"
    return root / (pdb_id.lower() + ".ent.pdb"), "PDBbind v2020 PP"


def split_ids(split_dir):
    split_dir = Path(split_dir)
    train = [x.upper() for x in read_nonempty_lines(split_dir / "training_list.txt")]
    val = [x.upper() for x in read_nonempty_lines(split_dir / "val_list.txt")]
    test = read_nonempty_lines(split_dir / "test_list.txt")
    marker = test.index("Test set 2")
    test1 = [x.upper() for x in test[1:marker]]
    test2 = [x.upper() for x in test[marker + 1:]]
    return {"train": train, "val": val, "test1": test1, "test2": test2}


def label_for(pdb_id, source_pdb_id, split, index, benchmark, corrected):
    if split == "test1":
        value = benchmark.get(pdb_id)
        if value is None:
            raise KeyError("ABv3 has no label for {}".format(pdb_id))
        source = "ABv3.xlsx"
        if value.get("qualifier"):
            source += ":" + value["qualifier"]
        return value["molar"], value["metric"], source
    value = index.get(source_pdb_id)
    if value is not None:
        source = "INDEX_general_PP.2020"
        if value.get("qualifier"):
            source += ":" + value["qualifier"]
        return value["affinity_molar"], value["affinity_metric"], source
    value = corrected.get(source_pdb_id)
    if value is None:
        raise KeyError("no affinity for {}".format(pdb_id))
    return value["molar"], value["metric"], "PPI_dataset_2283 replacement"


def choose_chains(pdb_id, source_pdb_id, split, path, corrected, ppb,
                  benchmark, overrides, cutoff, low_confidence):
    atoms = read_chain_atoms(path)
    candidates, authoritative = [], False
    override = overrides.get(pdb_id) or overrides.get(source_pdb_id)
    override_note = ""
    if override is not None:
        candidates.append((override["group_a"], override["group_b"],
                           "chain_overrides.tsv"))
        authoritative = True
        override_note = override["reason"]
    elif split == "test1" and pdb_id in benchmark:
        value = benchmark[pdb_id]
        candidates.append((value["group_a"], value["group_b"], "ABv3.xlsx"))
        authoritative = True
    elif source_pdb_id in corrected:
        value = corrected[source_pdb_id]
        candidates.append((value["group_a"], value["group_b"],
                           "PPI_dataset_2283.xlsx"))
        authoritative = True
    if not candidates:
        for value in ppb.get(pdb_id, []) + ppb.get(source_pdb_id, []):
            candidates.append((value["group_a"], value["group_b"],
                               "PPB-Affinity.csv:" + value["source"]))
    fallback_note = override_note
    ranked = rank_candidates(atoms, candidates, cutoff=cutoff) if candidates else []
    if ranked and ranked[0]["contact_pairs"] == 0 and authoritative:
        fallback_candidates = [
            (value["group_a"], value["group_b"],
             "PPB-Affinity.csv:" + value["source"] + ":declared_zero_contact_fallback")
            for value in ppb.get(pdb_id, []) + ppb.get(source_pdb_id, [])
        ]
        fallback = rank_candidates(atoms, fallback_candidates, cutoff=cutoff)
        if fallback and fallback[0]["contact_pairs"] > 0:
            ranked = fallback
            fallback_note = "authoritative pair had zero contacts; used PPB partner groups"
    if not ranked:
        ranked = infer_chain_pair(atoms, cutoff=cutoff)
        authoritative = False
    if not ranked:
        raise ValueError("no valid protein chain pair")
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    margin = confidence(best, runner)
    status = "ok"
    notes = fallback_note
    if best["contact_pairs"] == 0:
        status, notes = "invalid", "declared partners have no atoms within cutoff"
    elif not authoritative and best["source"] == "atom_contact":
        status = "inferred"
        if margin < low_confidence:
            equivalent = runner is not None and groups_sequence_equivalent(
                atoms, best["group_a"], best["group_b"],
                runner["group_a"], runner["group_b"])
            if equivalent:
                status = "inferred_symmetry"
                notes = "runner-up {} / {} is sequence-equivalent".format(
                    chains_text(runner["group_a"]), chains_text(runner["group_b"]))
            else:
                status = "review"
                notes = "automatic pair has close runner-up {} / {}".format(
                    chains_text(runner["group_a"]), chains_text(runner["group_b"]))
    return best, runner, margin, status, notes


def make_rows(args):
    corrected, replacements = corrected_records(args.corrected_xlsx)
    ppb = ppb_records(args.ppb_csv)
    benchmark = benchmark_records(args.affinity_benchmark_xlsx)
    overrides = chain_overrides(args.chain_overrides)
    index = parse_index_affinities(args.pdbbind_index)
    rows = []
    for split, ids in split_ids(args.split_dir).items():
        for pdb_id in ids:
            source_pdb_id = replacements.get(pdb_id, pdb_id)
            path, structure_source = structure_file(pdb_id, split, args)
            row = {field: "" for field in FIELDS}
            row.update({"sample_id": pdb_id, "pdb_id": pdb_id,
                        "source_pdb_id": source_pdb_id, "split": split,
                        "structure_path": str(path.resolve()),
                        "structure_source": structure_source})
            try:
                if not path.exists():
                    raise FileNotFoundError(path)
                molar, metric, label_source = label_for(
                    pdb_id, source_pdb_id, split, index, benchmark, corrected)
                best, runner, margin, status, notes = choose_chains(
                    pdb_id, source_pdb_id, split, path, corrected, ppb,
                    benchmark, overrides, args.contact_cutoff,
                    args.low_confidence)
                row.update({
                    "ligand_chains": chains_text(best["group_a"]),
                    "receptor_chains": chains_text(best["group_b"]),
                    "label": "{:.10g}".format(pk_from_molar(molar)),
                    "label_pk": "{:.10g}".format(pk_from_molar(molar)),
                    "affinity_molar": "{:.10g}".format(molar),
                    "affinity_metric": metric,
                    "label_source": label_source,
                    "chain_source": best["source"],
                    "chain_contact_pairs": best["contact_pairs"],
                    "chain_contact_residues": best["contact_residues"],
                    "minimum_interatomic_distance": "{:.5f}".format(
                        best["minimum_distance"]),
                    "runner_up_contact_pairs": (runner["contact_pairs"] if runner else 0),
                    "chain_confidence": "{:.6f}".format(margin),
                    "status": status,
                    "notes": notes,
                })
            except Exception as exc:
                row["status"] = "invalid"
                row["notes"] = "{}: {}".format(type(exc).__name__, exc)
            rows.append(row)
    return rows, replacements


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows, replacements = make_rows(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "all.tsv", rows)
    for split in ("train", "val", "test1", "test2"):
        write_tsv(out / (split + ".tsv"), [row for row in rows if row["split"] == split])
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["split"]][row["status"]] += 1
    summary = {
        "total": len(rows),
        "expected": {"train": 2350, "val": 80, "test1": 79, "test2": 81},
        "status_by_split": {key: dict(value) for key, value in counts.items()},
        "replacements": replacements,
        "paths": {key: str(value) for key, value in vars(args).items()
                  if key.endswith("dir") or key.endswith("csv") or key.endswith("xlsx")
                  or key.endswith("index")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if any(row["status"] == "invalid" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
