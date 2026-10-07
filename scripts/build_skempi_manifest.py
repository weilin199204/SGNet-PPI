#!/usr/bin/env python
"""Build the exact official SSIF S166 wild-type/mutant evaluation manifest."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from aligned_io import chains_text, parse_chain_group
from pdb_contacts import rank_candidates, read_chain_atoms
from build_manifest import FIELDS, write_tsv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-list", default=str(
        ROOT / "protocol" / "splits" / "skempi_2.0_subset_list.txt"))
    parser.add_argument("--ppb-csv", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--training-list", default=str(ROOT / "protocol" / "splits" / "training_list.txt"))
    parser.add_argument("--out", default=str(ROOT / "work" / "manifests" / "skempi_s166.tsv"))
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    return parser.parse_args()


def official_items(path):
    rows, section = [], None
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        if text == "SKEMPI 2.0 subset wild":
            section = "wild"
            continue
        if text == "SKEMPI 2.0 subset mutant":
            section = "mutant"
            continue
        if section is None:
            raise ValueError("sample before section header: {}".format(text))
        rows.append((text, section))
    counts = Counter(section for _, section in rows)
    if len(rows) != 166 or counts != {"wild": 26, "mutant": 140}:
        raise ValueError("unexpected S166 composition: {}".format(counts))
    return rows


def official_mutations(sample_id):
    tokens = sample_id.split("_")
    if len(tokens) == 1:
        return tuple()
    result = []
    for text in tokens[3:]:
        match = re.match(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$", text)
        if match is None:
            raise ValueError("invalid official mutation {} in {}".format(text, sample_id))
        wt, chain, residue_id, mutant = match.groups()
        result.append((chain, wt, residue_id, mutant))
    return tuple(sorted(result))


def ppb_mutations(text):
    if not str(text).strip():
        return tuple()
    result = []
    for item in re.split(r"[,;~\s]+", str(text).strip()):
        if not item:
            continue
        match = re.match(r"^([A-Za-z0-9])_([A-Z])(-?\d+[A-Za-z]?)([A-Z])$", item)
        if match is None:
            raise ValueError("invalid PPB mutation {}".format(item))
        chain, wt, residue_id, mutant = match.groups()
        result.append((chain, wt, residue_id, mutant))
    return tuple(sorted(result))


def ppb_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Source Data Set"] == "SKEMPI v2.0":
                row["_mutations"] = ppb_mutations(row["Mutations"])
                rows.append(row)
    return rows


def select_record(sample_id, rows):
    pdb_id = sample_id.split("_", 1)[0].upper()
    mutations = official_mutations(sample_id)
    matches = [row for row in rows
               if row["PDB"].upper() == pdb_id and row["_mutations"] == mutations]
    if len(matches) != 1:
        raise ValueError("{} maps to {} PPB records".format(sample_id, len(matches)))
    return matches[0], mutations


def main():
    args = parse_args()
    items, records = official_items(args.subset_list), ppb_rows(args.ppb_csv)
    output = []
    for sample_id, kind in items:
        record, mutations = select_record(sample_id, records)
        pdb_id = record["PDB"].upper()
        group_a = parse_chain_group(record["Ligand Chains"])
        group_b = parse_chain_group(record["Receptor Chains"])
        if kind == "mutant":
            tokens = sample_id.split("_")
            listed_a, listed_b = parse_chain_group(tokens[1]), parse_chain_group(tokens[2])
            if (set(group_a), set(group_b)) != (set(listed_a), set(listed_b)):
                raise ValueError("{} partner groups disagree with PPB".format(sample_id))
        structure = Path(args.pdb_dir) / (pdb_id + ".pdb")
        if not structure.is_file():
            raise FileNotFoundError(structure)
        ranked = rank_candidates(read_chain_atoms(structure), [
            (group_a, group_b, "SSIF S166 list + PPB-Affinity.csv")],
            cutoff=args.contact_cutoff)
        if not ranked or ranked[0]["contact_pairs"] <= 0:
            raise ValueError("{} partner groups have no contacts".format(sample_id))
        contact = ranked[0]
        kd = float(record["KD(M)"])
        if not math.isfinite(kd) or kd <= 0:
            raise ValueError("invalid Kd for {}".format(sample_id))
        row = {field: "" for field in FIELDS}
        row.update({
            "sample_id": sample_id,
            "pdb_id": pdb_id,
            "source_pdb_id": pdb_id,
            "split": "skempi",
            "mutation": record["Mutations"],
            "ligand_chains": chains_text(group_a),
            "receptor_chains": chains_text(group_b),
            "label": "{:.10g}".format(-math.log10(kd)),
            "label_pk": "{:.10g}".format(-math.log10(kd)),
            "affinity_molar": "{:.10g}".format(kd),
            "affinity_metric": "Kd",
            "label_source": "PPB-Affinity.csv:SKEMPI v2.0",
            "chain_source": contact["source"],
            "structure_path": str(structure.resolve()),
            "structure_source": "SKEMPI v2.0 WT structure",
            "chain_contact_pairs": contact["contact_pairs"],
            "chain_contact_residues": contact["contact_residues"],
            "minimum_interatomic_distance": "{:.5f}".format(contact["minimum_distance"]),
            "runner_up_contact_pairs": 0,
            "chain_confidence": "1.000000",
            "status": "ok",
            "notes": ("WT" if not mutations else
                      "mutant sequence with shared WT coordinates/surface"),
        })
        output.append(row)
    out = Path(args.out)
    write_tsv(out, output)
    training = {line.strip().upper() for line in Path(args.training_list).read_text().splitlines()
                if line.strip()}
    bases = {row["pdb_id"] for row in output}
    summary = {
        "samples": len(output), "wild": 26, "mutant": 140,
        "unique_base_structures": len(bases),
        "training_pdb_overlap": sorted(bases & training),
        "label_source": "unique exact match in PPB-Affinity.csv SKEMPI v2.0",
        "surface_policy": "mutants share the WT experimental surface",
    }
    (out.parent / "skempi_s166_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
