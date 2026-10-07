#!/usr/bin/env python
"""Evaluate the locked SGNet real arm and an optional shuffled control on S166."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from sgnet.model import SurfaceResidualV2
from train_sgnet import (
    evaluate_adapter, full_sequence_predictions, make_surface_loader, metrics,
    resolve_device, set_seed, torch_load)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sequence-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--include-shuffled", action="store_true",
        help="also load run-dir/shuffled/best_surface_residual_v2.pt")
    return parser.parse_args()


def subset_metrics(result, mask, key):
    return metrics(result["truth"][mask], result[key][mask])


def load_arm(run, arm, device):
    checkpoint = torch_load(
        run / arm / "best_surface_residual_v2.pt")
    config = checkpoint["model_config"]
    model = SurfaceResidualV2(
        normalization=checkpoint["surface_normalization"],
        baseline_mean=checkpoint["baseline_mean"],
        baseline_std=checkpoint["baseline_std"],
        hidden_channels=int(config["surface_hidden"]),
        graph_channels=int(config["graph_hidden"]),
        dropout=float(config["surface_dropout"]),
        initial_reliability=float(config["initial_reliability"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    with Path(args.data).open("rb") as handle:
        samples = pickle.load(handle)
    if len(samples) != 166:
        raise ValueError("expected 166 S166 samples, found {}".format(len(samples)))
    run = Path(args.run_dir)
    sequence_checkpoint = torch_load(args.sequence_checkpoint)
    esm_dim = int(samples[0]["sequence_a"].x.shape[1])
    baseline = full_sequence_predictions(
        samples, sequence_checkpoint, args, device, esm_dim)

    mutation_mask = np.asarray([bool(sample["mutation"]) for sample in samples])
    groups = {
        "all": np.ones(len(samples), dtype=bool),
        "wild": ~mutation_mask,
        "mutant": mutation_mask,
    }
    arms = ("real", "shuffled") if args.include_shuffled else ("real",)
    arm_results = {}
    for arm in arms:
        model, checkpoint = load_arm(run, arm, device)
        loader = make_surface_loader(
            samples, range(len(samples)), args.batch_size, False,
            args.seed + 100, checkpoint["surface_imputation_stats"])
        result = evaluate_adapter(model, loader, baseline, device)
        if result["pdb_ids"] != [sample["pdb_id"] for sample in samples]:
            raise ValueError("S166 sample order mismatch")
        arm_results[arm] = result

    summary = {}
    for group, mask in groups.items():
        base = subset_metrics(arm_results["real"], mask, "baseline")
        summary[group] = {"sequence": base}
        for arm in arms:
            result = arm_results[arm]
            score = subset_metrics(result, mask, "prediction")
            summary[group][arm] = score
            summary[group][arm + "_delta_mae"] = score["mae"] - base["mae"]
            summary[group][arm + "_delta_pearson"] = (
                score["pearson"] - base["pearson"])
            summary[group][arm + "_gate_mean"] = float(
                result["reliability"][mask].mean())
            summary[group][arm + "_correction_std"] = float(
                result["correction"][mask].std())

    out = run / "skempi_s166"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "surface_residual_v2_predictions.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id", "base_pdb_id", "mutation", "subset", "y_true",
            "sequence_prediction", "real_prediction", "real_correction",
            "real_gate"]
        if "shuffled" in arm_results:
            fields.extend((
                "shuffled_prediction", "shuffled_correction",
                "shuffled_gate"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        real = arm_results["real"]
        for index, sample in enumerate(samples):
            row = {
                "sample_id": sample["sample_id"],
                "base_pdb_id": sample["base_pdb_id"],
                "mutation": sample["mutation"],
                "subset": "mutant" if sample["mutation"] else "wild",
                "y_true": float(real["truth"][index]),
                "sequence_prediction": float(baseline[index]),
                "real_prediction": float(real["prediction"][index]),
                "real_correction": float(real["correction"][index]),
                "real_gate": float(real["reliability"][index]),
            }
            if "shuffled" in arm_results:
                shuffled = arm_results["shuffled"]
                row.update({
                    "shuffled_prediction": float(
                        shuffled["prediction"][index]),
                    "shuffled_correction": float(
                        shuffled["correction"][index]),
                    "shuffled_gate": float(
                        shuffled["reliability"][index]),
                })
            writer.writerow(row)
    report = {
        "protocol": "official SSIF S166: 26 WT + 140 mutants",
        "model_selection_used_s166": False,
        "mutant_structure_policy": "mutant ESM plus shared WT experimental surface",
        "known_training_pdb_overlap": ["1YCS"],
        "sequence_checkpoint": str(Path(args.sequence_checkpoint).resolve()),
        "v2_run": str(run.resolve()),
        "surface_arms": list(arms),
        "metrics": summary,
    }
    (out / "surface_residual_v2_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
