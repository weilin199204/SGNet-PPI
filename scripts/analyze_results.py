#!/usr/bin/env python
"""Recompute paper metrics and paired-bootstrap intervals from frozen predictions."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root", default=str(root / "results_snapshot"))
    parser.add_argument(
        "--output-dir", default=str(root / "reproduced_results"))
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--sample-bootstrap", type=int, default=50000)
    parser.add_argument("--cluster-bootstrap", type=int, default=20000)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def array(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "n": int(len(truth)),
        "mse": float(np.mean((prediction - truth) ** 2)),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "pearson": float(np.corrcoef(prediction, truth)[0, 1]),
        "spearman": float(spearmanr(prediction, truth).correlation),
    }


def metric_delta(truth, treatment, reference, metric):
    if metric == "MAE":
        return float(np.mean(np.abs(treatment - truth))
                     - np.mean(np.abs(reference - truth)))
    if metric == "Pearson":
        return float(np.corrcoef(treatment, truth)[0, 1]
                     - np.corrcoef(reference, truth)[0, 1])
    raise ValueError(metric)


def paired_bootstrap(truth, treatment, reference, count, seed, metric):
    rng = np.random.default_rng(seed)
    n = len(truth)
    values = np.empty(count, dtype=np.float64)
    batch_size = 256
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        indices = rng.integers(0, n, size=(stop - start, n))
        y = truth[indices]
        left = treatment[indices]
        right = reference[indices]
        if metric == "MAE":
            values[start:stop] = (
                np.mean(np.abs(left - y), axis=1)
                - np.mean(np.abs(right - y), axis=1))
        else:
            y_centered = y - y.mean(axis=1, keepdims=True)
            left_centered = left - left.mean(axis=1, keepdims=True)
            right_centered = right - right.mean(axis=1, keepdims=True)
            left_r = np.sum(y_centered * left_centered, axis=1) / np.sqrt(
                np.sum(y_centered ** 2, axis=1)
                * np.sum(left_centered ** 2, axis=1))
            right_r = np.sum(y_centered * right_centered, axis=1) / np.sqrt(
                np.sum(y_centered ** 2, axis=1)
                * np.sum(right_centered ** 2, axis=1))
            values[start:stop] = left_r - right_r
    values = values[np.isfinite(values)]
    if len(values) < 0.95 * count:
        raise RuntimeError("too few valid bootstrap replicates")
    return np.percentile(values, [2.5, 97.5]).tolist()


def cluster_bootstrap(truth, treatment, reference, groups, count, seed, metric):
    unique = np.asarray(sorted(set(groups)))
    by_group = {
        group: np.flatnonzero(np.asarray(groups) == group) for group in unique
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(count):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        value = metric_delta(
            truth[indices], treatment[indices], reference[indices], metric)
        if np.isfinite(value):
            values.append(value)
    if len(values) < 0.95 * count:
        raise RuntimeError("too few valid cluster-bootstrap replicates")
    return np.percentile(values, [2.5, 97.5]).tolist()


def interpretation(metric, low, high):
    if low <= 0.0 <= high:
        return "not significant"
    if metric == "MAE":
        return "significant improvement" if high < 0 else "significant degradation"
    return "significant improvement" if low > 0 else "significant degradation"


def append_metric_rows(output, dataset, subset, effective_units, predictions, role):
    truth = predictions["truth"]
    baseline = metrics(truth, predictions["sequence"])
    for model, key in (
            ("sequence", "sequence"),
            ("surface_residual_v2", "real"),
            ("shuffled_control", "shuffled")):
        score = metrics(truth, predictions[key])
        output.append({
            "dataset": dataset,
            "subset": subset,
            "n": score["n"],
            "effective_units": effective_units,
            "model": model,
            "mse": score["mse"],
            "mae": score["mae"],
            "pearson": score["pearson"],
            "spearman": score["spearman"],
            "delta_mae_vs_sequence": score["mae"] - baseline["mae"],
            "delta_pearson_vs_sequence": (
                score["pearson"] - baseline["pearson"]),
            "role": ("negative_control" if model == "shuffled_control"
                     else role),
        })


def external_predictions(root, split):
    real = read_csv(root / "real" / (split + "_predictions.csv"))
    shuffled = read_csv(root / "shuffled" / (split + "_predictions.csv"))
    real_ids = [row["pdb_id"] for row in real]
    if real_ids != [row["pdb_id"] for row in shuffled]:
        raise ValueError("{} real/control ID mismatch".format(split))
    truth = array(real, "y_true")
    if not np.allclose(truth, array(shuffled, "y_true"), atol=0.0, rtol=0.0):
        raise ValueError("{} real/control labels differ".format(split))
    sequence = array(real, "sequence_prediction")
    if not np.allclose(
            sequence, array(shuffled, "sequence_prediction"), atol=0.0, rtol=0.0):
        raise ValueError("{} real/control baselines differ".format(split))
    return {
        "ids": real_ids,
        "truth": truth,
        "sequence": sequence,
        "real": array(real, "surface_prediction"),
        "shuffled": array(shuffled, "surface_prediction"),
    }


def concatenate_predictions(parts):
    result = {"ids": []}
    for part in parts:
        result["ids"].extend(part["ids"])
    for key in ("truth", "sequence", "real", "shuffled"):
        result[key] = np.concatenate([part[key] for part in parts])
    return result


def subset_predictions(predictions, mask):
    return {
        key: (value[mask] if isinstance(value, np.ndarray) else
              [item for item, keep in zip(value, mask) if keep])
        for key, value in predictions.items()
    }


def s166_predictions(root):
    rows = read_csv(
        root / "skempi_s166" / "surface_residual_v2_predictions.csv")
    return {
        "ids": [row["sample_id"] for row in rows],
        "groups": [row["base_pdb_id"] for row in rows],
        "subsets": [row["subset"] for row in rows],
        "truth": array(rows, "y_true"),
        "sequence": array(rows, "sequence_prediction"),
        "real": array(rows, "real_prediction"),
        "shuffled": array(rows, "shuffled_prediction"),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.snapshot_root)
    metric_rows = []
    statistical_rows = []
    seed_offset = 0
    external = {}

    for split, role in (
            ("val", "model_selection"),
            ("test1", "external_test"),
            ("test2", "external_test")):
        predictions = external_predictions(root, split)
        external[split] = predictions
        append_metric_rows(
            metric_rows, "validation" if split == "val" else split,
            "all", len(predictions["truth"]), predictions, role)
        for comparison, left, right in (
                ("real_minus_sequence", "real", "sequence"),
                ("shuffled_minus_sequence", "shuffled", "sequence"),
                ("real_minus_shuffled", "real", "shuffled")):
            for metric in ("MAE", "Pearson"):
                seed_offset += 1
                point = metric_delta(
                    predictions["truth"], predictions[left],
                    predictions[right], metric)
                low, high = paired_bootstrap(
                    predictions["truth"], predictions[left],
                    predictions[right], args.sample_bootstrap,
                    args.seed + seed_offset, metric)
                statistical_rows.append({
                    "dataset": "validation" if split == "val" else split,
                    "subset": "all",
                    "comparison": comparison,
                    "metric": metric,
                    "point_delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "resampling_unit": "sample",
                    "n_resamples": args.sample_bootstrap,
                    "seed": args.seed + seed_offset,
                    "interpretation": interpretation(metric, low, high),
                })

    pooled = concatenate_predictions([external["test1"], external["test2"]])
    append_metric_rows(
        metric_rows, "test1_plus_test2", "all", len(pooled["truth"]),
        pooled, "supplementary_pooled")
    for comparison, left, right in (
            ("real_minus_sequence", "real", "sequence"),
            ("real_minus_shuffled", "real", "shuffled")):
        for metric in ("MAE", "Pearson"):
            seed_offset += 1
            point = metric_delta(
                pooled["truth"], pooled[left], pooled[right], metric)
            low, high = paired_bootstrap(
                pooled["truth"], pooled[left], pooled[right],
                args.sample_bootstrap, args.seed + seed_offset, metric)
            statistical_rows.append({
                "dataset": "test1_plus_test2",
                "subset": "all",
                "comparison": comparison,
                "metric": metric,
                "point_delta": point,
                "ci95_low": low,
                "ci95_high": high,
                "resampling_unit": "sample",
                "n_resamples": args.sample_bootstrap,
                "seed": args.seed + seed_offset,
                "interpretation": interpretation(metric, low, high),
            })

    s166 = s166_predictions(root)
    subset_masks = {
        "all": np.ones(len(s166["truth"]), dtype=bool),
        "wild_type": np.asarray([value == "wild" for value in s166["subsets"]]),
        "mutant": np.asarray([value == "mutant" for value in s166["subsets"]]),
        "without_1YCS": np.asarray([value != "1YCS" for value in s166["groups"]]),
    }
    for subset, mask in subset_masks.items():
        predictions = subset_predictions(s166, mask)
        dataset = "S166_without_1YCS" if subset == "without_1YCS" else "S166"
        output_subset = "all" if subset == "without_1YCS" else subset
        append_metric_rows(
            metric_rows, dataset, output_subset,
            len(set(predictions["groups"])), predictions,
            "sensitivity" if subset == "without_1YCS"
            else "supportive_external")
        for comparison, left, right in (
                ("real_minus_sequence", "real", "sequence"),
                ("real_minus_shuffled", "real", "shuffled")):
            for metric in ("MAE", "Pearson"):
                seed_offset += 1
                point = metric_delta(
                    predictions["truth"], predictions[left],
                    predictions[right], metric)
                low, high = cluster_bootstrap(
                    predictions["truth"], predictions[left],
                    predictions[right], predictions["groups"],
                    args.cluster_bootstrap, args.seed + seed_offset, metric)
                statistical_rows.append({
                    "dataset": dataset,
                    "subset": output_subset,
                    "comparison": comparison,
                    "metric": metric,
                    "point_delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "resampling_unit": "base_pdb_id",
                    "n_resamples": args.cluster_bootstrap,
                    "seed": args.seed + seed_offset,
                    "interpretation": interpretation(metric, low, high),
                })

    out = Path(args.output_dir)
    write_csv(out / "main_metrics.csv", metric_rows)
    write_csv(out / "statistical_checks.csv", statistical_rows)
    protocol = {
        "seed": args.seed,
        "sample_bootstrap": args.sample_bootstrap,
        "cluster_bootstrap": args.cluster_bootstrap,
        "sample_unit_datasets": ["validation", "test1", "test2"],
        "S166_cluster_unit": "base_pdb_id",
        "snapshot_root": str(root.resolve()),
    }
    (out / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    print("wrote {} metric rows and {} statistical rows to {}".format(
        len(metric_rows), len(statistical_rows), out.resolve()))


if __name__ == "__main__":
    main()
