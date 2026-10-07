#!/usr/bin/env python
"""Train SGNet on the official fixed splits."""
from __future__ import absolute_import, print_function

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgnet.data import (
    compute_surface_imputation_stats, materialize_surface_patch,
    surface_is_available, surface_model_patch, validate_affinity_samples)
from sgnet.sequence import FrozenSequenceEncoder
from sgnet.model import (
    SurfaceResidualV2, compute_cross_surface_normalization)


SPLITS = ("train", "val", "test1", "test2")
EXPECTED = {"train": 2350, "val": 80, "test1": 79, "test2": 81}


class IndexedDataset(Dataset):
    def __init__(self, samples, indices=None):
        self.samples = samples
        self.indices = (list(range(len(samples))) if indices is None
                        else [int(index) for index in indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return index, self.samples[index]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequence-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--oof-epochs", type=int, default=0,
                        help="0 uses best_epoch stored in the full sequence checkpoint")
    parser.add_argument("--oof-lr", type=float, default=3e-4)
    parser.add_argument("--sequence-weight-decay", type=float, default=1e-3)
    parser.add_argument("--reuse-oof", default=None,
                        help="Previously generated oof_sequence_predictions.csv")
    parser.add_argument("--surface-epochs", type=int, default=100)
    parser.add_argument("--surface-patience", type=int, default=20)
    parser.add_argument("--surface-lr", type=float, default=1e-4)
    parser.add_argument("--surface-weight-decay", type=float, default=1e-2)
    parser.add_argument("--surface-hidden", type=int, default=48)
    parser.add_argument("--graph-hidden", type=int, default=32)
    parser.add_argument("--surface-dropout", type=float, default=0.35)
    parser.add_argument("--initial-reliability", type=float, default=0.15)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--correction-penalty", type=float, default=0.02)
    parser.add_argument("--gate-penalty", type=float, default=0.002)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--target-mode", choices=("real", "shuffled", "both"),
                        default="both")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value):
    value = str(value).replace("\uff1a", ":")
    if value.isdigit():
        value = "cuda:" + value
    if value.startswith("cuda") and not torch.cuda.is_available():
        value = "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    return device


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(value, path):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cpu_state(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_splits(root):
    result = {}
    for split in SPLITS:
        path = Path(root) / (split + "_dataset_save.pkl")
        with path.open("rb") as handle:
            result[split] = pickle.load(handle)
        if len(result[split]) != EXPECTED[split]:
            raise ValueError("{} has {} samples, expected {}".format(
                split, len(result[split]), EXPECTED[split]))
        print("loaded {} {} samples".format(len(result[split]), split), flush=True)
    all_ids = [str(sample["pdb_id"]).upper()
               for split in SPLITS for sample in result[split]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("official splits overlap by PDB ID")
    return result


def collate_sequence(items):
    indices, samples = zip(*items)
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "pdb_ids": [sample["pdb_id"] for sample in samples],
        "sequence_a": Batch.from_data_list([sample["sequence_a"] for sample in samples]),
        "sequence_b": Batch.from_data_list([sample["sequence_b"] for sample in samples]),
        "target": torch.tensor(
            [float(sample["target"].y.reshape(-1)[0]) for sample in samples],
            dtype=torch.float32),
    }


def collate_surface(items, surface_stats):
    indices, samples = zip(*items)
    patches = []
    for sample in samples:
        patch = materialize_surface_patch(
            sample["patch"], sample["sequence_a"].num_nodes,
            sample["sequence_b"].num_nodes, surface_stats)
        patches.append(surface_model_patch(patch))
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "pdb_ids": [sample["pdb_id"] for sample in samples],
        "patch": Batch.from_data_list(patches),
        "target": torch.tensor(
            [float(sample["target"].y.reshape(-1)[0]) for sample in samples],
            dtype=torch.float32),
    }


def make_sequence_loader(samples, indices, batch_size, shuffle, seed):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        IndexedDataset(samples, indices), batch_size=batch_size,
        shuffle=shuffle, generator=generator, num_workers=0,
        collate_fn=collate_sequence)


def make_surface_loader(samples, indices, batch_size, shuffle, seed, surface_stats):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        IndexedDataset(samples, indices), batch_size=batch_size,
        shuffle=shuffle, generator=generator, num_workers=0,
        collate_fn=partial(collate_surface, surface_stats=surface_stats))


def metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    pearson = (float(np.corrcoef(prediction, truth)[0, 1])
               if len(truth) > 1 and prediction.std() > 0 and truth.std() > 0
               else float("nan"))
    return {
        "n": int(len(truth)),
        "mse": float(np.mean((prediction - truth) ** 2)),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "pearson": pearson,
        "spearman": float(spearmanr(prediction, truth).correlation),
    }


def optimizer_step(loss, step, n_steps, accumulation, optimizer, model, clip):
    (loss / accumulation).backward()
    if (step + 1) % accumulation == 0 or step + 1 == n_steps:
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def sequence_model(checkpoint, esm_dim, device, seed):
    config = checkpoint.get("model_config", {})
    set_seed(seed)
    model = FrozenSequenceEncoder(
        in_channels=esm_dim,
        hidden_channels=int(config.get("hidden_channels", 256)),
        head_channels=int(config.get("head_channels", 256)),
        dropout=float(config.get("dropout", 0.4)))
    return model.to(device), config


def train_sequence_epoch(model, loader, optimizer, accumulation, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    for step, batch in enumerate(loader):
        target = batch["target"].to(device).view(-1, 1)
        prediction = model(batch["sequence_a"].to(device),
                           batch["sequence_b"].to(device))
        loss = F.mse_loss(prediction, target)
        optimizer_step(loss, step, len(loader), accumulation, optimizer,
                       model, 0.0)
        total += float(loss.detach()) * target.shape[0]
        count += int(target.shape[0])
    return total / max(count, 1)


def predict_sequence(model, loader, device):
    model.eval()
    result = {}
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["sequence_a"].to(device),
                               batch["sequence_b"].to(device)).view(-1)
            for index, value in zip(batch["indices"].tolist(),
                                    prediction.cpu().tolist()):
                result[int(index)] = float(value)
    return result


def write_oof(path, samples, folds, truth, prediction):
    with path.open("w", newline="") as handle:
        fields = ("sample_index", "pdb_id", "fold", "y_true", "y_pred",
                  "oof_residual")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow({
                "sample_index": index,
                "pdb_id": sample["pdb_id"],
                "fold": int(folds[index]),
                "y_true": float(truth[index]),
                "y_pred": float(prediction[index]),
                "oof_residual": float(truth[index] - prediction[index]),
            })


def load_oof(path, samples):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(samples):
        raise ValueError("OOF row count does not match training set")
    prediction = np.full(len(samples), np.nan, dtype=np.float32)
    folds = np.zeros(len(samples), dtype=np.int64)
    for row in rows:
        index = int(row["sample_index"])
        sample = samples[index]
        if str(row["pdb_id"]).upper() != str(sample["pdb_id"]).upper():
            raise ValueError("OOF PDB mismatch at index {}".format(index))
        truth = float(sample["target"].y.reshape(-1)[0])
        if not np.isclose(float(row["y_true"]), truth, atol=1e-5, rtol=0):
            raise ValueError("OOF label mismatch at index {}".format(index))
        prediction[index] = float(row["y_pred"])
        folds[index] = int(row["fold"])
    if not np.isfinite(prediction).all() or np.any(folds < 1):
        raise ValueError("OOF predictions are incomplete")
    return prediction, folds


def build_oof_predictions(samples, checkpoint, args, device, esm_dim, out):
    n = len(samples)
    truth = np.asarray([
        float(sample["target"].y.reshape(-1)[0]) for sample in samples],
        dtype=np.float32)
    prediction = np.full(n, np.nan, dtype=np.float32)
    fold_assignment = np.zeros(n, dtype=np.int64)
    splitter = KFold(n_splits=args.oof_folds, shuffle=True,
                     random_state=args.seed)
    epochs = (int(args.oof_epochs) if args.oof_epochs > 0
              else int(checkpoint["best_epoch"]))
    if epochs < 1:
        raise ValueError("OOF epoch count must be positive")
    oof_dir = out / "oof_models"
    oof_dir.mkdir(parents=True, exist_ok=True)
    fold_summary = []

    for fold_index, (train_index, heldout_index) in enumerate(
            splitter.split(np.arange(n)), 1):
        fold_seed = args.seed + 10000 + fold_index
        model, config = sequence_model(
            checkpoint, esm_dim, device, fold_seed)
        train_loader = make_sequence_loader(
            samples, train_index, args.batch_size, True, fold_seed + 100)
        heldout_loader = make_sequence_loader(
            samples, heldout_index, args.batch_size, False, fold_seed + 200)
        optimizer = Adam(model.parameters(), lr=args.oof_lr,
                         weight_decay=args.sequence_weight_decay)
        history = []
        for epoch in range(1, epochs + 1):
            train_mse = train_sequence_epoch(
                model, train_loader, optimizer, args.accumulation_steps, device)
            history.append({"epoch": epoch, "train_mse": train_mse})
            if epoch == 1 or epoch == epochs or epoch % 5 == 0:
                print("OOF fold {}/{} epoch {}/{} train={:.4f}".format(
                    fold_index, args.oof_folds, epoch, epochs, train_mse),
                    flush=True)
        heldout_predictions = predict_sequence(model, heldout_loader, device)
        for index in heldout_index:
            prediction[index] = heldout_predictions[int(index)]
            fold_assignment[index] = fold_index
        fold_metrics = metrics(truth[heldout_index], prediction[heldout_index])
        fold_summary.append({
            "fold": fold_index, "seed": fold_seed,
            "train_n": int(len(train_index)),
            "heldout_n": int(len(heldout_index)),
            "fixed_epochs": epochs,
            "heldout_metrics": fold_metrics,
        })
        atomic_torch_save({
            "student_state": cpu_state(model), "model_config": config,
            "fold": fold_index, "fixed_epochs": epochs,
            "training_indices": train_index,
            "heldout_indices": heldout_index,
            "heldout_used_for_training_or_selection": False,
        }, oof_dir / "fold_{}.pt".format(fold_index))
        (oof_dir / "fold_{}_history.json".format(fold_index)).write_text(
            json.dumps(history, indent=2) + "\n")
        print("OOF fold {} MAE={:.4f} R={:.4f}".format(
            fold_index, fold_metrics["mae"], fold_metrics["pearson"]), flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(prediction).all() or np.any(fold_assignment < 1):
        raise RuntimeError("OOF generation did not cover every training sample")
    oof_path = out / "oof_sequence_predictions.csv"
    write_oof(oof_path, samples, fold_assignment, truth, prediction)
    summary = {
        "protocol": "seed-20 random 5-fold cross-fitting",
        "fixed_epochs": epochs,
        "epoch_source": "best_epoch of locked full-training sequence checkpoint",
        "heldout_used_for_training_or_selection": False,
        "pooled_metrics": metrics(truth, prediction),
        "residual_mean": float(np.mean(truth - prediction)),
        "residual_std": float(np.std(truth - prediction)),
        "folds": fold_summary,
    }
    (out / "oof_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return prediction, fold_assignment, oof_path, summary


def full_sequence_predictions(samples, checkpoint, args, device, esm_dim):
    model, _ = sequence_model(checkpoint, esm_dim, device, args.seed)
    model.load_state_dict(checkpoint["student_state"], strict=True)
    loader = make_sequence_loader(
        samples, range(len(samples)), args.batch_size, False, args.seed + 700)
    mapping = predict_sequence(model, loader, device)
    prediction = np.asarray([mapping[index] for index in range(len(samples))],
                            dtype=np.float32)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction


def train_adapter_epoch(model, loader, optimizer, baseline, residual_target,
                        residual_scale, args, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "data": 0.0, "correction": 0.0, "gate": 0.0}
    count = 0
    for step, batch in enumerate(loader):
        indices = batch["indices"].numpy()
        base = torch.from_numpy(baseline[indices]).to(device).view(-1, 1)
        target = torch.from_numpy(residual_target[indices]).to(device).view(-1, 1)
        _, correction, _, gate, _ = model.forward_features(
            batch["patch"].to(device), base)
        standardized_correction = correction / residual_scale
        standardized_target = target / residual_scale
        data_loss = F.smooth_l1_loss(
            standardized_correction, standardized_target, beta=args.huber_beta)
        correction_loss = standardized_correction.square().mean()
        gate_loss = gate.square().mean()
        loss = (data_loss + args.correction_penalty * correction_loss
                + args.gate_penalty * gate_loss)
        optimizer_step(loss, step, len(loader), args.accumulation_steps,
                       optimizer, model, args.gradient_clip)
        batch_n = len(indices)
        totals["loss"] += float(loss.detach()) * batch_n
        totals["data"] += float(data_loss.detach()) * batch_n
        totals["correction"] += float(correction_loss.detach()) * batch_n
        totals["gate"] += float(gate_loss.detach()) * batch_n
        count += batch_n
    return {name: value / max(count, 1) for name, value in totals.items()}


def evaluate_adapter(model, loader, baseline, device):
    model.eval()
    truth, prediction, correction, raw, reliability = [], [], [], [], []
    pdb_ids, availability = [], []
    with torch.no_grad():
        for batch in loader:
            indices = batch["indices"].numpy()
            base = torch.from_numpy(baseline[indices]).to(device).view(-1, 1)
            pred, corr, raw_value, gate, _ = model.forward_features(
                batch["patch"].to(device), base)
            truth.extend(batch["target"].numpy().tolist())
            prediction.extend(pred.view(-1).cpu().tolist())
            correction.extend(corr.view(-1).cpu().tolist())
            raw.extend(raw_value.view(-1).cpu().tolist())
            reliability.extend(gate.view(-1).cpu().tolist())
            pdb_ids.extend(batch["pdb_ids"])
            availability.extend(
                batch["patch"].surface_available.reshape(-1).bool().cpu().tolist())
    truth = np.asarray(truth, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    correction = np.asarray(correction, dtype=np.float32)
    reliability = np.asarray(reliability, dtype=np.float32)
    return {
        "pdb_ids": pdb_ids,
        "truth": truth,
        "baseline": np.asarray(baseline, dtype=np.float32),
        "prediction": prediction,
        "correction": correction,
        "raw_residual": np.asarray(raw, dtype=np.float32),
        "reliability": reliability,
        "surface_available": np.asarray(availability, dtype=bool),
        "baseline_metrics": metrics(truth, baseline),
        "metrics": metrics(truth, prediction),
        "gate": {
            "mean": float(reliability.mean()),
            "std": float(reliability.std()),
            "q05": float(np.quantile(reliability, 0.05)),
            "q50": float(np.quantile(reliability, 0.50)),
            "q95": float(np.quantile(reliability, 0.95)),
        },
        "correction_mean": float(correction.mean()),
        "correction_std": float(correction.std()),
    }


def write_predictions(path, result):
    with path.open("w", newline="") as handle:
        fields = ("pdb_id", "y_true", "sequence_prediction",
                  "surface_prediction", "surface_correction",
                  "raw_surface_residual", "reliability_gate",
                  "surface_available")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in zip(
                result["pdb_ids"], result["truth"], result["baseline"],
                result["prediction"], result["correction"],
                result["raw_residual"], result["reliability"],
                result["surface_available"]):
            writer.writerow(dict(zip(fields, [
                values[0], float(values[1]), float(values[2]), float(values[3]),
                float(values[4]), float(values[5]), float(values[6]),
                bool(values[7])])))


def acceptance(baseline_metrics, surface_metrics):
    delta_mae = surface_metrics["mae"] - baseline_metrics["mae"]
    delta_pearson = surface_metrics["pearson"] - baseline_metrics["pearson"]
    passed = ((delta_mae <= -0.03 and delta_pearson >= -0.01)
              or (delta_pearson >= 0.015 and delta_mae <= 0.01))
    return {
        "passed": bool(passed),
        "criterion": "dMAE<=-0.03 or dPearson>=0.015; other metric may worsen at most 0.01",
        "delta_mae": float(delta_mae),
        "delta_pearson": float(delta_pearson),
    }


def fit_adapter(mode, splits, oof_prediction, final_baselines, surface_stats,
                normalization, args, device, out):
    mode_out = out / mode
    mode_out.mkdir(parents=True, exist_ok=True)
    truth = np.asarray([
        float(sample["target"].y.reshape(-1)[0]) for sample in splits["train"]],
        dtype=np.float32)
    residual = truth - oof_prediction
    if mode == "shuffled":
        residual_target = residual[
            np.random.default_rng(args.seed + 50000).permutation(len(residual))]
    else:
        residual_target = residual.copy()
    residual_scale = max(float(np.std(residual)), 1e-4)

    set_seed(args.seed + 20000)
    model = SurfaceResidualV2(
        normalization=normalization,
        baseline_mean=float(np.mean(oof_prediction)),
        baseline_std=float(np.std(oof_prediction)),
        hidden_channels=args.surface_hidden,
        graph_channels=args.graph_hidden,
        dropout=args.surface_dropout,
        initial_reliability=args.initial_reliability).to(device)
    initial_digest = state_digest(model)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    train_loader = make_surface_loader(
        splits["train"], range(len(splits["train"])), args.batch_size, True,
        args.seed + 20100, surface_stats)
    val_loader = make_surface_loader(
        splits["val"], range(len(splits["val"])), args.batch_size, False,
        args.seed + 20200, surface_stats)

    initial = evaluate_adapter(
        model, val_loader, final_baselines["val"], device)
    if not np.array_equal(initial["prediction"], initial["baseline"]):
        raise AssertionError("zero-initialized v2 does not exactly reproduce baseline")
    best_mse = initial["metrics"]["mse"]
    best_epoch, best_state, wait = 0, cpu_state(model), 0
    history = [{
        "epoch": 0, "train": None, "val_metrics": initial["metrics"],
        "baseline_metrics": initial["baseline_metrics"], "gate": initial["gate"],
    }]
    optimizer = AdamW(model.parameters(), lr=args.surface_lr,
                      weight_decay=args.surface_weight_decay)

    for epoch in range(1, args.surface_epochs + 1):
        train_result = train_adapter_epoch(
            model, train_loader, optimizer, oof_prediction, residual_target,
            residual_scale, args, device)
        val_result = evaluate_adapter(
            model, val_loader, final_baselines["val"], device)
        history.append({
            "epoch": epoch, "train": train_result,
            "val_metrics": val_result["metrics"],
            "baseline_metrics": val_result["baseline_metrics"],
            "gate": val_result["gate"],
            "correction_std": val_result["correction_std"],
        })
        print("{} epoch {} train={:.4f} val={:.4f} MAE={:.4f} R={:.4f} "
              "gate={:.3f} corr_sd={:.3f}".format(
                  mode, epoch, train_result["loss"],
                  val_result["metrics"]["mse"], val_result["metrics"]["mae"],
                  val_result["metrics"]["pearson"], val_result["gate"]["mean"],
                  val_result["correction_std"]), flush=True)
        if val_result["metrics"]["mse"] < best_mse - args.min_delta:
            best_mse, best_epoch = val_result["metrics"]["mse"], epoch
            best_state, wait = cpu_state(model), 0
        else:
            wait += 1
            if wait >= args.surface_patience:
                break

    model.load_state_dict(best_state, strict=True)
    evaluations = {}
    for split in ("val", "test1", "test2"):
        loader = make_surface_loader(
            splits[split], range(len(splits[split])), args.batch_size, False,
            args.seed + 20300, surface_stats)
        result = evaluate_adapter(
            model, loader, final_baselines[split], device)
        write_predictions(mode_out / (split + "_predictions.csv"), result)
        evaluations[split] = {
            "baseline": result["baseline_metrics"],
            "surface_residual_v2": result["metrics"],
            "delta_mae": (result["metrics"]["mae"]
                          - result["baseline_metrics"]["mae"]),
            "delta_pearson": (result["metrics"]["pearson"]
                              - result["baseline_metrics"]["pearson"]),
            "gate": result["gate"],
            "correction_mean": result["correction_mean"],
            "correction_std": result["correction_std"],
        }

    selection = acceptance(
        evaluations["val"]["baseline"],
        evaluations["val"]["surface_residual_v2"])
    checkpoint_payload = {
        "model_state": best_state,
        "model_config": {
            "surface_hidden": args.surface_hidden,
            "graph_hidden": args.graph_hidden,
            "surface_dropout": args.surface_dropout,
            "initial_reliability": args.initial_reliability,
        },
        "surface_normalization": normalization,
        "surface_imputation_stats": surface_stats,
        "baseline_mean": float(np.mean(oof_prediction)),
        "baseline_std": float(np.std(oof_prediction)),
        "oof_residual_mean": float(np.mean(residual)),
        "oof_residual_std": residual_scale,
        "best_epoch": best_epoch,
        "target_mode": mode,
    }
    atomic_torch_save(checkpoint_payload, mode_out / "best_surface_residual_v2.pt")
    (mode_out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    summary = {
        "protocol": "surface_residual_v2",
        "target_mode": mode,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "initial_state_sha256": initial_digest,
        "cross_edge_only": True,
        "intra_surface_edges_consumed": False,
        "oof_residual_mean": float(np.mean(residual)),
        "oof_residual_std": residual_scale,
        "val_acceptance_gate": selection,
        "evaluation": evaluations,
    }
    (mode_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("{} summary:\n{}".format(mode, json.dumps(summary, indent=2)), flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.data_root)
    validation = {split: validate_affinity_samples(samples)
                  for split, samples in splits.items()}
    esm_dim = int(splits["train"][0]["sequence_a"].x.shape[1])
    checkpoint = torch_load(args.sequence_checkpoint)
    if "student_state" not in checkpoint:
        raise ValueError("sequence checkpoint is missing student_state")

    if args.reuse_oof:
        oof_prediction, oof_folds = load_oof(
            args.reuse_oof, splits["train"])
        oof_path = Path(args.reuse_oof).resolve()
        oof_summary = {
            "protocol": "reused verified OOF predictions",
            "source": str(oof_path),
            "pooled_metrics": metrics([
                float(sample["target"].y.reshape(-1)[0])
                for sample in splits["train"]], oof_prediction),
        }
    else:
        oof_prediction, oof_folds, oof_path, oof_summary = (
            build_oof_predictions(
                splits["train"], checkpoint, args, device, esm_dim, out))

    final_baselines = {
        split: full_sequence_predictions(
            splits[split], checkpoint, args, device, esm_dim)
        for split in ("val", "test1", "test2")
    }
    surface_stats = compute_surface_imputation_stats(
        splits["train"], range(len(splits["train"])))
    normalization = compute_cross_surface_normalization(
        splits["train"], range(len(splits["train"])), surface_stats)
    (out / "surface_imputation_stats.json").write_text(
        json.dumps(surface_stats, indent=2) + "\n")
    (out / "surface_normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n")

    modes = ("real", "shuffled") if args.target_mode == "both" else (args.target_mode,)
    summaries = {}
    for mode in modes:
        summaries[mode] = fit_adapter(
            mode, splits, oof_prediction, final_baselines, surface_stats,
            normalization, args, device, out)

    comparison = {
        "protocol": {
            "name": "surface_residual_v2",
            "split": "official SSIF fixed 2350/80/79/81",
            "seed": args.seed,
            "oof": "random seed-20 5-fold cross-fitting inside train only",
            "surface_normalization": "training split only",
            "encoder": "one-layer cross-edge-only patch encoder",
            "selection": "official val only; test1/test2 locked",
            "sequence_checkpoint": str(Path(args.sequence_checkpoint).resolve()),
            "oof_predictions": str(oof_path),
        },
        "validation": validation,
        "oof_summary": oof_summary,
        "arms": summaries,
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    params = vars(args).copy()
    params.update({
        "data_root": str(Path(args.data_root).resolve()),
        "sequence_checkpoint": str(Path(args.sequence_checkpoint).resolve()),
        "output_dir": str(out.resolve()),
        "device_resolved": str(device),
        "test_used_for_selection": False,
        "scheduler": None,
    })
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print("Surface Residual V2 complete: {}".format(out), flush=True)


if __name__ == "__main__":
    main()
