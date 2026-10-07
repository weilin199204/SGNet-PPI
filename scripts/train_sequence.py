#!/usr/bin/env python
"""Train the SGNet sequence baseline on the official fixed splits."""
from __future__ import absolute_import, print_function

import argparse
import csv
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgnet.data import validate_affinity_samples
from sgnet.sequence import FrozenSequenceEncoder

SPLITS = ("train", "val", "test1", "test2")
EXPECTED = {"train": 2350, "val": 80, "test1": 79, "test2": 81}


class SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return index, self.samples[index]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
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
    all_ids = [sample["pdb_id"] for split in SPLITS for sample in result[split]]
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
        "target": Batch.from_data_list([sample["target"] for sample in samples]),
    }


def make_loader(samples, batch_size, shuffle, seed):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        SampleDataset(samples), batch_size=batch_size, shuffle=shuffle,
        generator=generator, num_workers=0, collate_fn=collate_sequence)


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


def cpu_state(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def atomic_torch_save(value, path):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def optimizer_step(loss, step, n_steps, accumulation, optimizer):
    (loss / accumulation).backward()
    if (step + 1) % accumulation == 0 or step + 1 == n_steps:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def train_sequence_epoch(model, loader, optimizer, accumulation, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    for step, batch in enumerate(loader):
        sequence_a = batch["sequence_a"].to(device)
        sequence_b = batch["sequence_b"].to(device)
        target = batch["target"].to(device).y.view(-1, 1)
        prediction = model(sequence_a, sequence_b)
        loss = F.mse_loss(prediction, target)
        optimizer_step(loss, step, len(loader), accumulation, optimizer)
        total += float(loss.detach()) * target.shape[0]
        count += int(target.shape[0])
    return total / max(count, 1)



def evaluate_sequence(model, loader, device):
    model.eval()
    truth, prediction, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device).y.view(-1)
            pred = model(batch["sequence_a"].to(device),
                         batch["sequence_b"].to(device)).view(-1)
            truth.extend(target.cpu().tolist())
            prediction.extend(pred.cpu().tolist())
            ids.extend(batch["pdb_ids"])
    return {"pdb_ids": ids, "truth": np.asarray(truth),
            "prediction": np.asarray(prediction),
            "metrics": metrics(truth, prediction)}



def fit_sequence(splits, args, device, esm_dim):
    set_seed(args.seed)
    model = FrozenSequenceEncoder(
        esm_dim, args.hidden_channels, args.head_channels,
        args.dropout).to(device)
    train_loader = make_loader(splits["train"], args.batch_size, True,
                               args.seed + 101)
    val_loader = make_loader(splits["val"], args.batch_size, False,
                             args.seed + 102)
    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    best_mse, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_mse = train_sequence_epoch(
            model, train_loader, optimizer, args.accumulation_steps, device)
        result = evaluate_sequence(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_metrics": result["metrics"]})
        print("sequence epoch {} train={:.4f} val={:.4f} MAE={:.4f} R={:.4f}".format(
            epoch, train_mse, result["metrics"]["mse"],
            result["metrics"]["mae"], result["metrics"]["pearson"]), flush=True)
        if result["metrics"]["mse"] < best_mse - args.min_delta:
            best_mse, best_epoch = result["metrics"]["mse"], epoch
            best_state, wait = cpu_state(model), 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("sequence training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    return model, best_state, best_epoch, history



def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    splits = load_splits(args.data_root)
    validation = {split: validate_affinity_samples(samples)
                  for split, samples in splits.items()}
    esm_dim = int(splits["train"][0]["sequence_a"].x.shape[1])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, state, best_epoch, history = fit_sequence(
        splits, args, device, esm_dim)
    config = {
        "esm_channels": esm_dim,
        "hidden_channels": args.hidden_channels,
        "head_channels": args.head_channels,
        "dropout": args.dropout,
    }
    atomic_torch_save({
        "student_state": state,
        "model_config": config,
        "best_epoch": best_epoch,
        "selection_split": "val",
    }, out / "best_sequence.pt")

    evaluations = {}
    for split in ("val", "test1", "test2"):
        loader = make_loader(
            splits[split], args.batch_size, False, args.seed + 300)
        result = evaluate_sequence(model, loader, device)
        evaluations[split] = result["metrics"]
        with (out / (split + "_predictions.csv")).open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("pdb_id", "y_true", "prediction"))
            writer.writerows(zip(
                result["pdb_ids"], result["truth"], result["prediction"]))
        print("{} MAE={:.4f} R={:.4f}".format(
            split, result["metrics"]["mae"],
            result["metrics"]["pearson"]), flush=True)

    params = vars(args).copy()
    params.update({
        "data_root": str(Path(args.data_root).resolve()),
        "output_dir": str(out.resolve()),
        "device_resolved": str(device),
        "split_policy": "official SSIF lists: 2350/80/79/81",
        "test_used_for_selection": False,
    })
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    summary = {
        "best_epoch": best_epoch,
        "validation": validation,
        "evaluation": evaluations,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
