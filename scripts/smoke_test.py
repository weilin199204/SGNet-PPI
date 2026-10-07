#!/usr/bin/env python
"""Run fast architecture and pretrained-checkpoint checks on synthetic data."""
from __future__ import absolute_import, print_function

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from sgnet.data import compute_surface_imputation_stats
from sgnet.model import SurfaceResidualV2, compute_cross_surface_normalization
from sgnet.sequence import FrozenSequenceEncoder
from train_sgnet import collate_sequence, collate_surface, torch_load


def synthetic_patch(generator, n_a, n_b, available=True):
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 2, 1, 3],
        [1, 0, 3, 2, 2, 0, 3, 1],
    ], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return Data(
        x=torch.randn((4, 23), generator=generator),
        edge_index=edge_index,
        edge_attr=torch.randn((8, 42), generator=generator),
        patch_area=torch.rand((4, 1), generator=generator) + 0.5,
        chain_id=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_type=edge_type,
        residue_ids=torch.full((4, 3), -1, dtype=torch.long),
        residue_weights=torch.zeros((4, 3), dtype=torch.float32),
        residue_counts=torch.tensor([[n_a, n_b]], dtype=torch.long),
        patch_graph_version=torch.tensor([5], dtype=torch.long),
        surface_available=torch.tensor([available], dtype=torch.bool),
    )


def synthetic_samples():
    generator = torch.Generator().manual_seed(20)
    samples = []
    for index, (n_a, n_b) in enumerate(((5, 7), (6, 4), (8, 5), (4, 6))):
        samples.append({
            "pdb_id": "SYN{:02d}".format(index),
            "sequence_a": Data(x=torch.randn((n_a, 1280), generator=generator)),
            "sequence_b": Data(x=torch.randn((n_b, 1280), generator=generator)),
            "target": Data(
                x=torch.empty((n_a + n_b, 0), dtype=torch.float32),
                y=torch.tensor([6.0 + index], dtype=torch.float32)),
            "patch": synthetic_patch(generator, n_a, n_b),
        })
    return samples


def architecture_checks(samples):
    indices = list(range(len(samples)))
    imputation = compute_surface_imputation_stats(samples, indices)
    normalization = compute_cross_surface_normalization(
        samples, indices, imputation)
    batch = collate_surface(
        [(index, sample) for index, sample in enumerate(samples)], imputation)
    patch = batch["patch"]
    baseline = batch["target"].view(-1, 1)
    model = SurfaceResidualV2(
        normalization, float(baseline.mean()), float(baseline.std()),
        hidden_channels=48, graph_channels=32, dropout=0.35)
    model.eval()
    with torch.no_grad():
        prediction, correction, _, _, _ = model.forward_features(patch, baseline)
        if not torch.equal(prediction, baseline):
            raise AssertionError("zero initialization does not reproduce baseline")
        if not torch.equal(correction, torch.zeros_like(correction)):
            raise AssertionError("initial correction is not exactly zero")
        graph_before = model.surface(patch)[0]
        changed = patch.clone()
        intra = changed.edge_type.long() == 0
        changed.edge_attr[intra] = 10000.0 * torch.randn_like(
            changed.edge_attr[intra])
        graph_after = model.surface(changed)[0]
        if not torch.equal(graph_before, graph_after):
            raise AssertionError("intra-edge attributes affect cross-only encoder")
    return batch


def checkpoint_checks(batch, sequence_batch):
    sequence_checkpoint = torch_load(ROOT / "model" / "best_sequence.pt")
    sequence_config = sequence_checkpoint["model_config"]
    sequence = FrozenSequenceEncoder(
        int(sequence_config["esm_channels"]),
        int(sequence_config["hidden_channels"]),
        int(sequence_config["head_channels"]),
        float(sequence_config["dropout"]))
    sequence.load_state_dict(sequence_checkpoint["student_state"], strict=True)
    sequence.eval()
    with torch.no_grad():
        baseline = sequence(sequence_batch["sequence_a"], sequence_batch["sequence_b"])

    checkpoint = torch_load(
        ROOT / "model" / "real" / "best_surface_residual_v2.pt")
    config = checkpoint["model_config"]
    model = SurfaceResidualV2(
        normalization=checkpoint["surface_normalization"],
        baseline_mean=checkpoint["baseline_mean"],
        baseline_std=checkpoint["baseline_std"],
        hidden_channels=int(config["surface_hidden"]),
        graph_channels=int(config["graph_hidden"]),
        dropout=float(config["surface_dropout"]),
        initial_reliability=float(config["initial_reliability"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(batch["patch"], baseline)
    if prediction.shape != (len(batch["pdb_ids"]), 1):
        raise AssertionError("unexpected real output shape")
    if not bool(torch.isfinite(prediction).all()):
        raise AssertionError("real checkpoint produced non-finite output")


def main():
    samples = synthetic_samples()
    batch = architecture_checks(samples)
    sequence_batch = collate_sequence(
        [(index, sample) for index, sample in enumerate(samples)])
    checkpoint_checks(batch, sequence_batch)
    print("smoke_ok=True")
    print("synthetic_complexes={}".format(len(samples)))
    print("cross_edge_only=True")
    print("zero_initialization_exact=True")
    print("pretrained_checkpoints_loaded=2")


if __name__ == "__main__":
    main()
