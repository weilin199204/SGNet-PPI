"""Low-capacity cross-surface residual model with train-only normalization."""
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
from torch_scatter import scatter

from .data import materialize_surface_patch, surface_is_available


PATCH_GRAPH_VERSION = 5
PATCH_NODE_DIM = 23
PATCH_EDGE_DIM = 42


def _batch(data):
    value = getattr(data, "batch", None)
    if value is None:
        return torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
    return value.long()


def _num_graphs(batch):
    return int(batch.max().item()) + 1 if batch.numel() else 0


def _moments(sum_value, square_sum, weight, minimum_std=1e-4):
    mean = sum_value / max(float(weight), 1.0)
    variance = square_sum / max(float(weight), 1.0) - mean.square()
    return mean, variance.clamp_min(float(minimum_std) ** 2).sqrt()


def compute_cross_surface_normalization(samples, indices, imputation_stats):
    """Fit scaling on training complexes and cross-edge support only."""
    node_sum = torch.zeros(PATCH_NODE_DIM, dtype=torch.float64)
    node_square = torch.zeros(PATCH_NODE_DIM, dtype=torch.float64)
    node_weight = 0.0
    edge_sum = torch.zeros(PATCH_EDGE_DIM, dtype=torch.float64)
    edge_square = torch.zeros(PATCH_EDGE_DIM, dtype=torch.float64)
    edge_count = 0
    contexts = []
    available = 0

    for raw_index in indices:
        sample = samples[int(raw_index)]
        original = sample["patch"]
        patch = materialize_surface_patch(
            original, sample["sequence_a"].num_nodes,
            sample["sequence_b"].num_nodes, imputation_stats)
        cross = patch.edge_type.detach().cpu().long() == 1
        if not bool(cross.any()):
            raise ValueError("{} has no cross surface edge".format(sample["pdb_id"]))
        edge_index = patch.edge_index.detach().cpu().long()[:, cross]
        incident = torch.unique(edge_index.reshape(-1))
        x = torch.nan_to_num(patch.x.detach().cpu().double()[incident])
        area = torch.nan_to_num(
            patch.patch_area.detach().cpu().double().reshape(-1)[incident]
        ).clamp_min(1e-8)
        node_sum += (x * area[:, None]).sum(dim=0)
        node_square += (x.square() * area[:, None]).sum(dim=0)
        node_weight += float(area.sum())

        edge = torch.nan_to_num(patch.edge_attr.detach().cpu().double()[cross])
        edge_sum += edge.sum(dim=0)
        edge_square += edge.square().sum(dim=0)
        edge_count += int(edge.shape[0])
        contexts.append((math.log1p(float(area.sum())),
                         math.log1p(int(cross.sum()))))
        available += int(surface_is_available(original))

    if node_weight <= 0 or edge_count <= 0 or not contexts:
        raise ValueError("training split has no usable cross-surface statistics")
    node_mean, node_std = _moments(node_sum, node_square, node_weight)
    edge_mean, edge_std = _moments(edge_sum, edge_square, edge_count)
    context = torch.tensor(contexts, dtype=torch.float64)
    context_mean = context.mean(dim=0)
    context_std = context.std(dim=0, unbiased=False).clamp_min(1e-4)
    return {
        "version": 2,
        "scope": "training samples; cross-edge incident nodes and edge_type==1 edges",
        "n_complexes": int(len(contexts)),
        "surface_available": int(available),
        "node_area_weight": float(node_weight),
        "cross_edge_count": int(edge_count),
        "node_mean": node_mean.float().tolist(),
        "node_std": node_std.float().tolist(),
        "cross_edge_mean": edge_mean.float().tolist(),
        "cross_edge_std": edge_std.float().tolist(),
        "context_mean": context_mean.float().tolist(),
        "context_std": context_std.float().tolist(),
    }


class CrossEdgeSurfaceEncoder(nn.Module):
    """One-message-layer encoder that cannot consume intra-chain patch edges."""

    def __init__(self, normalization, hidden_channels=48, graph_channels=32,
                 dropout=0.35):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.graph_channels = int(graph_channels)
        self.register_buffer(
            "node_mean", torch.tensor(normalization["node_mean"]).view(1, -1))
        self.register_buffer(
            "node_std", torch.tensor(normalization["node_std"]).view(1, -1))
        self.register_buffer(
            "edge_mean", torch.tensor(normalization["cross_edge_mean"]).view(1, -1))
        self.register_buffer(
            "edge_std", torch.tensor(normalization["cross_edge_std"]).view(1, -1))
        self.register_buffer(
            "context_mean", torch.tensor(normalization["context_mean"]).view(1, -1))
        self.register_buffer(
            "context_std", torch.tensor(normalization["context_std"]).view(1, -1))

        self.node_encoder = nn.Sequential(
            nn.Linear(PATCH_NODE_DIM, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout))
        self.edge_encoder = nn.Sequential(
            nn.Linear(PATCH_EDGE_DIM, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout))
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout))
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout))
        self.update_norm = nn.LayerNorm(hidden_channels)
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(4 * hidden_channels + 3),
            nn.Linear(4 * hidden_channels + 3, graph_channels), nn.SiLU(),
            nn.Dropout(dropout))

    @staticmethod
    def _validate(data):
        version = getattr(data, "patch_graph_version", None)
        if version is None or torch.any(version.long() != PATCH_GRAPH_VERSION):
            raise ValueError("surface residual v2 requires interface_patch_full v5")
        if data.x.shape[-1] != PATCH_NODE_DIM:
            raise ValueError("expected 23 patch node features")
        if data.edge_attr.shape[-1] != PATCH_EDGE_DIM:
            raise ValueError("expected 42 patch edge features")

    def forward(self, data):
        self._validate(data)
        graph_ids = _batch(data)
        n_graphs = _num_graphs(graph_ids)
        cross = data.edge_type.long() == 1
        if not bool(cross.any()):
            raise ValueError("batch contains no cross-surface edge")
        edge_index = data.edge_index.long()[:, cross]
        source, target = edge_index

        node = (torch.nan_to_num(data.x.float()) - self.node_mean) / self.node_std
        edge = ((torch.nan_to_num(data.edge_attr.float()[cross]) - self.edge_mean)
                / self.edge_std)
        hidden = self.node_encoder(node)
        edge_hidden = self.edge_encoder(edge)
        message = self.message(torch.cat(
            (hidden[source], hidden[target], edge_hidden), dim=-1))
        aggregate = scatter(message, target, dim=0, dim_size=hidden.shape[0],
                            reduce="mean")
        incident_nodes = torch.cat((source, target), dim=0)
        incident_count = scatter(
            torch.ones_like(incident_nodes, dtype=hidden.dtype), incident_nodes, dim=0,
            dim_size=hidden.shape[0], reduce="sum")
        incident = incident_count > 0
        updated = self.update_norm(hidden + self.update(
            torch.cat((hidden, aggregate), dim=-1)))

        chain = data.chain_id.long().clamp(0, 1)
        groups = 2 * graph_ids + chain
        area = torch.nan_to_num(data.patch_area.float().reshape(-1)).clamp_min(1e-8)
        weight = area * incident.float()
        area_sum = scatter(weight, groups, dim=0, dim_size=2 * n_graphs,
                           reduce="sum")
        partner_mean = scatter(
            updated * weight[:, None], groups, dim=0, dim_size=2 * n_graphs,
            reduce="sum") / area_sum[:, None].clamp_min(1e-8)
        partner_mean = partner_mean.reshape(n_graphs, 2, -1)

        edge_graph = graph_ids[source]
        cross_mean = scatter(message, edge_graph, dim=0, dim_size=n_graphs,
                             reduce="mean")
        area_by_graph = area_sum.reshape(n_graphs, 2).sum(dim=1)
        edge_count = scatter(
            torch.ones_like(edge_graph, dtype=hidden.dtype), edge_graph, dim=0,
            dim_size=n_graphs, reduce="sum")
        context = torch.stack((torch.log1p(area_by_graph),
                               torch.log1p(edge_count)), dim=1)
        context = (context - self.context_mean) / self.context_std
        availability = getattr(data, "surface_available", None)
        if availability is None:
            availability = context.new_ones((n_graphs, 1))
        else:
            availability = availability.float().reshape(n_graphs, 1)

        left, right = partner_mean[:, 0], partner_mean[:, 1]
        graph_input = torch.cat((left + right, torch.abs(left - right),
                                 left * right, cross_mean, context,
                                 availability), dim=-1)
        return self.graph_projection(graph_input), availability, context


class SurfaceResidualV2(nn.Module):
    """Surface correction with sample-wise reliability and exact zero baseline."""

    def __init__(self, normalization, baseline_mean, baseline_std,
                 hidden_channels=48, graph_channels=32, dropout=0.35,
                 initial_reliability=0.15):
        super().__init__()
        self.surface = CrossEdgeSurfaceEncoder(
            normalization, hidden_channels, graph_channels, dropout)
        self.register_buffer("baseline_mean", torch.tensor(float(baseline_mean)))
        self.register_buffer("baseline_std", torch.tensor(max(float(baseline_std), 1e-4)))
        self.residual_head = nn.Linear(graph_channels, 1)
        self.reliability_head = nn.Linear(graph_channels + 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.reliability_head.weight)
        probability = min(max(float(initial_reliability), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.reliability_head.bias,
                          math.log(probability / (1.0 - probability)))

    def forward_features(self, patch, baseline):
        baseline = baseline.float().reshape(-1, 1)
        graph_hidden, available, context = self.surface(patch)
        baseline_z = (baseline - self.baseline_mean) / self.baseline_std
        raw_residual = self.residual_head(graph_hidden)
        reliability = torch.sigmoid(self.reliability_head(
            torch.cat((graph_hidden, baseline_z), dim=-1))) * available
        correction = reliability * raw_residual
        return baseline + correction, correction, raw_residual, reliability, context

    def forward(self, patch, baseline):
        return self.forward_features(patch, baseline)[0]
