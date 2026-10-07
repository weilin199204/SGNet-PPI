"""Surface data loading and fold-local missing-patch imputation."""
from __future__ import absolute_import
import numpy as np
import torch
from torch_geometric.data import Data


def surface_is_available(patch):
    value = getattr(patch, "surface_available", None)
    return True if value is None else bool(value.reshape(-1)[0].item())


def compute_surface_imputation_stats(samples, indices):
    """Compute a fold-local mean surface prototype from available patches only."""
    node_sum = None
    node_weight = torch.zeros(2, dtype=torch.float64)
    edge_sum = None
    edge_count = torch.zeros(2, dtype=torch.float64)
    chain_areas = [[], []]
    residue_slots = None
    available = 0
    for index in indices:
        patch = samples[int(index)]["patch"]
        if not surface_is_available(patch):
            continue
        x = patch.x.detach().cpu().double()
        edge_attr = patch.edge_attr.detach().cpu().double()
        chain = patch.chain_id.detach().cpu().long().clamp(0, 1)
        edge_type = patch.edge_type.detach().cpu().long().clamp(0, 1)
        area = patch.patch_area.detach().cpu().double().reshape(-1).clamp_min(1e-8)
        if node_sum is None:
            node_sum = torch.zeros((2, x.shape[1]), dtype=torch.float64)
            edge_sum = torch.zeros((2, edge_attr.shape[1]), dtype=torch.float64)
            residue_slots = int(patch.residue_ids.shape[1])
        if x.shape[1] != node_sum.shape[1] or edge_attr.shape[1] != edge_sum.shape[1]:
            raise ValueError("inconsistent patch feature dimensions")
        if int(patch.residue_ids.shape[1]) != residue_slots:
            raise ValueError("inconsistent patch residue slot count")
        for partner in (0, 1):
            mask = chain == partner
            if not bool(mask.any()):
                raise ValueError("available patch is missing partner {}".format(partner))
            weight = area[mask]
            node_sum[partner] += (x[mask] * weight[:, None]).sum(dim=0)
            node_weight[partner] += weight.sum()
            chain_areas[partner].append(float(weight.sum()))
        for kind in (0, 1):
            mask = edge_type == kind
            if bool(mask.any()):
                edge_sum[kind] += edge_attr[mask].sum(dim=0)
                edge_count[kind] += int(mask.sum())
        available += 1
    if available == 0 or node_sum is None or torch.any(node_weight <= 0):
        raise ValueError("training split has no available surface patches")
    edge_mean = edge_sum / edge_count.clamp_min(1.0)[:, None]
    return {
        "version": 1,
        "available_complexes": int(available),
        "node_mean_by_chain": (node_sum / node_weight[:, None]).float().tolist(),
        "edge_mean_by_type": edge_mean.float().tolist(),
        "chain_area_mean": [float(np.mean(values)) for values in chain_areas],
        "residue_slots": int(residue_slots),
    }


def mean_surface_patch(n_a, n_b, stats):
    """Four-node prototype whose raw values are training-fold surface means."""
    node_mean = torch.tensor(stats["node_mean_by_chain"], dtype=torch.float32)
    edge_mean = torch.tensor(stats["edge_mean_by_type"], dtype=torch.float32)
    chain_area = torch.tensor(stats["chain_area_mean"], dtype=torch.float32)
    slots = int(stats["residue_slots"])
    x = torch.stack((node_mean[0], node_mean[0], node_mean[1], node_mean[1]))
    chain_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 2, 1, 3],
        [1, 0, 3, 2, 2, 0, 3, 1],
    ], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    edge_attr = edge_mean[edge_type].clone()
    patch_area = torch.tensor([
        chain_area[0] / 2.0, chain_area[0] / 2.0,
        chain_area[1] / 2.0, chain_area[1] / 2.0,
    ], dtype=torch.float32).reshape(-1, 1).clamp_min(1e-8)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.zeros((4, 3), dtype=torch.float32),
        normal=torch.zeros((4, 3), dtype=torch.float32),
        patch_area=patch_area,
        chain_id=chain_id,
        edge_type=edge_type,
        edge_vector=torch.zeros((8, 3), dtype=torch.float32),
        residue_ids=torch.full((4, slots), -1, dtype=torch.long),
        residue_weights=torch.zeros((4, slots), dtype=torch.float32),
        residue_counts=torch.tensor([[int(n_a), int(n_b)]], dtype=torch.long),
        patch_graph_version=torch.tensor([5], dtype=torch.long),
        surface_available=torch.tensor([False], dtype=torch.bool),
    )


def materialize_surface_patch(patch, n_a, n_b, stats):
    if surface_is_available(patch):
        return patch
    if stats is None:
        raise ValueError("missing surface requires fold-local imputation statistics")
    return mean_surface_patch(n_a, n_b, stats)


SURFACE_MODEL_ATTRS = (
    "x", "edge_index", "edge_attr", "patch_area", "chain_id", "edge_type",
    "residue_ids", "residue_weights", "residue_counts", "patch_graph_version",
)


def surface_model_patch(patch):
    """Normalize real and imputed patches to the exact model input schema."""
    missing = [name for name in SURFACE_MODEL_ATTRS if not hasattr(patch, name)]
    if missing:
        raise ValueError("surface patch missing model fields {}".format(missing))
    values = {name: getattr(patch, name) for name in SURFACE_MODEL_ATTRS}
    values["surface_available"] = torch.tensor(
        [surface_is_available(patch)], dtype=torch.bool)
    return Data(**values)


def validate_affinity_samples(samples):
    errors, ids, residues, available = [], [], 0, 0
    for index, sample in enumerate(samples):
        pdb_id = str(sample.get("pdb_id", "sample_{}".format(index))).upper()
        ids.append(pdb_id)
        try:
            sequence_a = sample["sequence_a"]
            sequence_b = sample["sequence_b"]
            target = sample["target"]
            patch = sample["patch"]
            n_a, n_b = int(sequence_a.num_nodes), int(sequence_b.num_nodes)
            if n_a < 1 or n_b < 1 or sequence_a.x.shape[1] != sequence_b.x.shape[1]:
                raise ValueError("invalid sequence embedding shapes")
            if int(target.num_nodes) != n_a + n_b or target.y.numel() != 1:
                raise ValueError("target/sequence mismatch")
            counts = patch.residue_counts.long().reshape(-1, 2)[0].tolist()
            if counts != [n_a, n_b]:
                raise ValueError("patch residue_counts mismatch")
            if int(patch.patch_graph_version.reshape(-1)[0]) != 5:
                raise ValueError("patch graph version is not 5")
            if patch.x.ndim != 2 or patch.x.shape[1] != 23:
                raise ValueError("patch node feature width is not 23")
            if patch.edge_attr.ndim != 2 or patch.edge_attr.shape[1] != 42:
                raise ValueError("patch edge feature width is not 42")
            if not bool(torch.isfinite(sequence_a.x).all()) or not bool(torch.isfinite(sequence_b.x).all()):
                raise ValueError("non-finite sequence embedding")
            if not bool(torch.isfinite(target.y).all()):
                raise ValueError("non-finite affinity")
            residues += n_a + n_b
            available += int(surface_is_available(patch))
        except Exception as exc:
            errors.append("{}: {}".format(pdb_id, exc))
    if len(set(ids)) != len(ids):
        errors.append("duplicate PDB IDs")
    if errors:
        raise ValueError("Invalid SGNet affinity data: {}".format(errors[:20]))
    return {
        "n_complexes": len(samples),
        "n_residues": int(residues),
        "surface_available": int(available),
        "surface_missing": int(len(samples) - available),
    }
