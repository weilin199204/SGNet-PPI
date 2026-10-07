"""Build SGNet4 interface patches from complete molecular surface meshes.

The builder combines the original two PLY meshes with heavy-atom residue
records saved by graph_construct. Interface vertices are selected from the
union of the bound-complex PLY label, local desolvation, cross-chain atom
contacts, and cross-surface proximity before mesh context is added.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from .surface import find_surface_files, read_surface


PATCH_GRAPH_VERSION = 5
SURFACE_INPUT_DIM = 3
PROBE_RADIUS = 1.4

INTERFACE_SOURCE_NAMES = (
    "ply_iface",
    "delta_sasa",
    "atom_contact",
    "surface_proximity",
)

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "OTHER": 1.70,
}
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "SE": 1.20,
    "OTHER": 0.77,
}
POLAR_ELEMENTS = {"N", "O", "S", "P"}

DONOR_ATOMS = {
    "R": {"NE", "NH1", "NH2"},
    "K": {"NZ"},
    "H": {"ND1", "NE2"},
    "N": {"ND2"},
    "Q": {"NE2"},
    "S": {"OG"},
    "T": {"OG1"},
    "W": {"NE1"},
    "Y": {"OH"},
    "C": {"SG"},
}
ACCEPTOR_ATOMS = {
    "D": {"OD1", "OD2"},
    "E": {"OE1", "OE2"},
    "N": {"OD1"},
    "Q": {"OE1"},
    "H": {"ND1", "NE2"},
    "S": {"OG"},
    "T": {"OG1"},
    "Y": {"OH"},
    "C": {"SG"},
    "M": {"SD"},
}
POSITIVE_ATOMS = {
    "R": {"NE", "NH1", "NH2"},
    "K": {"NZ"},
    "H": {"ND1", "NE2"},
}
NEGATIVE_ATOMS = {
    "D": {"OD1", "OD2"},
    "E": {"OE1", "OE2"},
}

PATCH_NODE_FEATURE_NAMES = (
    "log_surface_area",
    "log_isolated_sasa",
    "log_complex_sasa",
    "log_delta_sasa",
    "buried_fraction",
    "log_polar_delta_sasa",
    "log_apolar_delta_sasa",
    "log_donor_delta_sasa",
    "log_acceptor_delta_sasa",
    "log_positive_delta_sasa",
    "log_negative_delta_sasa",
    "charge_area_mean",
    "hbond_area_mean",
    "hydropathy_area_mean",
    "mean_curvature",
    "gaussian_curvature",
    "shape_index",
    "curvedness",
    "normal_coherence",
    "mean_partner_surface_distance",
    "minimum_partner_surface_distance",
    "core_area_fraction",
    "curvature_valid",
)

PATCH_EDGE_FEATURE_NAMES = (
    "is_intra_patch",
    "is_cross_patch",
    *("center_distance_rbf_{}".format(i) for i in range(8)),
    "center_distance",
    "normal_dot",
    "source_facing",
    "target_facing",
    "log_relation_support",
    "minimum_surface_distance",
    "mean_surface_distance",
    "surface_distance_std",
    "log_surface_pair_count",
    "reciprocal_surface_fraction",
    "surface_charge_product",
    "surface_charge_abs_difference",
    "surface_hbond_product",
    "surface_hbond_abs_difference",
    "surface_hydropathy_product",
    "surface_hydropathy_abs_difference",
    "mean_curvature_sum",
    "shape_index_abs_difference",
    "log_atom_contact_count",
    "log_tight_atom_contact_count",
    "minimum_atom_distance",
    "mean_atom_distance",
    "log_polar_polar_contact_count",
    "log_apolar_apolar_contact_count",
    "log_mixed_contact_count",
    "log_potential_hbond_count",
    "log_directional_hbond_count",
    "mean_hbond_alignment",
    "log_salt_bridge_count",
    "log_clash_count",
    "log_delta_sasa_sum",
    "log_delta_sasa_abs_difference",
)

PATCH_NODE_DIM = len(PATCH_NODE_FEATURE_NAMES)
PATCH_EDGE_DIM = len(PATCH_EDGE_FEATURE_NAMES)


@dataclass(frozen=True)
class PatchBuildConfig:
    patch_radius: float = 4.0
    surface_cross_cutoff: float = 5.0
    atom_contact_cutoff: float = 4.5
    tight_contact_cutoff: float = 3.5
    hbond_cutoff: float = 3.5
    salt_bridge_cutoff: float = 4.0
    clash_overlap: float = 0.4
    delta_sasa_threshold: float = 0.1
    context_hops: int = 1
    sasa_points: int = 192
    max_patches_per_chain: int = 384
    residue_map_k: int = 3
    atom_surface_k: int = 16
    atom_surface_max_gap: float = 1.75
    atom_surface_sigma: float = 0.75
    surface_interface_cutoff: float = 5.0
    min_delta_projection: float = 0.8
    edge_rbf_bins: int = 8
    edge_rbf_cutoff: float = 12.0


def _element(atom):
    atom_name = str(atom.get("type", "")).strip().upper()
    pdbqt_type = str(atom.get("pdbqt_type", "")).strip().upper()
    text = pdbqt_type or atom_name
    while text and text[0].isdigit():
        text = text[1:]
    for candidate in ("CL", "BR", "SE"):
        if text.startswith(candidate):
            return candidate
    value = text[:1]
    return value if value in VDW_RADII else "OTHER"


def _atom_name(atom):
    return str(atom.get("type", "")).strip().upper()


def _is_heavy(atom):
    return _element(atom) != "H"


def _position(atom):
    return [float(atom["x"]), float(atom["y"]), float(atom["z"])]


def _atom_flags(residue_type, atom_name, element):
    donor = ((atom_name == "N" and residue_type != "P")
             or atom_name in DONOR_ATOMS.get(residue_type, set()))
    acceptor = (atom_name in {"O", "OXT"}
                or atom_name in ACCEPTOR_ATOMS.get(residue_type, set()))
    positive = atom_name in POSITIVE_ATOMS.get(residue_type, set())
    negative = (atom_name == "OXT"
                or atom_name in NEGATIVE_ATOMS.get(residue_type, set()))
    polar = element in POLAR_ELEMENTS
    return donor, acceptor, positive, negative, polar


def _atom_table(residues):
    positions = []
    radii = []
    residue_ids = []
    elements = []
    atom_names = []
    flags = []
    for residue_id, residue in enumerate(residues):
        residue_type = str(residue.get("type", "X")).strip().upper()
        for atom in residue.get("atoms", []):
            if not _is_heavy(atom):
                continue
            element = _element(atom)
            atom_name = _atom_name(atom)
            positions.append(_position(atom))
            radii.append(VDW_RADII.get(element, VDW_RADII["OTHER"]))
            residue_ids.append(residue_id)
            elements.append(element)
            atom_names.append(atom_name)
            flags.append(_atom_flags(residue_type, atom_name, element))
    if not positions:
        raise ValueError("No heavy atoms found in one partner")
    flags = np.asarray(flags, dtype=bool)
    table = {
        "pos": np.asarray(positions, dtype=np.float64),
        "radius": np.asarray(radii, dtype=np.float64),
        "residue_id": np.asarray(residue_ids, dtype=np.int64),
        "element": np.asarray(elements, dtype=object),
        "atom_name": np.asarray(atom_names, dtype=object),
        "donor": flags[:, 0],
        "acceptor": flags[:, 1],
        "positive": flags[:, 2],
        "negative": flags[:, 3],
        "polar": flags[:, 4],
        "num_residues": len(residues),
    }
    table["donor_direction"] = _donor_directions(table)
    table["sample_frame"] = _sampling_frames(table["pos"])
    return table


def _sampling_frames(pos):
    """Construct rotation-equivariant local frames for SASA sphere samples."""
    tree = cKDTree(pos)
    k = min(12, len(pos))
    _, neighbor_ids = tree.query(pos, k=k)
    if k == 1:
        neighbor_ids = neighbor_ids[:, None]
    frames = np.zeros((len(pos), 3, 3), dtype=np.float64)
    for atom_id in range(len(pos)):
        candidates = [int(i) for i in np.atleast_1d(neighbor_ids[atom_id])
                      if int(i) != atom_id]
        if not candidates:
            frames[atom_id] = np.eye(3)
            continue
        e1 = pos[candidates[0]] - pos[atom_id]
        e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = None
        for neighbor in candidates[1:]:
            value = pos[neighbor] - pos[atom_id]
            value = value - np.dot(value, e1) * e1
            norm = np.linalg.norm(value)
            if norm > 1e-5:
                e2 = value / norm
                break
        if e2 is None:
            axis = np.eye(3)[int(np.argmin(np.abs(e1)))]
            value = axis - np.dot(axis, e1) * e1
            e2 = value / max(np.linalg.norm(value), 1e-12)
        e3 = np.cross(e1, e2)
        e3 /= max(np.linalg.norm(e3), 1e-12)
        e2 = np.cross(e3, e1)
        frames[atom_id] = np.stack([e1, e2, e3], axis=1)
    return frames


def _donor_directions(table):
    pos = table["pos"]
    residue_ids = table["residue_id"]
    elements = table["element"]
    result = np.zeros_like(pos)
    for atom_id in np.flatnonzero(table["donor"]):
        candidates = np.flatnonzero(residue_ids == residue_ids[atom_id])
        candidates = candidates[candidates != atom_id]
        if not candidates.size:
            continue
        distances = np.linalg.norm(pos[candidates] - pos[atom_id], axis=1)
        covalent_limit = np.asarray([
            COVALENT_RADII.get(elements[atom_id], 0.77)
            + COVALENT_RADII.get(elements[j], 0.77) + 0.5
            for j in candidates
        ])
        bonded = candidates[(distances >= 0.6) & (distances <= covalent_limit)]
        if not bonded.size:
            bonded = candidates[np.argsort(distances)[:1]]
        center = pos[bonded].mean(axis=0)
        direction = pos[atom_id] - center
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            result[atom_id] = direction / norm
    return result


def _fibonacci_sphere(n_points):
    index = np.arange(n_points, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / n_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * index
    return np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), z], axis=1
    )


def _shrake_rupley_sasa(pos, radii, frames, n_points):
    if len(pos) == 0:
        return np.empty(0, dtype=np.float64)
    sphere = _fibonacci_sphere(n_points)
    expanded = radii + PROBE_RADIUS
    tree = cKDTree(pos)
    max_radius = float(expanded.max())
    area = np.zeros(len(pos), dtype=np.float64)
    for atom_id in range(len(pos)):
        oriented = sphere @ frames[atom_id].T
        points = pos[atom_id] + expanded[atom_id] * oriented
        neighbors = tree.query_ball_point(
            pos[atom_id], expanded[atom_id] + max_radius
        )
        neighbors = [neighbor for neighbor in neighbors if neighbor != atom_id]
        accessible = np.ones(n_points, dtype=bool)
        for start in range(0, len(neighbors), 32):
            ids = np.asarray(neighbors[start:start + 32], dtype=np.int64)
            if not ids.size:
                continue
            delta = points[:, None, :] - pos[ids][None, :, :]
            blocked = np.any(
                np.einsum("pni,pni->pn", delta, delta)
                < np.square(expanded[ids])[None, :],
                axis=1,
            )
            accessible &= ~blocked
            if not accessible.any():
                break
        area[atom_id] = (
            4.0 * math.pi * expanded[atom_id] ** 2
            * accessible.sum() / n_points
        )
    return area


def compute_desolvation(table_a, table_b, n_points):
    sasa_a = _shrake_rupley_sasa(
        table_a["pos"], table_a["radius"], table_a["sample_frame"], n_points
    )
    sasa_b = _shrake_rupley_sasa(
        table_b["pos"], table_b["radius"], table_b["sample_frame"], n_points
    )
    split = len(table_a["pos"])
    complex_pos = np.concatenate([table_a["pos"], table_b["pos"]], axis=0)
    complex_radii = np.concatenate([table_a["radius"], table_b["radius"]])
    complex_frames = np.concatenate(
        [table_a["sample_frame"], table_b["sample_frame"]], axis=0
    )
    sasa_complex = _shrake_rupley_sasa(
        complex_pos, complex_radii, complex_frames, n_points
    )
    complex_a, complex_b = sasa_complex[:split], sasa_complex[split:]
    return (
        sasa_a,
        complex_a,
        np.maximum(0.0, sasa_a - complex_a),
    ), (
        sasa_b,
        complex_b,
        np.maximum(0.0, sasa_b - complex_b),
    )


def _cross_contact_atom_masks(table_a, table_b, cutoff):
    neighbors = cKDTree(table_a["pos"]).query_ball_tree(
        cKDTree(table_b["pos"]), cutoff
    )
    contact_a = np.asarray([bool(values) for values in neighbors], dtype=bool)
    contact_b = np.zeros(len(table_b["pos"]), dtype=bool)
    for values in neighbors:
        if values:
            contact_b[np.asarray(values, dtype=np.int64)] = True
    return contact_a, contact_b


def _median_nearest(source, target):
    if not len(source) or not len(target):
        return float("inf")
    return float(np.median(cKDTree(target).query(source, k=1)[0]))


def _surface_partner_map(surface, table_a, table_b):
    chain = surface.chain_id.detach().cpu().numpy().astype(np.int64)
    values = sorted(np.unique(chain).tolist())
    if values != [0, 1]:
        raise ValueError("Surface graph chain_id must contain exactly 0 and 1")
    pos = surface.pos.detach().cpu().numpy().astype(np.float64)
    pos0, pos1 = pos[chain == 0], pos[chain == 1]
    direct = (_median_nearest(pos0, table_a["pos"])
              + _median_nearest(pos1, table_b["pos"]))
    swapped = (_median_nearest(pos0, table_b["pos"])
               + _median_nearest(pos1, table_a["pos"]))
    if not np.isfinite(min(direct, swapped)):
        raise ValueError("Cannot align surface chains to atom partners")
    partner = chain.copy() if direct <= swapped else 1 - chain
    return partner, bool(swapped < direct), direct, swapped


def _mesh_edges(surface, partner):
    edge_type = surface.edge_type.detach().cpu().numpy().astype(np.int64)
    edge_index = surface.edge_index.detach().cpu().numpy().astype(np.int64)
    keep = ((edge_type == 0)
            & (partner[edge_index[0]] == partner[edge_index[1]]))
    return edge_index[:, keep]


def _expand_context(core, mesh_edges, hops):
    selected = core.copy()
    frontier = core.copy()
    src, dst = mesh_edges
    for _ in range(max(0, int(hops))):
        reached = np.zeros_like(selected)
        reached[dst[frontier[src]]] = True
        reached[src[frontier[dst]]] = True
        reached &= ~selected
        selected |= reached
        frontier = reached
        if not frontier.any():
            break
    return selected


def _radius_aware_vertex_owners(surface_pos, atom_table, k):
    """Assign vertices by distance to the candidate atomic surface, not center."""
    candidate_count = min(max(1, int(k)), len(atom_table["pos"]))
    distance, candidates = cKDTree(atom_table["pos"]).query(
        surface_pos, k=candidate_count
    )
    if candidate_count == 1:
        distance = distance[:, None]
        candidates = candidates[:, None]
    gap = np.abs(distance - atom_table["radius"][candidates])
    choice = np.argmin(gap, axis=1)
    rows = np.arange(len(surface_pos))
    return (
        candidates[rows, choice].astype(np.int64),
        distance[rows, choice].astype(np.float64),
        gap[rows, choice].astype(np.float64),
    )


def _allocate_atom_values(
    surface_pos, surface_area, atom_table, atom_values, k, max_gap, sigma
):
    """Soft-project atom quantities to a same-chain mesh with mass conservation."""
    if not len(surface_pos):
        raise ValueError("Cannot project atoms to an empty surface")
    nearest, nearest_distance, nearest_gap = _radius_aware_vertex_owners(
        surface_pos, atom_table, k
    )
    allocated = {
        name: np.zeros(len(surface_pos), dtype=np.float64)
        for name in atom_values
    }
    captured = np.zeros(len(atom_table["pos"]), dtype=bool)
    candidate_count = min(max(1, int(k)), len(surface_pos))
    distances, vertex_candidates = cKDTree(surface_pos).query(
        atom_table["pos"], k=candidate_count
    )
    if candidate_count == 1:
        distances = distances[:, None]
        vertex_candidates = vertex_candidates[:, None]
    gaps = np.abs(distances - atom_table["radius"][:, None])
    sigma = max(float(sigma), 1e-6)
    for atom_id in range(len(atom_table["pos"])):
        valid = np.isfinite(gaps[atom_id]) & (gaps[atom_id] <= max_gap)
        if not valid.any():
            continue
        vertex_ids = np.asarray(vertex_candidates[atom_id, valid], dtype=np.int64)
        atom_gaps = gaps[atom_id, valid]
        weights = np.exp(-0.5 * np.square(atom_gaps / sigma))
        weights *= np.maximum(surface_area[vertex_ids], 1e-6)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        weights /= weight_sum
        for name, values in atom_values.items():
            allocated[name][vertex_ids] += float(values[atom_id]) * weights
        captured[atom_id] = True
    allocated["nearest_atom"] = nearest
    allocated["nearest_atom_distance"] = nearest_distance
    allocated["nearest_atom_surface_gap"] = nearest_gap
    allocated["captured_atom_mask"] = captured
    return allocated


def _build_geodesic_matrix(pos, vertex_ids, mesh_edges):
    old_to_new = np.full(len(pos), -1, dtype=np.int64)
    old_to_new[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int64)
    src = old_to_new[mesh_edges[0]]
    dst = old_to_new[mesh_edges[1]]
    valid = (src >= 0) & (dst >= 0) & (src != dst)
    pairs = {}
    for u, v in zip(src[valid].tolist(), dst[valid].tolist()):
        key = (min(u, v), max(u, v))
        value = float(np.linalg.norm(pos[vertex_ids[u]] - pos[vertex_ids[v]]))
        if value > 0 and (key not in pairs or value < pairs[key]):
            pairs[key] = value
    if not pairs:
        return coo_matrix((len(vertex_ids), len(vertex_ids))).tocsr()
    row, col, values = [], [], []
    for (u, v), value in pairs.items():
        row.extend([u, v])
        col.extend([v, u])
        values.extend([value, value])
    return coo_matrix(
        (values, (row, col)), shape=(len(vertex_ids), len(vertex_ids))
    ).tocsr()


def _geodesic_patch_assignment(
    pos, vertex_ids, mesh_edges, priority, radius, max_patches
):
    graph = _build_geodesic_matrix(pos, vertex_ids, mesh_edges)
    if graph.nnz == 0:
        if len(vertex_ids) > max_patches:
            raise ValueError("Too many disconnected surface vertices for patch limit")
        return np.arange(len(vertex_ids), dtype=np.int64), np.arange(
            len(vertex_ids), dtype=np.int64
        )

    component_count, labels = connected_components(graph, directed=False)
    if component_count > max_patches:
        raise ValueError(
            "Surface has {} mesh components, exceeding patch limit {}".format(
                component_count, max_patches
            )
        )
    centers = []
    for component in range(component_count):
        ids = np.flatnonzero(labels == component)
        centers.append(int(ids[np.argmax(priority[ids])]))

    distance_to_centers = dijkstra(graph, directed=False, indices=centers)
    if distance_to_centers.ndim == 1:
        distance_to_centers = distance_to_centers[None, :]
    min_distance = distance_to_centers.min(axis=0)
    while len(centers) < max_patches:
        candidate = int(np.argmax(min_distance))
        if not np.isfinite(min_distance[candidate]) or min_distance[candidate] <= radius:
            break
        centers.append(candidate)
        new_distance = dijkstra(graph, directed=False, indices=candidate)
        min_distance = np.minimum(min_distance, new_distance)

    all_distance = dijkstra(graph, directed=False, indices=centers)
    if all_distance.ndim == 1:
        all_distance = all_distance[None, :]
    assignment = np.argmin(all_distance, axis=0).astype(np.int64)
    if not np.isfinite(all_distance[assignment, np.arange(len(vertex_ids))]).all():
        raise RuntimeError("One or more surface vertices lack a patch center")
    return assignment, np.asarray(centers, dtype=np.int64)


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denominator = max(float(weights.sum()), 1e-12)
    return np.sum(values * weights.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0) / denominator


def _patch_curvature(pos, normal, area):
    if len(pos) < 10 or float(area.sum()) <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    center = _weighted_mean(pos, area)
    aggregate_normal = _weighted_mean(normal, area)
    normal_norm = np.linalg.norm(aggregate_normal)
    if normal_norm < 1e-8:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    aggregate_normal /= normal_norm
    centered = pos - center
    covariance = (centered * area[:, None]).T @ centered / max(area.sum(), 1e-12)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues[-2] < 1e-4:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    tangent = vectors[:, -2:]
    if np.dot(np.cross(tangent[:, 0], tangent[:, 1]), aggregate_normal) < 0:
        tangent[:, 1] *= -1
    uv = centered @ tangent
    height = centered @ aggregate_normal
    scale = np.sqrt(np.maximum(
        _weighted_mean(np.square(uv), area), 1e-6
    ))
    if float(scale.min()) < 0.15 or float(scale.max() / scale.min()) > 8.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    normalized_uv = uv / scale
    design = np.stack(
        [0.5 * normalized_uv[:, 0] ** 2,
         normalized_uv[:, 0] * normalized_uv[:, 1],
         0.5 * normalized_uv[:, 1] ** 2,
         normalized_uv[:, 0], normalized_uv[:, 1], np.ones(len(uv))],
        axis=1,
    )
    weighted_design = design * np.sqrt(area[:, None].clip(min=1e-12))
    weighted_height = height * np.sqrt(area.clip(min=1e-12))
    try:
        gram = weighted_design.T @ weighted_design
        ridge = 1e-4 * max(float(np.trace(gram)) / len(gram), 1e-8)
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]),
            weighted_design.T @ weighted_height,
        )
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    hessian = np.asarray(
        [[coefficients[0] / scale[0] ** 2,
          coefficients[1] / (scale[0] * scale[1])],
         [coefficients[1] / (scale[0] * scale[1]),
          coefficients[2] / scale[1] ** 2]],
        dtype=np.float64,
    )
    principal = np.linalg.eigvalsh(hessian)
    k1, k2 = float(principal[1]), float(principal[0])
    mean = 0.5 * (k1 + k2)
    gaussian = k1 * k2
    shape = (2.0 / math.pi) * math.atan2(k1 + k2, k1 - k2 + 1e-12)
    curvedness = math.sqrt(0.5 * (k1 * k1 + k2 * k2))
    values = np.asarray([mean, gaussian, shape, curvedness])
    if not np.isfinite(values).all() or float(np.max(np.abs(principal))) > 2.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return mean, gaussian, shape, curvedness, 1.0


def _residue_mapping(vertex_ids, nearest_atom, atom_table, area, k):
    residue_ids = atom_table["residue_id"][nearest_atom[vertex_ids]]
    weights = defaultdict(float)
    for residue_id, value in zip(residue_ids.tolist(), area[vertex_ids].tolist()):
        weights[int(residue_id)] += float(value)
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:k]
    output_ids = np.full(k, -1, dtype=np.int64)
    output_weights = np.zeros(k, dtype=np.float64)
    if ranked:
        total = sum(value for _, value in ranked)
        for index, (residue_id, value) in enumerate(ranked):
            output_ids[index] = residue_id
            output_weights[index] = value / max(total, 1e-12)
    return output_ids, output_weights


def _build_patches_for_partner(
    partner_id,
    surface,
    surface_partner,
    mesh_edges,
    allocation,
    atom_table,
    partner_distance,
    interface_sources,
    config,
):
    pos = surface.pos.detach().cpu().numpy().astype(np.float64)
    normal = surface.normal.detach().cpu().numpy().astype(np.float64)
    surface_x = surface.x.detach().cpu().numpy().astype(np.float64)
    area = surface.vertex_area.detach().cpu().numpy().reshape(-1).astype(np.float64)
    area = np.maximum(area, 0.0)
    partner_mask = surface_partner == partner_id
    core = partner_mask & np.any(interface_sources, axis=1)
    selected = _expand_context(core, mesh_edges, config.context_hops) & partner_mask
    vertex_ids = np.flatnonzero(selected)
    if not len(vertex_ids):
        raise ValueError("Partner {} has no selected surface vertices".format(partner_id))

    local_priority = (
        allocation["delta"][vertex_ids]
        + area[vertex_ids] * core[vertex_ids].astype(np.float64)
    )
    assignment, centers = _geodesic_patch_assignment(
        pos,
        vertex_ids,
        mesh_edges,
        local_priority,
        config.patch_radius,
        config.max_patches_per_chain,
    )
    patch_count = len(centers)
    global_assignment = np.full(len(pos), -1, dtype=np.int64)
    global_assignment[vertex_ids] = assignment

    features = []
    patch_pos = []
    patch_normal = []
    patch_area = []
    patch_core = []
    residue_ids = []
    residue_weights = []
    vertex_counts = []
    source_fractions = []
    for patch_id in range(patch_count):
        ids = vertex_ids[assignment == patch_id]
        weights = area[ids]
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        total_area = float(area[ids].sum())
        center = _weighted_mean(pos[ids], weights)
        average_normal = _weighted_mean(normal[ids], weights)
        coherence = float(np.linalg.norm(average_normal))
        if coherence > 1e-8:
            average_normal /= coherence
        else:
            average_normal = np.zeros(3, dtype=np.float64)

        isolated = float(allocation["isolated"][ids].sum())
        complex_sasa = float(allocation["complex"][ids].sum())
        delta = float(allocation["delta"][ids].sum())
        polar_delta = float(allocation["polar_delta"][ids].sum())
        apolar_delta = float(allocation["apolar_delta"][ids].sum())
        donor_delta = float(allocation["donor_delta"][ids].sum())
        acceptor_delta = float(allocation["acceptor_delta"][ids].sum())
        positive_delta = float(allocation["positive_delta"][ids].sum())
        negative_delta = float(allocation["negative_delta"][ids].sum())
        chemistry = _weighted_mean(surface_x[ids], weights)
        curvature_radius = max(4.0, 1.5 * config.patch_radius)
        curvature_ids = vertex_ids[
            np.linalg.norm(pos[vertex_ids] - center, axis=1) <= curvature_radius
        ]
        curvature = _patch_curvature(
            pos[curvature_ids], normal[curvature_ids], area[curvature_ids]
        )
        core_fraction = float(area[ids][core[ids]].sum()) / max(total_area, 1e-12)
        values = [
            math.log1p(max(total_area, 0.0)),
            math.log1p(max(isolated, 0.0)),
            math.log1p(max(complex_sasa, 0.0)),
            math.log1p(max(delta, 0.0)),
            delta / max(isolated, 1e-12),
            math.log1p(max(polar_delta, 0.0)),
            math.log1p(max(apolar_delta, 0.0)),
            math.log1p(max(donor_delta, 0.0)),
            math.log1p(max(acceptor_delta, 0.0)),
            math.log1p(max(positive_delta, 0.0)),
            math.log1p(max(negative_delta, 0.0)),
            float(chemistry[0]),
            float(chemistry[1]),
            float(chemistry[2]),
            *curvature[:4],
            coherence,
            float(_weighted_mean(partner_distance[ids], weights)),
            float(partner_distance[ids].min()),
            core_fraction,
            curvature[4],
        ]
        if len(values) != PATCH_NODE_DIM:
            raise RuntimeError("Unexpected patch node feature dimension")
        mapping_ids, mapping_weights = _residue_mapping(
            ids,
            allocation["nearest_atom"],
            atom_table,
            area,
            config.residue_map_k,
        )
        features.append(values)
        patch_pos.append(center)
        patch_normal.append(average_normal)
        patch_area.append(total_area)
        patch_core.append(bool(core[ids].any()))
        residue_ids.append(mapping_ids)
        residue_weights.append(mapping_weights)
        vertex_counts.append(len(ids))
        source_fractions.append([
            float(area[ids][interface_sources[ids, source_id]].sum())
            / max(total_area, 1e-12)
            for source_id in range(len(INTERFACE_SOURCE_NAMES))
        ])

    return {
        "x": np.asarray(features, dtype=np.float64),
        "pos": np.asarray(patch_pos, dtype=np.float64),
        "normal": np.asarray(patch_normal, dtype=np.float64),
        "area": np.asarray(patch_area, dtype=np.float64),
        "core": np.asarray(patch_core, dtype=bool),
        "residue_ids": np.asarray(residue_ids, dtype=np.int64),
        "residue_weights": np.asarray(residue_weights, dtype=np.float64),
        "vertex_counts": np.asarray(vertex_counts, dtype=np.int64),
        "source_fractions": np.asarray(source_fractions, dtype=np.float64),
        "vertex_ids": vertex_ids,
        "vertex_assignment": global_assignment,
    }


def _rbf(distance, cutoff, bins):
    centers = np.linspace(0.0, float(cutoff), bins, dtype=np.float64)
    spacing = float(cutoff) / max(bins - 1, 1)
    return np.exp(-np.square(distance - centers) / max(spacing * spacing, 1e-12))


def _new_surface_stats():
    return {
        "count": 0,
        "weight": 0.0,
        "distance_sum": 0.0,
        "distance_sq_sum": 0.0,
        "distance_min": float("inf"),
        "normal_dot": 0.0,
        "source_facing": 0.0,
        "target_facing": 0.0,
        "charge_product": 0.0,
        "charge_difference": 0.0,
        "hbond_product": 0.0,
        "hbond_difference": 0.0,
        "hydropathy_product": 0.0,
        "hydropathy_difference": 0.0,
        "mutual": 0,
        "vertices_a": set(),
        "vertices_b": set(),
    }


def _surface_pair_stats(
    surface, partner, patch_for_vertex, cutoff
):
    pos = surface.pos.detach().cpu().numpy().astype(np.float64)
    normal = surface.normal.detach().cpu().numpy().astype(np.float64)
    x = surface.x.detach().cpu().numpy().astype(np.float64)
    area = surface.vertex_area.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ids_a = np.flatnonzero((partner == 0) & (patch_for_vertex >= 0))
    ids_b = np.flatnonzero((partner == 1) & (patch_for_vertex >= 0))
    if not len(ids_a) or not len(ids_b):
        return {}
    tree_a, tree_b = cKDTree(pos[ids_a]), cKDTree(pos[ids_b])
    neighbors = tree_a.query_ball_tree(tree_b, cutoff)
    nearest_b = tree_b.query(pos[ids_a], k=1)[1]
    nearest_a = tree_a.query(pos[ids_b], k=1)[1]
    stats = defaultdict(_new_surface_stats)
    for local_a, targets in enumerate(neighbors):
        vertex_a = int(ids_a[local_a])
        for local_b in targets:
            vertex_b = int(ids_b[local_b])
            patch_a = int(patch_for_vertex[vertex_a])
            patch_b = int(patch_for_vertex[vertex_b])
            key = (patch_a, patch_b)
            value = stats[key]
            vector = pos[vertex_b] - pos[vertex_a]
            distance = float(np.linalg.norm(vector))
            direction = vector / max(distance, 1e-12)
            weight = math.sqrt(max(float(area[vertex_a] * area[vertex_b]), 1e-12))
            value["count"] += 1
            value["weight"] += weight
            value["distance_sum"] += weight * distance
            value["distance_sq_sum"] += weight * distance * distance
            value["distance_min"] = min(value["distance_min"], distance)
            value["normal_dot"] += weight * float(np.dot(normal[vertex_a], normal[vertex_b]))
            value["source_facing"] += weight * float(np.dot(normal[vertex_a], direction))
            value["target_facing"] += weight * float(np.dot(normal[vertex_b], -direction))
            value["charge_product"] += weight * float(x[vertex_a, 0] * x[vertex_b, 0])
            value["charge_difference"] += weight * float(abs(x[vertex_a, 0] - x[vertex_b, 0]))
            value["hbond_product"] += weight * float(x[vertex_a, 1] * x[vertex_b, 1])
            value["hbond_difference"] += weight * float(abs(x[vertex_a, 1] - x[vertex_b, 1]))
            value["hydropathy_product"] += weight * float(x[vertex_a, 2] * x[vertex_b, 2])
            value["hydropathy_difference"] += weight * float(abs(x[vertex_a, 2] - x[vertex_b, 2]))
            value["mutual"] += int(
                int(nearest_b[local_a]) == int(local_b)
                and int(nearest_a[local_b]) == int(local_a)
            )
            value["vertices_a"].add(vertex_a)
            value["vertices_b"].add(vertex_b)
    for value in stats.values():
        value["contact_area"] = 0.5 * (
            float(area[list(value["vertices_a"])].sum())
            + float(area[list(value["vertices_b"])].sum())
        )
    return dict(stats)


def _new_atom_stats():
    return {
        "count": 0,
        "tight": 0,
        "distance_sum": 0.0,
        "distance_min": float("inf"),
        "polar_polar": 0,
        "apolar_apolar": 0,
        "mixed": 0,
        "potential_hbond": 0,
        "directional_hbond": 0,
        "hbond_alignment_sum": 0.0,
        "salt": 0,
        "clash": 0,
    }


def _atom_to_patch(table, surface_pos, surface_vertex_ids, patch_for_vertex, config):
    selected_pos = surface_pos[surface_vertex_ids]
    candidate_count = min(max(1, config.atom_surface_k), len(selected_pos))
    distance, candidate = cKDTree(selected_pos).query(
        table["pos"], k=candidate_count
    )
    if candidate_count == 1:
        distance = distance[:, None]
        candidate = candidate[:, None]
    gap = np.abs(distance - table["radius"][:, None])
    choice = np.argmin(gap, axis=1)
    rows = np.arange(len(table["pos"]))
    mapped = np.full(len(table["pos"]), -1, dtype=np.int64)
    valid = gap[rows, choice] <= config.atom_surface_max_gap
    vertices = surface_vertex_ids[candidate[rows[valid], choice[valid]]]
    mapped[valid] = patch_for_vertex[vertices]
    return mapped


def _atom_pair_stats(table_a, table_b, atom_patch_a, atom_patch_b, config):
    neighbors = cKDTree(table_a["pos"]).query_ball_tree(
        cKDTree(table_b["pos"]), config.atom_contact_cutoff
    )
    stats = defaultdict(_new_atom_stats)
    for atom_a, targets in enumerate(neighbors):
        for atom_b in targets:
            patch_a = int(atom_patch_a[atom_a])
            patch_b = int(atom_patch_b[atom_b])
            if patch_a < 0 or patch_b < 0:
                continue
            value = stats[(patch_a, patch_b)]
            vector = table_b["pos"][atom_b] - table_a["pos"][atom_a]
            distance = float(np.linalg.norm(vector))
            value["count"] += 1
            value["tight"] += int(distance <= config.tight_contact_cutoff)
            value["distance_sum"] += distance
            value["distance_min"] = min(value["distance_min"], distance)
            polar_a = bool(table_a["polar"][atom_a])
            polar_b = bool(table_b["polar"][atom_b])
            value["polar_polar"] += int(polar_a and polar_b)
            value["apolar_apolar"] += int(not polar_a and not polar_b)
            value["mixed"] += int(polar_a != polar_b)

            donor_a = bool(table_a["donor"][atom_a])
            donor_b = bool(table_b["donor"][atom_b])
            acceptor_a = bool(table_a["acceptor"][atom_a])
            acceptor_b = bool(table_b["acceptor"][atom_b])
            hbond = ((donor_a and acceptor_b) or (acceptor_a and donor_b))
            if hbond and distance <= config.hbond_cutoff:
                value["potential_hbond"] += 1
                direction = vector / max(distance, 1e-12)
                alignments = []
                if donor_a and acceptor_b:
                    alignments.append(float(np.dot(
                        table_a["donor_direction"][atom_a], direction
                    )))
                if donor_b and acceptor_a:
                    alignments.append(float(np.dot(
                        table_b["donor_direction"][atom_b], -direction
                    )))
                alignment = max(alignments) if alignments else 0.0
                alignment = max(-1.0, min(1.0, alignment))
                value["hbond_alignment_sum"] += alignment
                value["directional_hbond"] += int(alignment >= 0.5)

            opposite_charge = (
                bool(table_a["positive"][atom_a] and table_b["negative"][atom_b])
                or bool(table_a["negative"][atom_a] and table_b["positive"][atom_b])
            )
            value["salt"] += int(
                opposite_charge and distance <= config.salt_bridge_cutoff
            )
            clash_limit = (
                table_a["radius"][atom_a] + table_b["radius"][atom_b]
                - config.clash_overlap
            )
            value["clash"] += int(distance < clash_limit)
    return dict(stats)


def _patch_boundary_stats(mesh_edges, patch_for_vertex, partner, surface_pos):
    stats = defaultdict(lambda: {"count": 0, "boundary": 0.0})
    seen = set()
    for src, dst in mesh_edges.T.tolist():
        patch_src = int(patch_for_vertex[src])
        patch_dst = int(patch_for_vertex[dst])
        if patch_src < 0 or patch_dst < 0 or patch_src == patch_dst:
            continue
        if partner[src] != partner[dst]:
            continue
        key = (min(patch_src, patch_dst), max(patch_src, patch_dst))
        vertex_key = (min(src, dst), max(src, dst))
        if vertex_key in seen:
            continue
        seen.add(vertex_key)
        stats[key]["count"] += 1
        stats[key]["boundary"] += float(np.linalg.norm(surface_pos[src] - surface_pos[dst]))
    return dict(stats)


def _edge_features(
    src,
    dst,
    edge_type,
    node,
    surface_stats,
    atom_stats,
    support_area,
    config,
):
    vector = node["pos"][dst] - node["pos"][src]
    center_distance = float(np.linalg.norm(vector))
    direction = vector / max(center_distance, 1e-12)
    normal_dot = float(np.dot(node["normal"][src], node["normal"][dst]))
    source_facing = float(np.dot(node["normal"][src], direction))
    target_facing = float(np.dot(node["normal"][dst], -direction))

    if surface_stats:
        weight = max(float(surface_stats["weight"]), 1e-12)
        mean_surface = surface_stats["distance_sum"] / weight
        variance = max(
            0.0,
            surface_stats["distance_sq_sum"] / weight - mean_surface ** 2,
        )
        minimum_surface = surface_stats["distance_min"]
        surface_count = surface_stats["count"]
        reciprocal = surface_stats["mutual"] / max(surface_count, 1)
        chemistry = [
            surface_stats["charge_product"] / weight,
            surface_stats["charge_difference"] / weight,
            surface_stats["hbond_product"] / weight,
            surface_stats["hbond_difference"] / weight,
            surface_stats["hydropathy_product"] / weight,
            surface_stats["hydropathy_difference"] / weight,
        ]
    else:
        minimum_surface = mean_surface = center_distance
        variance = 0.0
        surface_count = 0
        reciprocal = 0.0
        chemistry = [
            node["surface_x"][src, 0] * node["surface_x"][dst, 0],
            abs(node["surface_x"][src, 0] - node["surface_x"][dst, 0]),
            node["surface_x"][src, 1] * node["surface_x"][dst, 1],
            abs(node["surface_x"][src, 1] - node["surface_x"][dst, 1]),
            node["surface_x"][src, 2] * node["surface_x"][dst, 2],
            abs(node["surface_x"][src, 2] - node["surface_x"][dst, 2]),
        ]

    if atom_stats:
        atom_count = atom_stats["count"]
        atom_min = atom_stats["distance_min"]
        atom_mean = atom_stats["distance_sum"] / max(atom_count, 1)
        potential_hbond = atom_stats["potential_hbond"]
        alignment = (
            atom_stats["hbond_alignment_sum"] / max(potential_hbond, 1)
        )
    else:
        atom_count = 0
        atom_min = atom_mean = 0.0
        potential_hbond = 0
        alignment = 0.0
        atom_stats = _new_atom_stats()

    delta_src = math.expm1(max(float(node["x"][src, 3]), 0.0))
    delta_dst = math.expm1(max(float(node["x"][dst, 3]), 0.0))
    values = [
        float(edge_type == 0),
        float(edge_type == 1),
        *_rbf(center_distance, config.edge_rbf_cutoff, config.edge_rbf_bins),
        center_distance,
        normal_dot,
        source_facing,
        target_facing,
        math.log1p(max(float(support_area), 0.0)),
        minimum_surface,
        mean_surface,
        math.sqrt(variance),
        math.log1p(surface_count),
        reciprocal,
        *chemistry,
        float(node["x"][src, 14] + node["x"][dst, 14]),
        abs(float(node["x"][src, 16] - node["x"][dst, 16])),
        math.log1p(atom_count),
        math.log1p(atom_stats["tight"]),
        atom_min,
        atom_mean,
        math.log1p(atom_stats["polar_polar"]),
        math.log1p(atom_stats["apolar_apolar"]),
        math.log1p(atom_stats["mixed"]),
        math.log1p(potential_hbond),
        math.log1p(atom_stats["directional_hbond"]),
        alignment,
        math.log1p(atom_stats["salt"]),
        math.log1p(atom_stats["clash"]),
        math.log1p(max(0.0, delta_src + delta_dst)),
        math.log1p(abs(delta_src - delta_dst)),
    ]
    if len(values) != PATCH_EDGE_DIM:
        raise RuntimeError(
            "Unexpected patch edge feature dimension: {} != {}".format(
                len(values), PATCH_EDGE_DIM
            )
        )
    return values, vector


def _combine_patch_nodes(patch_a, patch_b, residue_count_a):
    node = {}
    for key in ("x", "pos", "normal", "area", "core", "vertex_counts", "source_fractions"):
        node[key] = np.concatenate([patch_a[key], patch_b[key]], axis=0)
    node["surface_x"] = node["x"][:, 11:14]
    node["chain_id"] = np.concatenate([
        np.zeros(len(patch_a["x"]), dtype=np.int64),
        np.ones(len(patch_b["x"]), dtype=np.int64),
    ])
    node["residue_ids"] = np.concatenate(
        [patch_a["residue_ids"], patch_b["residue_ids"]], axis=0
    )
    node["residue_weights"] = np.concatenate(
        [patch_a["residue_weights"], patch_b["residue_weights"]], axis=0
    )
    node["owner_residue_id"] = node["residue_ids"][:, 0]
    node["owner_global_residue_id"] = node["owner_residue_id"].copy()
    node["owner_global_residue_id"][node["chain_id"] == 1] += residue_count_a
    return node


def _ensure_intra_connectivity(pairs, node):
    pairs = dict(pairs)
    degree = np.zeros(len(node["x"]), dtype=np.int64)
    for src, dst in pairs:
        degree[src] += 1
        degree[dst] += 1
    for partner_id in (0, 1):
        ids = np.flatnonzero(node["chain_id"] == partner_id)
        if len(ids) <= 1:
            continue
        tree = cKDTree(node["pos"][ids])
        for src in ids[degree[ids] == 0]:
            _, local = tree.query(node["pos"][src], k=min(2, len(ids)))
            targets = np.atleast_1d(local)
            target = next((int(ids[index]) for index in targets
                           if int(ids[index]) != int(src)), None)
            if target is not None:
                key = (min(int(src), target), max(int(src), target))
                pairs.setdefault(key, {"count": 0, "boundary": 0.0})
                degree[src] += 1
                degree[target] += 1
    return pairs


def build_interface_patch_graph(surface, residues_a, residues_b, config=None):
    config = config or PatchBuildConfig()
    if surface.x.ndim != 2 or surface.x.shape[1] != SURFACE_INPUT_DIM:
        raise ValueError("Expected three complete-surface node features")
    required = (
        "pos", "normal", "vertex_area", "chain_id", "core_mask",
        "edge_index", "edge_type",
    )
    missing = [name for name in required if getattr(surface, name, None) is None]
    if missing:
        raise ValueError("Surface graph missing {}".format(", ".join(missing)))
    if config.atom_surface_k < 1:
        raise ValueError("atom_surface_k must be positive")
    if config.atom_surface_max_gap <= 0 or config.atom_surface_sigma <= 0:
        raise ValueError("Atom-surface projection distances must be positive")

    table_a, table_b = _atom_table(residues_a), _atom_table(residues_b)
    desolvation_a, desolvation_b = compute_desolvation(
        table_a, table_b, config.sasa_points
    )
    contact_a, contact_b = _cross_contact_atom_masks(
        table_a, table_b, config.atom_contact_cutoff
    )
    surface_partner, swapped, direct_score, swapped_score = _surface_partner_map(
        surface, table_a, table_b
    )
    surface_pos = surface.pos.detach().cpu().numpy().astype(np.float64)
    surface_area = surface.vertex_area.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if np.any(surface_area < 0) or not np.isfinite(surface_area).all():
        raise ValueError("Invalid surface vertex areas")
    mesh_edges = _mesh_edges(surface, surface_partner)

    allocation_by_partner = {}
    partner_inputs = (
        (table_a, desolvation_a, contact_a),
        (table_b, desolvation_b, contact_b),
    )
    for partner_id, (table, desolvation, contact_mask) in enumerate(partner_inputs):
        vertex_ids = np.flatnonzero(surface_partner == partner_id)
        isolated, complex_sasa, delta = desolvation
        atom_values = {
            "isolated": isolated,
            "complex": complex_sasa,
            "delta": delta,
            "polar_delta": delta * table["polar"],
            "apolar_delta": delta * ~table["polar"],
            "donor_delta": delta * table["donor"],
            "acceptor_delta": delta * table["acceptor"],
            "positive_delta": delta * table["positive"],
            "negative_delta": delta * table["negative"],
            "delta_support": (delta >= config.delta_sasa_threshold).astype(np.float64),
            "contact_support": contact_mask.astype(np.float64),
        }
        local = _allocate_atom_values(
            surface_pos[vertex_ids],
            surface_area[vertex_ids],
            table,
            atom_values,
            config.atom_surface_k,
            config.atom_surface_max_gap,
            config.atom_surface_sigma,
        )
        allocation = {
            name: np.zeros(len(surface_pos), dtype=value.dtype)
            for name, value in local.items()
            if name != "captured_atom_mask"
        }
        for name, value in allocation.items():
            value[vertex_ids] = local[name]
        allocation["captured_atom_mask"] = local["captured_atom_mask"]
        allocation["atom_delta"] = delta
        allocation_by_partner[partner_id] = allocation

    for partner_id, (table, desolvation, _) in enumerate(partner_inputs):
        total_delta = float(desolvation[2].sum())
        captured = allocation_by_partner[partner_id]["captured_atom_mask"]
        projected_delta = float(desolvation[2][captured].sum())
        projection_fraction = projected_delta / max(total_delta, 1e-12)
        if (
            total_delta >= 1.0
            and projection_fraction + 1e-8 < config.min_delta_projection
        ):
            raise ValueError(
                "Incomplete PLY for partner {}: only {:.4f} of delta SASA "
                "maps to the molecular surface (required {:.4f})".format(
                    partner_id, projection_fraction, config.min_delta_projection
                )
            )

    ids_a = np.flatnonzero(surface_partner == 0)
    ids_b = np.flatnonzero(surface_partner == 1)
    distance_a = cKDTree(surface_pos[ids_b]).query(surface_pos[ids_a], k=1)[0]
    distance_b = cKDTree(surface_pos[ids_a]).query(surface_pos[ids_b], k=1)[0]
    partner_distance = np.zeros(len(surface_pos), dtype=np.float64)
    partner_distance[ids_a] = distance_a
    partner_distance[ids_b] = distance_b

    ply_iface = getattr(surface, "ply_iface_mask", surface.core_mask)
    ply_iface = ply_iface.detach().cpu().numpy().astype(bool)
    interface_sources = np.zeros(
        (len(surface_pos), len(INTERFACE_SOURCE_NAMES)), dtype=bool
    )
    interface_sources[:, 0] = ply_iface
    interface_sources[:, 3] = partner_distance <= config.surface_interface_cutoff
    for partner_id in (0, 1):
        vertex_ids = np.flatnonzero(surface_partner == partner_id)
        allocation = allocation_by_partner[partner_id]
        interface_sources[vertex_ids, 1] = allocation["delta_support"][vertex_ids] > 0
        interface_sources[vertex_ids, 2] = allocation["contact_support"][vertex_ids] > 0

    patches_a = _build_patches_for_partner(
        0, surface, surface_partner, mesh_edges, allocation_by_partner[0],
        table_a, partner_distance, interface_sources, config,
    )
    patches_b = _build_patches_for_partner(
        1, surface, surface_partner, mesh_edges, allocation_by_partner[1],
        table_b, partner_distance, interface_sources, config,
    )
    offset_b = len(patches_a["x"])
    patches_b["vertex_assignment"] = np.where(
        patches_b["vertex_assignment"] >= 0,
        patches_b["vertex_assignment"] + offset_b,
        -1,
    )
    patch_for_vertex = np.where(
        patches_a["vertex_assignment"] >= 0,
        patches_a["vertex_assignment"],
        patches_b["vertex_assignment"],
    )
    node = _combine_patch_nodes(patches_a, patches_b, len(residues_a))

    boundary = _ensure_intra_connectivity(
        _patch_boundary_stats(mesh_edges, patch_for_vertex, surface_partner, surface_pos), node
    )
    surface_pairs = _surface_pair_stats(
        surface, surface_partner, patch_for_vertex, config.surface_cross_cutoff
    )
    atom_patch_a = _atom_to_patch(
        table_a, surface_pos, patches_a["vertex_ids"], patch_for_vertex, config
    )
    atom_patch_b = _atom_to_patch(
        table_b, surface_pos, patches_b["vertex_ids"], patch_for_vertex, config
    )
    atom_pairs = _atom_pair_stats(
        table_a, table_b, atom_patch_a, atom_patch_b, config
    )

    directed_edges = []
    for (src, dst), value in sorted(boundary.items()):
        directed_edges.extend([
            (src, dst, 0, None, None, value["boundary"]),
            (dst, src, 0, None, None, value["boundary"]),
        ])
    cross_keys = sorted(set(surface_pairs) | set(atom_pairs))
    for src, dst in cross_keys:
        surface_value = surface_pairs.get((src, dst))
        atom_value = atom_pairs.get((src, dst))
        support = surface_value["contact_area"] if surface_value else 0.0
        directed_edges.extend([
            (src, dst, 1, surface_value, atom_value, support),
            (dst, src, 1, surface_value, atom_value, support),
        ])
    if not directed_edges:
        raise ValueError("Patch graph has no edges")
    if not cross_keys:
        raise ValueError("Patch graph has no cross-interface edges")

    edge_index = []
    edge_type = []
    edge_attr = []
    edge_vector = []
    for src, dst, relation, surface_value, atom_value, support in directed_edges:
        features, vector = _edge_features(
            src, dst, relation, node, surface_value, atom_value, support, config
        )
        edge_index.append([src, dst])
        edge_type.append(relation)
        edge_attr.append(features)
        edge_vector.append(vector)

    total_delta_a = float(desolvation_a[2].sum())
    total_delta_b = float(desolvation_b[2].sum())
    captured_delta_a = float(
        allocation_by_partner[0]["delta"][patches_a["vertex_ids"]].sum()
    )
    captured_delta_b = float(
        allocation_by_partner[1]["delta"][patches_b["vertex_ids"]].sum()
    )
    diagnostics = [
        total_delta_a,
        total_delta_b,
        captured_delta_a,
        captured_delta_b,
        captured_delta_a / max(total_delta_a, 1e-12),
        captured_delta_b / max(total_delta_b, 1e-12),
        direct_score,
        swapped_score,
    ]
    full_counts = [len(ids_a), len(ids_b)]
    selected_counts = [len(patches_a["vertex_ids"]), len(patches_b["vertex_ids"])]
    source_counts = []
    for source_id in range(len(INTERFACE_SOURCE_NAMES)):
        source_counts.extend([
            int(interface_sources[ids_a, source_id].sum()),
            int(interface_sources[ids_b, source_id].sum()),
        ])

    graph = Data(
        x=torch.tensor(node["x"], dtype=torch.float32),
        pos=torch.tensor(node["pos"], dtype=torch.float32),
        normal=torch.tensor(node["normal"], dtype=torch.float32),
        patch_area=torch.tensor(node["area"], dtype=torch.float32).view(-1, 1),
        chain_id=torch.tensor(node["chain_id"], dtype=torch.long),
        core_mask=torch.tensor(node["core"], dtype=torch.bool),
        interface_source_fraction=torch.tensor(
            node["source_fractions"], dtype=torch.float32
        ),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        edge_type=torch.tensor(edge_type, dtype=torch.long),
        edge_vector=torch.tensor(np.asarray(edge_vector), dtype=torch.float32),
        residue_ids=torch.tensor(node["residue_ids"], dtype=torch.long),
        residue_weights=torch.tensor(node["residue_weights"], dtype=torch.float32),
        owner_residue_id=torch.tensor(node["owner_residue_id"], dtype=torch.long),
        owner_global_residue_id=torch.tensor(
            node["owner_global_residue_id"], dtype=torch.long
        ),
        patch_vertex_count=torch.tensor(node["vertex_counts"], dtype=torch.long),
        residue_counts=torch.tensor(
            [[len(residues_a), len(residues_b)]], dtype=torch.long
        ),
        desolvation_summary=torch.tensor([diagnostics], dtype=torch.float32),
        interface_selection_summary=torch.tensor(
            [[*full_counts, *selected_counts, *source_counts]], dtype=torch.long
        ),
        surface_chain_swapped=torch.tensor([swapped], dtype=torch.bool),
        patch_graph_version=torch.tensor([PATCH_GRAPH_VERSION], dtype=torch.long),
    )
    graph.x = torch.nan_to_num(graph.x)
    graph.edge_attr = torch.nan_to_num(graph.edge_attr)
    graph.edge_vector = torch.nan_to_num(graph.edge_vector)
    graph.interface_source_fraction = torch.nan_to_num(
        graph.interface_source_fraction
    )
    return graph


def _load_atom_residues(atom_graph_path):
    with Path(atom_graph_path).open("rb") as handle:
        graph_construct = pickle.load(handle)
    if not isinstance(graph_construct, tuple) or len(graph_construct) < 4:
        raise ValueError("graph_construct inter_graph must be a four-item tuple")
    residues = graph_construct[3]
    if not isinstance(residues, tuple) or len(residues) != 2:
        raise ValueError("graph_construct does not contain two residue groups")
    return residues[0], residues[1]


def _raw_iface_mask(raw_ply, vertex_count):
    names = {prop.name for prop in raw_ply["vertex"].properties}
    if "iface" not in names:
        return torch.zeros(vertex_count, dtype=torch.bool)
    values = np.asarray(raw_ply["vertex"]["iface"], dtype=np.float64)
    return torch.as_tensor(values > 0.5, dtype=torch.bool)


def load_complete_ply_surface(surface_dir, pdb_id, surface_index=None):
    """Load two uncropped PLY meshes into the surface format used downstream."""
    paths = find_surface_files(surface_dir, pdb_id, surface_index=surface_index)
    parts, raw = read_surface(surface_dir, pdb_id, surface_index=surface_index)
    first, second = parts["protein_0"], parts["protein_1"]
    offset = first.num_nodes
    mesh_a = first.edge_index.long()
    mesh_b = second.edge_index.long() + offset
    iface_a = _raw_iface_mask(raw["protein_0"], first.num_nodes)
    iface_b = _raw_iface_mask(raw["protein_1"], second.num_nodes)
    graph = Data(
        x=torch.cat([first.x, second.x], dim=0),
        pos=torch.cat([first.pos, second.pos], dim=0),
        normal=torch.cat([first.normal, second.normal], dim=0),
        vertex_area=torch.cat([first.vertex_area, second.vertex_area], dim=0),
        chain_id=torch.cat([
            torch.zeros(first.num_nodes, dtype=torch.long),
            torch.ones(second.num_nodes, dtype=torch.long),
        ]),
        core_mask=torch.cat([iface_a, iface_b], dim=0),
        ply_iface_mask=torch.cat([iface_a, iface_b], dim=0),
        edge_index=torch.cat([mesh_a, mesh_b], dim=1),
        edge_type=torch.zeros(
            mesh_a.shape[1] + mesh_b.shape[1], dtype=torch.long
        ),
        full_surface_vertex_counts=torch.tensor(
            [[first.num_nodes, second.num_nodes]], dtype=torch.long
        ),
    )
    return graph, paths


def load_patch_inputs(surface_path, atom_graph_path):
    """Legacy loader retained only for reproducible surfgraph_3 comparisons."""
    with Path(surface_path).open("rb") as handle:
        surface = pickle.load(handle)
    residues_a, residues_b = _load_atom_residues(atom_graph_path)
    return surface, residues_a, residues_b


def _save_patch_graph(graph, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)


def build_interface_patch_graph_from_files(
    surface_path, atom_graph_path, output_path, config=None, overwrite=False
):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return None, "skipped"
    surface, residues_a, residues_b = load_patch_inputs(
        surface_path, atom_graph_path
    )
    graph = build_interface_patch_graph(
        surface, residues_a, residues_b, config=config
    )
    _save_patch_graph(graph, output_path)
    return graph, "built"


def build_interface_patch_graph_from_ply(
    surface_dir,
    pdb_id,
    atom_graph_path,
    output_path,
    surface_index=None,
    config=None,
    overwrite=False,
):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return None, "skipped"
    surface, _ = load_complete_ply_surface(
        surface_dir, pdb_id, surface_index=surface_index
    )
    residues_a, residues_b = _load_atom_residues(atom_graph_path)
    graph = build_interface_patch_graph(
        surface, residues_a, residues_b, config=config
    )
    _save_patch_graph(graph, output_path)
    return graph, "built"


def write_patch_metadata(out_root, config=None, graph_name="interface_patch_full"):
    config = config or PatchBuildConfig()
    directory = Path(out_root) / "surfgraph_4"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": PATCH_GRAPH_VERSION,
        "graph_name": graph_name,
        "surface_input": "complete_ply",
        "node_dim": PATCH_NODE_DIM,
        "edge_dim": PATCH_EDGE_DIM,
        "node_features": list(PATCH_NODE_FEATURE_NAMES),
        "edge_features": list(PATCH_EDGE_FEATURE_NAMES),
        "interface_source_features": list(INTERFACE_SOURCE_NAMES),
        "interface_selection_summary": [
            "full_vertices_a", "full_vertices_b",
            "selected_vertices_a", "selected_vertices_b",
            *(
                "{}_vertices_{}".format(source, partner)
                for source in INTERFACE_SOURCE_NAMES
                for partner in ("a", "b")
            ),
        ],
        "config": asdict(config),
    }
    path = directory / "{}_metadata.json".format(graph_name)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path
