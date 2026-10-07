import csv
import pickle
from pathlib import Path

import numpy as np
import torch

SURFACE_FEATURE_NAMES = ("charge", "hbond", "hphob")
NORMAL_NAMES = ("nx", "ny", "nz")


def _surface_dependencies():
    try:
        from plyfile import PlyData
        from torch_geometric.data import Data
        from torch_geometric.transforms import Cartesian, FaceToEdge
    except ImportError as exc:
        raise ImportError(
            "Surface graph building requires plyfile and torch_geometric. "
            "Install those dependencies before using --surface-dir."
        ) from exc
    return PlyData, Data, FaceToEdge, Cartesian


def _vertex_property_names(ply_data):
    return [prop.name for prop in ply_data["vertex"].properties]


def _face_tensor(face_values):
    faces = [torch.as_tensor(face, dtype=torch.long) for face in face_values if len(face) == 3]
    if not faces:
        return torch.empty((3, 0), dtype=torch.long)
    return torch.stack(faces, dim=-1)


def _data_from_parts(vertex_data, face_values=None):
    _, Data, FaceToEdge, Cartesian = _surface_dependencies()
    required = ("x", "y", "z") + SURFACE_FEATURE_NAMES
    missing = [name for name in required if name not in vertex_data]
    if missing:
        raise ValueError("Surface PLY missing properties: {}".format(", ".join(missing)))

    pos = torch.stack([torch.as_tensor(vertex_data[name]) for name in ("x", "y", "z")], dim=-1).float()
    x = torch.stack([torch.as_tensor(vertex_data[name]) for name in SURFACE_FEATURE_NAMES], dim=-1).float()
    if all(name in vertex_data for name in NORMAL_NAMES):
        normal = torch.stack([torch.as_tensor(vertex_data[name]) for name in NORMAL_NAMES], dim=-1).float()
        normal = torch.nn.functional.normalize(torch.nan_to_num(normal), dim=-1)
    else:
        normal = torch.zeros_like(pos)

    face = _face_tensor([] if face_values is None else face_values)
    vertex_area = torch.zeros(pos.shape[0], 1, dtype=torch.float32)
    if face.numel():
        p0, p1, p2 = pos[face[0]], pos[face[1]], pos[face[2]]
        triangle_area = 0.5 * torch.linalg.vector_norm(
            torch.linalg.cross(p1 - p0, p2 - p0, dim=-1), dim=-1
        )
        for row in face:
            vertex_area[:, 0].index_add_(0, row, triangle_area / 3.0)
    data = Data(x=torch.nan_to_num(x), pos=pos, normal=normal,
                vertex_area=vertex_area, face=face)
    data = FaceToEdge()(data)
    data = Cartesian(cat=False)(data)
    return data


def read_surface_data(ply_data):
    vertex_data = {
        name: torch.as_tensor(ply_data["vertex"][name])
        for name in _vertex_property_names(ply_data)
    }
    face_values = ply_data["face"]["vertex_indices"] if "face" in ply_data else None
    return _data_from_parts(vertex_data, face_values)


def read_surface_dict(data):
    return _data_from_parts(data["vertex"], data.get("face", {}).get("vertex_indices"))


_SURFACE_INDEX_CACHE = {}


def canonical_sample_id(value):
    parts = str(value).rsplit("_", 2)
    return (parts[0] if len(parts) == 3 else str(value)).upper()


def load_surface_index(surface_index):
    if surface_index is None:
        return {}
    surface_index = Path(surface_index)
    key = str(surface_index)
    if key in _SURFACE_INDEX_CACHE:
        return _SURFACE_INDEX_CACHE[key]

    mapping = {}
    with surface_index.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="	")
        for row in reader:
            dataset_entry = row.get("dataset_entry")
            ligand_ply = row.get("ligand_ply")
            receptor_ply = row.get("receptor_ply")
            if dataset_entry and ligand_ply and receptor_ply:
                mapping[canonical_sample_id(dataset_entry)] = (Path(ligand_ply), Path(receptor_ply))
    _SURFACE_INDEX_CACHE[key] = mapping
    return mapping


def find_surface_files(surface_dir, pdb_id, surface_index=None):
    indexed = load_surface_index(surface_index).get(str(pdb_id).upper())
    if indexed:
        missing = [path for path in indexed if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Surface index paths for {} do not exist: {}".format(
                    pdb_id, ",".join(str(path) for path in missing)
                )
            )
        return list(indexed)

    surface_dir = Path(surface_dir)
    files = sorted(surface_dir.glob("{}_*.ply".format(pdb_id)))
    if len(files) != 2:
        raise FileNotFoundError(
            "Expected exactly two surface PLY files for {}, found {} under {}".format(
                pdb_id, len(files), surface_dir
            )
        )
    return files


def read_surface(surface_dir, pdb_id, surface_index=None):
    PlyData, _, _, _ = _surface_dependencies()
    surface = {}
    raw_ply = {}
    for index, surface_file in enumerate(find_surface_files(surface_dir, pdb_id, surface_index=surface_index)):
        with surface_file.open("rb") as handle:
            ply_data = PlyData.read(handle)
        key = "protein_{}".format(index)
        raw_ply[key] = ply_data
        surface[key] = read_surface_data(ply_data)
    return surface, raw_ply


def build_surface_conv_graph(data, protein_id):
    surface = data[protein_id]
    node_attr = torch.nan_to_num(surface.x)
    return node_attr, surface.edge_index, surface.edge_attr.float()


def crop_surface(surface, vertex_ids):
    """Return the induced patch while preserving original boundary mesh edges."""
    _, Data, _, _ = _surface_dependencies()
    ids = torch.as_tensor(vertex_ids, dtype=torch.long)
    old_to_new = torch.full((surface.num_nodes,), -1, dtype=torch.long)
    old_to_new[ids] = torch.arange(ids.numel())
    src, dst = surface.edge_index
    keep = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)
    return Data(
        x=surface.x[ids], pos=surface.pos[ids], normal=surface.normal[ids],
        vertex_area=surface.vertex_area[ids],
        edge_index=old_to_new[surface.edge_index[:, keep]],
        edge_attr=surface.edge_attr[keep], original_vertex_id=ids,
    )


def _nearby_vertex_ids(pos_a, pos_b, cutoff):
    ids_a = set()
    ids_b = set()
    for start in range(0, pos_a.shape[0], 4096):
        chunk = pos_a[start : start + 4096]
        distances = torch.cdist(chunk, pos_b)
        rows, cols = torch.nonzero(distances <= cutoff, as_tuple=True)
        ids_a.update((rows + start).tolist())
        ids_b.update(cols.tolist())
    return sorted(ids_a), sorted(ids_b)


def get_interaction_residue_id(dis_thred, pos1, pos2):
    return _nearby_vertex_ids(pos1, pos2, dis_thred)


def get_new_data(ply_data, vertex_ids):
    vertex_id_set = set(vertex_ids)
    id_map = {old_id: new_id for new_id, old_id in enumerate(vertex_ids)}
    vertex = {}
    for name in _vertex_property_names(ply_data):
        vertex[name] = torch.as_tensor(ply_data["vertex"][name][vertex_ids])

    if "face" not in ply_data:
        return {"vertex": vertex}

    faces = []
    for face in ply_data["face"]["vertex_indices"]:
        face_ids = [int(idx) for idx in face]
        if set(face_ids).issubset(vertex_id_set):
            faces.append(np.asarray([id_map[idx] for idx in face_ids], dtype=np.int32))
    return {"vertex": vertex, "face": {"vertex_indices": faces}}


def get_new_surface(surface, cross_distance_cutoff=5.0):
    ids_a, ids_b = get_interaction_residue_id(
        cross_distance_cutoff, surface["protein_0"].pos, surface["protein_1"].pos
    )
    if not ids_a or not ids_b:
        raise ValueError("No interacting surface vertices found within {}".format(cross_distance_cutoff))

    return {
        "protein_0": crop_surface(surface["protein_0"], ids_a),
        "protein_1": crop_surface(surface["protein_1"], ids_b),
    }


def get_interaction_residue_pair(dis_thred, pos1, pos2, max_neighbors=10):
    pairs = []
    for index in range(pos1.shape[0]):
        distances = torch.norm(pos2 - pos1[index].unsqueeze(0), dim=-1)
        hits = torch.nonzero(distances <= dis_thred, as_tuple=False).flatten()
        if hits.numel():
            hit_distances = distances[hits]
            order = torch.argsort(hit_distances)[:max_neighbors]
            for target, distance in zip(hits[order].tolist(), hit_distances[order].tolist()):
                pairs.append((index, target, float(distance)))
    return pairs


def get_edge_index_attr(inter_pairs, pos_A, pos_B):
    if not inter_pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )

    source_list = []
    des_list = []
    edge_attr_list_a = []
    edge_attr_list_b = []
    for source_res, des_res_raw, distance in inter_pairs:
        des_res = int(des_res_raw) + pos_A.shape[0]
        source_list.append(int(source_res))
        des_list.append(des_res)
        signed_weight = 1.0 / max(float(distance) ** 2, 1e-6)
        edge_attr_list_a.append(torch.tensor([signed_weight], dtype=torch.float32))
        edge_attr_list_b.append(torch.tensor([-signed_weight], dtype=torch.float32))

    source = torch.tensor(source_list, dtype=torch.long).unsqueeze(0)
    des = torch.tensor(des_list, dtype=torch.long).unsqueeze(0)
    edge_index = torch.cat((torch.cat((source, des), dim=1), torch.cat((des, source), dim=1)), dim=0)
    edge_attr = torch.cat((torch.stack(edge_attr_list_a), torch.stack(edge_attr_list_b)), dim=0)
    return edge_index, edge_attr.float()


def get_new_pos(data, ids):
    return [data.pos[index] for index in ids]


def get_new_x(data, ids):
    return [data.x[index] for index in ids]


def build_surface_cross_conv_graph(data, cross_distance_cutoff=5.0):
    ids_a, ids_b = get_interaction_residue_id(
        cross_distance_cutoff, data["protein_0"].pos, data["protein_1"].pos
    )
    surf_a_pos = torch.stack(get_new_pos(data["protein_0"], ids_a))
    surf_b_pos = torch.stack(get_new_pos(data["protein_1"], ids_b))
    surf_a_x = torch.stack(get_new_x(data["protein_0"], ids_a))
    surf_b_x = torch.stack(get_new_x(data["protein_1"], ids_b))
    pairs = get_interaction_residue_pair(cross_distance_cutoff, surf_a_pos, surf_b_pos)
    edge_index, edge_attr = get_edge_index_attr(pairs, surf_a_pos, surf_b_pos)
    return torch.cat([surf_a_x, surf_b_x], dim=0), edge_index, edge_attr


def build_surface_cross_conv_graph2(data, cross_distance_cutoff=5.0):
    edge_x = torch.cat([data["protein_0"].x, data["protein_1"].x], dim=0)
    pos = torch.cat([data["protein_0"].pos, data["protein_1"].pos], dim=0)  # new: combined pos
    pairs = get_interaction_residue_pair(
        cross_distance_cutoff, data["protein_0"].pos, data["protein_1"].pos
    )
    edge_index, edge_attr = get_edge_index_attr(pairs, data["protein_0"].pos, data["protein_1"].pos)
    return edge_x, edge_index, edge_attr, pos  # new: return pos


def _rbf(distance, cutoff, bins=8):
    centers = torch.linspace(0.0, float(cutoff), bins, device=distance.device)
    spacing = float(cutoff) / max(bins - 1, 1)
    return torch.exp(-(distance.unsqueeze(-1) - centers) ** 2 / max(spacing ** 2, 1e-6))


def _typed_edge_features(edge_index, edge_type, pos, normal, x, cutoff, rbf_bins=8):
    src, dst = edge_index
    vector = pos[dst] - pos[src]
    distance = torch.linalg.vector_norm(vector, dim=-1)
    direction = vector / distance.unsqueeze(-1).clamp_min(1e-8)
    orientation = torch.stack([
        (normal[src] * normal[dst]).sum(-1),
        (normal[src] * direction).sum(-1),
        (normal[dst] * -direction).sum(-1),
    ], dim=-1)
    edge_attr = torch.cat([
        torch.nn.functional.one_hot(edge_type, num_classes=2).float(),
        _rbf(distance, cutoff, rbf_bins),
        (distance / cutoff).unsqueeze(-1),
        orientation,
        x[src] * x[dst],
    ], dim=-1)
    return torch.nan_to_num(edge_attr), torch.nan_to_num(vector)


def build_interface_surface_graph(data, cross_cutoff=5.0, max_neighbors=10, rbf_bins=8):
    """One graph with local mesh context and explicit cross-interface interactions."""
    _, Data, _, _ = _surface_dependencies()
    surf_a, surf_b = data["protein_0"], data["protein_1"]
    n_a = surf_a.num_nodes
    x = torch.cat([surf_a.x, surf_b.x])
    pos = torch.cat([surf_a.pos, surf_b.pos])
    normal = torch.cat([surf_a.normal, surf_b.normal])
    vertex_area = torch.cat([surf_a.vertex_area, surf_b.vertex_area])
    chain_id = torch.cat([
        torch.zeros(n_a, dtype=torch.long),
        torch.ones(surf_b.num_nodes, dtype=torch.long),
    ])
    mesh_a = surf_a.edge_index
    mesh_b = surf_b.edge_index + n_a
    pairs_ab = get_interaction_residue_pair(
        cross_cutoff, surf_a.pos, surf_b.pos, max_neighbors=max_neighbors)
    pairs_ba = get_interaction_residue_pair(
        cross_cutoff, surf_b.pos, surf_a.pos, max_neighbors=max_neighbors)
    pair_index = [[i, n_a + j] for i, j, _ in pairs_ab]
    pair_index += [[n_a + i, j] for i, j, _ in pairs_ba]
    cross = (torch.tensor(pair_index, dtype=torch.long).T if pair_index
             else torch.empty((2, 0), dtype=torch.long))
    edge_index = torch.cat([mesh_a, mesh_b, cross], dim=1)
    edge_type = torch.cat([
        torch.zeros(mesh_a.shape[1] + mesh_b.shape[1], dtype=torch.long),
        torch.ones(cross.shape[1], dtype=torch.long),
    ])
    edge_attr, edge_vector = _typed_edge_features(
        edge_index, edge_type, pos, normal, x, cross_cutoff, rbf_bins)
    core_a, core_b = _nearby_vertex_ids(surf_a.pos, surf_b.pos, cross_cutoff)
    core_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    core_mask[torch.as_tensor(core_a, dtype=torch.long)] = True
    core_mask[n_a + torch.as_tensor(core_b, dtype=torch.long)] = True
    return Data(
        x=x, pos=pos, normal=normal, vertex_area=vertex_area,
        chain_id=chain_id, core_mask=core_mask,
        edge_index=edge_index, edge_type=edge_type,
        edge_attr=edge_attr, edge_vector=edge_vector,
    )


def save_pickle(path, obj, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def build_surface_graphs_for_entry(
    pdb_id,
    surface_dir,
    out_dir,
    surface_index=None,
    preselect_cutoff=8.0,
    cross_cutoff=5.0,
    max_cross_neighbors=10,
    rbf_bins=8,
    overwrite=False,
):
    _, Data, _, _ = _surface_dependencies()
    surface, _ = read_surface(surface_dir, pdb_id, surface_index=surface_index)
    new_surface = get_new_surface(surface, cross_distance_cutoff=preselect_cutoff)

    intra_surf1 = build_surface_conv_graph(new_surface, "protein_0")
    intra_surf2 = build_surface_conv_graph(new_surface, "protein_1")
    inter_surf = build_surface_cross_conv_graph2(new_surface, cross_distance_cutoff=cross_cutoff)
    interface_surface = build_interface_surface_graph(
        new_surface, cross_cutoff=cross_cutoff,
        max_neighbors=max_cross_neighbors, rbf_bins=rbf_bins)

    out_dir = Path(out_dir)
    outputs = {
        "intra_surf1": Data(x=intra_surf1[0], edge_index=intra_surf1[1],
                            edge_attr=intra_surf1[2],
                            pos=new_surface["protein_0"].pos,
                            normal=new_surface["protein_0"].normal,
                            vertex_area=new_surface["protein_0"].vertex_area),
        "intra_surf2": Data(x=intra_surf2[0], edge_index=intra_surf2[1],
                            edge_attr=intra_surf2[2],
                            pos=new_surface["protein_1"].pos,
                            normal=new_surface["protein_1"].normal,
                            vertex_area=new_surface["protein_1"].vertex_area),
        "inter_surf": Data(x=inter_surf[0], edge_index=inter_surf[1],
                           edge_attr=inter_surf[2], pos=inter_surf[3]),
        "interface_surface": interface_surface,
    }
    for name, graph in outputs.items():
        graph_version = "surfgraph_3" if name == "interface_surface" else "surfgraph_2"
        save_pickle(out_dir / graph_version / name / pdb_id, graph, overwrite=overwrite)
    return outputs
