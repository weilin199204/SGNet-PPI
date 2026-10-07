import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from .esm import EsmEmbedder
from .graph_ops import (
    encode_edge_features,
    find_residue_pairs,
    inter_edge_index,
    intra_edge_index,
)
from .metadata import Entry, normalize_pdb_id
from .structure import concat_and_reindex, parse_residue_chains


@dataclass
class BuildConfig:
    pdb_dir: Path
    out_dir: Path
    pdbqt_dir: Path = None
    surface_dir: Path = None
    surface_index: Path = None
    pythonsh: Path = None
    prepare_receptor4: Path = None
    stage: str = "all"
    allow_pdb_fallback: bool = True
    inter_threshold: float = 15.0
    interface_mask_threshold: float = 6.0
    intra_threshold: float = 3.5
    surface_preselect_cutoff: float = 8.0
    surface_cross_cutoff: float = 5.0
    surface_max_neighbors: int = 10
    surface_rbf_bins: int = 8
    device: str = "cpu"
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    limit: int = None
    overwrite: bool = False


@dataclass
class BuildResult:
    built: list
    failed: list


def find_pdb_file(pdb_dir, pdb_id):
    for suffix in (".pdb", ".ent.pdb"):
        candidate = pdb_dir / "{}{}".format(pdb_id, suffix)
        if candidate.exists():
            return candidate
    matches = list(pdb_dir.glob("{}*.pdb".format(pdb_id)))
    if matches:
        return matches[0]

    prefix = str(pdb_id).lower()
    case_matches = []
    for path in pdb_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith(prefix) and name.endswith(".pdb"):
            case_matches.append(path)
    return sorted(case_matches, key=lambda path: path.name.lower())[0] if case_matches else None


def find_pdbqt_file(pdbqt_dir, pdb_id):
    if not pdbqt_dir:
        return None
    candidate = pdbqt_dir / "{}.pdbqt".format(pdb_id)
    return candidate if candidate.exists() else None


def convert_to_pdbqt(pdb_file, pdbqt_file, config):
    if not config.pythonsh or not config.prepare_receptor4:
        raise FileNotFoundError(
            "{} does not exist and PDBQT conversion was not configured.".format(pdbqt_file)
        )
    pdbqt_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.pythonsh),
        str(config.prepare_receptor4),
        "-r",
        str(pdb_file),
        "-A",
        "hydrogens",
        "-o",
        str(pdbqt_file),
    ]
    subprocess.run(command, check=True)
    return pdbqt_file


def select_structure_file(entry, config):
    pdb_file = find_pdb_file(config.pdb_dir, entry.pdb_id)
    if pdb_file is None:
        raise FileNotFoundError("No PDB file found for {}".format(entry.pdb_id))

    pdbqt_file = find_pdbqt_file(config.pdbqt_dir, entry.pdb_id)
    if pdbqt_file:
        return pdbqt_file

    if config.pdbqt_dir and config.pythonsh and config.prepare_receptor4:
        return convert_to_pdbqt(pdb_file, config.pdbqt_dir / "{}.pdbqt".format(entry.pdb_id), config)

    if config.allow_pdb_fallback:
        return pdb_file
    raise FileNotFoundError("No PDBQT file found for {}".format(entry.pdb_id))


def extract_ca_coords(residues):
    """Extract Cα coordinates from a list of residue dicts. Returns (N, 3) tensor."""
    coords = []
    for res in residues:
        for atom in res["atoms"]:
            if atom["type"] == "CA":
                coords.append([float(atom["x"]), float(atom["y"]), float(atom["z"])])
                break
    if not coords:
        return torch.zeros((len(residues), 3), dtype=torch.float32)
    return torch.tensor(coords, dtype=torch.float32)


def build_intermediate(entry, structure_file, config):
    ligand_chains = parse_residue_chains(structure_file, entry.ligand_chains)
    receptor_chains = parse_residue_chains(structure_file, entry.receptor_chains)
    if not ligand_chains or not receptor_chains:
        raise ValueError("Could not parse requested ligand/receptor chains.")

    residues_a, seqs_a = concat_and_reindex(ligand_chains)
    residues_b, seqs_b = concat_and_reindex(receptor_chains)
    if not residues_a or not residues_b:
        raise ValueError("Empty residue list after parsing chains.")

    inter_pairs = find_residue_pairs(config.inter_threshold, residues_a, residues_b)
    inter_info = (
        inter_pairs,
        inter_edge_index(inter_pairs, sum(len(seq) for seq in seqs_a)),
        (seqs_a, seqs_b),
        (residues_a, residues_b),  # new: residue lists for CA coord extraction
    )

    intra_pairs_a = [
        pair
        for pair in find_residue_pairs(config.intra_threshold, residues_a, residues_a)
        if int(pair[0]["number"]) != int(pair[1]["number"])
    ]
    intra_pairs_b = [
        pair
        for pair in find_residue_pairs(config.intra_threshold, residues_b, residues_b)
        if int(pair[0]["number"]) != int(pair[1]["number"])
    ]
    intra_a = (intra_pairs_a, intra_edge_index(intra_pairs_a), seqs_a, residues_a)  # new: residues_a
    intra_b = (intra_pairs_b, intra_edge_index(intra_pairs_b), seqs_b, residues_b)  # new: residues_b
    return inter_info, intra_a, intra_b


def save_pickle(path, obj, overwrite=False):
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def make_inter_data(entry, inter_info, embedder, config):
    from torch_geometric.data import Data
    seqs_a, seqs_b = inter_info[2]
    residues_a, residues_b = inter_info[3]  # new
    x_a = embedder.embed_sequences(list(seqs_a))
    x_b = embedder.embed_sequences(list(seqs_b))
    edge_attr = encode_edge_features(
        inter_info[0],
        max_distance=config.inter_threshold,
        duplicate_for_bidirectional=True,
    )
    pos = torch.cat(
        [extract_ca_coords(residues_a), extract_ca_coords(residues_b)], dim=0
    )  # new: (Na+Nb, 3)
    data = Data(
        x=torch.cat((x_a, x_b), dim=0),
        edge_index=inter_info[1],
        edge_attr=edge_attr,
        pos=pos,  # new
    )
    if entry.label is not None:
        data.y = torch.tensor([round(float(entry.label), 2)], dtype=torch.float32)
    return data


def make_intra_data(intra_info, embedder, config, interface_residue_indices=None):
    from torch_geometric.data import Data
    residues = intra_info[3]  # new
    x = embedder.embed_sequences(list(intra_info[2]))
    edge_attr = encode_edge_features(
        intra_info[0],
        max_distance=config.intra_threshold,
        duplicate_for_bidirectional=False,
    )
    pos = extract_ca_coords(residues)  # new: (N, 3)
    N = x.shape[0]
    if interface_residue_indices is not None:
        mask = torch.zeros(N, dtype=torch.bool)
        for idx in interface_residue_indices:
            if idx < N:
                mask[idx] = True
    else:
        mask = torch.ones(N, dtype=torch.bool)
    return Data(
        x=x,
        edge_index=intra_info[1],
        edge_attr=edge_attr,
        pos=pos,              # new
        interface_mask=mask,  # new
    )


def build_graphs(entries, config):
    out = config.out_dir
    built = []
    failed = []
    selected = entries[: config.limit] if config.limit else entries
    embedder = None

    for entry in selected:
        entry = Entry(
            pdb_id=normalize_pdb_id(entry.pdb_id),
            ligand_chains=entry.ligand_chains,
            receptor_chains=entry.receptor_chains,
            label=entry.label,
        )
        try:
            if config.stage == "surface":
                if not config.surface_dir:
                    raise ValueError("--stage surface requires --surface-dir")
                from .surface import build_surface_graphs_for_entry

                build_surface_graphs_for_entry(
                    entry.pdb_id,
                    surface_dir=config.surface_dir,
                    out_dir=out,
                    surface_index=config.surface_index,
                    preselect_cutoff=config.surface_preselect_cutoff,
                    cross_cutoff=config.surface_cross_cutoff,
                    max_cross_neighbors=config.surface_max_neighbors,
                    rbf_bins=config.surface_rbf_bins,
                    overwrite=config.overwrite,
                )
                built.append(entry.pdb_id)
                continue

            structure_file = select_structure_file(entry, config)
            inter_info, intra_a, intra_b = build_intermediate(entry, structure_file, config)

            save_pickle(
                out / "graph_construct" / "inter_graph" / entry.pdb_id,
                inter_info,
                overwrite=config.overwrite,
            )
            save_pickle(
                out / "graph_construct" / "individual_graph" / "{}_1".format(entry.pdb_id),
                intra_a,
                overwrite=config.overwrite,
            )
            save_pickle(
                out / "graph_construct" / "individual_graph" / "{}_2".format(entry.pdb_id),
                intra_b,
                overwrite=config.overwrite,
            )

            if config.stage in {"graph", "all"}:
                if embedder is None:
                    embedder = EsmEmbedder(model_name=config.esm_model, device=config.device)

                # Compute interface residue indices for masked pooling in sgnet2
                residues_a = intra_a[3]
                residues_b = intra_b[3]
                interface_pairs = find_residue_pairs(config.interface_mask_threshold, residues_a, residues_b)
                # Build sets of residue numbers that appear in inter-chain contacts
                interface_idx_a = {res_a["number"] for res_a, _ in interface_pairs}
                interface_idx_b = {res_b["number"] for _, res_b in interface_pairs}
                # Map residue numbers to positional indices within each chain
                num_to_idx_a = {res["number"]: i for i, res in enumerate(residues_a)}
                num_to_idx_b = {res["number"]: i for i, res in enumerate(residues_b)}
                iface_pos_a = [num_to_idx_a[n] for n in interface_idx_a if n in num_to_idx_a]
                iface_pos_b = [num_to_idx_b[n] for n in interface_idx_b if n in num_to_idx_b]

                save_pickle(
                    out / "graph" / "inter_graph" / entry.pdb_id,
                    make_inter_data(entry, inter_info, embedder, config),
                    overwrite=config.overwrite,
                )
                save_pickle(
                    out / "graph" / "individual_graph" / "{}_1".format(entry.pdb_id),
                    make_intra_data(intra_a, embedder, config,
                                    interface_residue_indices=iface_pos_a),
                    overwrite=config.overwrite,
                )
                save_pickle(
                    out / "graph" / "individual_graph" / "{}_2".format(entry.pdb_id),
                    make_intra_data(intra_b, embedder, config,
                                    interface_residue_indices=iface_pos_b),
                    overwrite=config.overwrite,
                )

            if config.surface_dir:
                from .surface import build_surface_graphs_for_entry

                build_surface_graphs_for_entry(
                    entry.pdb_id,
                    surface_dir=config.surface_dir,
                    out_dir=out,
                    surface_index=config.surface_index,
                    preselect_cutoff=config.surface_preselect_cutoff,
                    cross_cutoff=config.surface_cross_cutoff,
                    max_cross_neighbors=config.surface_max_neighbors,
                    rbf_bins=config.surface_rbf_bins,
                    overwrite=config.overwrite,
                )

            built.append(entry.pdb_id)
        except Exception as exc:
            failed.append((entry.pdb_id, str(exc)))

    write_indexes(out, entries=selected, built=built, failed=failed)
    return BuildResult(built=built, failed=failed)


def write_indexes(out_dir, entries, built, failed):
    index_dir = out_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    entry_by_id = {entry.pdb_id: entry for entry in entries}
    with (index_dir / "built_index.tsv").open("w") as handle:
        handle.write("pdb_id\tligand_chains\treceptor_chains\tlabel\n")
        for pdb_id in built:
            entry = entry_by_id[pdb_id]
            label = "" if entry.label is None else str(entry.label)
            handle.write(
                "{}\t{}\t{}\t{}\n".format(
                    pdb_id,
                    ",".join(entry.ligand_chains),
                    ",".join(entry.receptor_chains),
                    label,
                )
            )
    with (index_dir / "failed.tsv").open("w") as handle:
        handle.write("pdb_id\treason\n")
        for pdb_id, reason in failed:
            handle.write("{}\t{}\n".format(pdb_id, reason))
