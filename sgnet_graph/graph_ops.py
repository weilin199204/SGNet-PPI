import math

import numpy as np
import torch

from .constants import ATOM_PAIRS


def distance(atom_a, atom_b):
    return math.sqrt(
        (float(atom_a["x"]) - float(atom_b["x"])) ** 2
        + (float(atom_a["y"]) - float(atom_b["y"])) ** 2
        + (float(atom_a["z"]) - float(atom_b["z"])) ** 2
    )


def find_residue_pairs(threshold, residues_a, residues_b):
    pairs = []
    for residue_a in residues_a:
        for residue_b in residues_b:
            matched = False
            for atom_a in residue_a["atoms"]:
                for atom_b in residue_b["atoms"]:
                    if distance(atom_a, atom_b) <= threshold:
                        pairs.append((residue_a, residue_b))
                        matched = True
                        break
                if matched:
                    break
    return pairs


def inter_edge_index(pairs, len_a):
    sources = []
    targets = []
    for residue_a, residue_b in pairs:
        sources.append(int(residue_a["number"]))
        targets.append(int(residue_b["number"]) + len_a)
    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    source = torch.tensor(sources, dtype=torch.long).unsqueeze(0)
    target = torch.tensor(targets, dtype=torch.long).unsqueeze(0)
    return torch.cat(
        (
            torch.cat((source, target), dim=1),
            torch.cat((target, source), dim=1),
        ),
        dim=0,
    )


def intra_edge_index(pairs):
    sources = []
    targets = []
    for residue_a, residue_b in pairs:
        source = int(residue_a["number"])
        target = int(residue_b["number"])
        if source == target:
            continue
        sources.append(source)
        targets.append(target)
    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor((sources, targets), dtype=torch.long)


def encode_edge_features(pairs, max_distance, bins=10, duplicate_for_bidirectional=False):
    feature_rows = []
    pair_count = len(ATOM_PAIRS)
    for residue_a, residue_b in pairs:
        encoding = np.zeros(pair_count * bins, dtype=np.float32)
        for atom_a in residue_a["atoms"]:
            for atom_b in residue_b["atoms"]:
                dis = distance(atom_a, atom_b)
                bin_id = math.ceil(dis / (max_distance / bins))
                bin_id = min(max(bin_id, 1), bins)
                type_a = str(atom_a["pdbqt_type"])
                type_b = str(atom_b["pdbqt_type"])
                pair_type = "{}_{}".format(type_a, type_b)
                if pair_type not in ATOM_PAIRS:
                    pair_type = "{}_{}".format(type_b, type_a)
                if pair_type not in ATOM_PAIRS:
                    continue
                index = (bin_id - 1) * pair_count + ATOM_PAIRS.index(pair_type)
                encoding[index] += 1.0
        feature_rows.append(torch.from_numpy(encoding))
    if not feature_rows:
        data = torch.empty((0, pair_count * bins), dtype=torch.float32)
    else:
        data = torch.stack(feature_rows, dim=0)
    if duplicate_for_bidirectional:
        data = torch.cat((data, data), dim=0)
    return data
