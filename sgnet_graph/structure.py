"""PDB/PDBQT parsing with deterministic alternate-location handling."""

import copy
import re
from collections import OrderedDict
from pathlib import Path

from .constants import AA3_TO_1, STANDARD_RESIDUES


def infer_element(atom_name, element_field=""):
    explicit = re.sub(r"[^A-Za-z]", "", str(element_field)).upper()
    if explicit in {"C", "N", "O", "S", "P", "H", "SE"}:
        return explicit
    name = re.sub(r"^[0-9]+", "", str(atom_name).strip()).upper()
    if name.startswith("SE"):
        return "SE"
    return name[:1] or "C"


def infer_autodock_type(line, atom_name, element=None):
    raw = line[77:79].strip() if len(line) >= 79 else ""
    if raw and raw.upper() not in {"C", "N", "O", "S", "P", "H", "SE"}:
        return raw
    element = (element or infer_element(atom_name)).upper()
    if element == "C":
        return "C"
    if element == "O":
        return "OA"
    if element == "N":
        return "N"
    if element in {"S", "SE"}:
        return "SA"
    if element == "H":
        return "HD"
    return element[:2] or "C"


def _float_field(text, default):
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return float(default)


def _altloc_priority(atom):
    altloc = atom.get("altloc", "")
    preferred = 2 if altloc == "" else (1 if altloc == "A" else 0)
    return (float(atom.get("occupancy", 0.0)), preferred, -int(atom.get("serial", 0)))


def parse_residue_chains(structure_file, chains):
    """Parse standard amino acids and retain one deterministic atom conformer."""
    wanted = set(chains)
    grouped = OrderedDict()
    structure_file = Path(structure_file)

    with structure_file.open(errors="ignore") as handle:
        for line in handle:
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM"} or len(line) < 54:
                continue
            chain = line[21].strip()
            if chain not in wanted:
                continue
            resname = line[17:20].strip().upper()
            if resname not in STANDARD_RESIDUES:
                continue
            atom_name = line[12:16].strip().upper()
            altloc = line[16].strip().upper()
            resseq_text = line[22:26].strip()
            icode = line[26].strip().upper()
            try:
                resseq = int(resseq_text)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            occupancy = _float_field(line[54:60] if len(line) >= 60 else "", 1.0)
            bfactor = _float_field(line[60:66] if len(line) >= 66 else "", 0.0)
            serial = int(line[6:11].strip() or 0)
            element = infer_element(
                atom_name, line[76:78] if structure_file.suffix.lower() != ".pdbqt" else ""
            )
            atom = {
                "type": atom_name,
                "pdbqt_type": infer_autodock_type(line, atom_name, element),
                "element": element,
                "x": x,
                "y": y,
                "z": z,
                "occupancy": occupancy,
                "bfactor": bfactor,
                "altloc": altloc,
                "serial": serial,
            }
            key = (chain, resseq, icode, resname)
            atoms = grouped.setdefault(key, OrderedDict())
            current = atoms.get(atom_name)
            if current is None or _altloc_priority(atom) > _altloc_priority(current):
                atoms[atom_name] = atom

    by_chain = {chain: [] for chain in chains}
    for (chain, resseq, icode, resname), atoms_by_name in grouped.items():
        atoms = list(atoms_by_name.values())
        if not any(atom["type"] == "CA" for atom in atoms):
            continue
        by_chain[chain].append(
            {
                "type": AA3_TO_1.get(resname, "X"),
                "resname": resname,
                "resseq": resseq,
                "icode": icode,
                "residue_id": "{}:{}{}".format(chain, resseq, icode),
                "number": len(by_chain[chain]),
                "atoms": copy.deepcopy(atoms),
                "chain": chain,
            }
        )
    return [by_chain[chain] for chain in chains if by_chain.get(chain)]


def concat_and_reindex(chain_residue_lists):
    merged = []
    sequences = []
    offset = 0
    for residues in chain_residue_lists:
        seq = "".join(str(residue["type"]) for residue in residues)
        if not seq:
            continue
        sequences.append(seq)
        for index, residue in enumerate(copy.deepcopy(residues)):
            residue["number"] = offset + index
            merged.append(residue)
        offset += len(seq)
    return merged, sequences

