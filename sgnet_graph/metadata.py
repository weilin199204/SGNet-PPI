import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    pdb_id: str
    ligand_chains: tuple
    receptor_chains: tuple
    label: float = None


def normalize_pdb_id(value):
    name = Path(str(value).strip()).name
    lower_name = name.lower()
    for suffix in (".ent.pdb", ".pdbqt", ".pdb"):
        if lower_name.endswith(suffix):
            return name[: -len(suffix)].upper()
    return name.upper()


def parse_chain_list(value):
    if isinstance(value, str):
        text = value.strip().strip('"').strip("'")
        if not text or text.upper() == "#N/A":
            return tuple()
        parts = [p for p in re.split(r"[,;\s]+", text) if p]
    else:
        parts = list(value)

    chains = []
    for part in parts:
        clean = re.sub(r"[^A-Za-z0-9]", "", str(part))
        if not clean:
            continue
        if len(clean) == 1:
            chains.append(clean)
        else:
            chains.extend(list(clean))
    return tuple(dict.fromkeys(chains))


def _find_column(row, names):
    lower = {k.lower().strip(): k for k in row}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return row[key]
    return None


def read_metadata_csv(path, allow_missing_labels=False):
    path = Path(path)
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(handle, dialect=dialect)
        entries = []
        for row in reader:
            pdb = _find_column(row, ("PDB", "PDB File", "Filename", "pdb_id"))
            ligand = _find_column(row, ("Ligand Chains", "ligand_chains", "chain_a"))
            receptor = _find_column(row, ("Receptor Chains", "receptor_chains", "chain_b"))
            label_text = _find_column(row, ("label", "KD(M)", "Kd", "y"))

            if not pdb:
                continue
            ligand_chains = parse_chain_list(ligand or "")
            receptor_chains = parse_chain_list(receptor or "")
            if not ligand_chains or not receptor_chains:
                continue

            label = None
            if label_text not in (None, ""):
                label = float(label_text)
            elif not allow_missing_labels:
                raise ValueError(
                    "Missing label for {}. Pass --allow-missing-labels for inference graphs.".format(pdb)
                )

            entries.append(
                Entry(
                    pdb_id=normalize_pdb_id(pdb),
                    ligand_chains=ligand_chains,
                    receptor_chains=receptor_chains,
                    label=label,
                )
            )
    return entries


def read_chain_index_with_labels(chain_index, label_index=None, allow_missing_labels=False):
    labels = {}
    if label_index:
        with Path(label_index).open() as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 2:
                    labels[normalize_pdb_id(fields[0])] = float(fields[1])

    entries = []
    with Path(chain_index).open() as handle:
        for line in handle:
            if not line.strip() or "\t" not in line:
                continue
            pdb_text, chain_text = line.rstrip("\n").split("\t", 1)
            pdb_id = normalize_pdb_id(pdb_text)
            if pdb_id.lower() in {"filename", "pdb"}:
                continue
            groups = [g for g in chain_text.split(";") if g.strip()]
            if len(groups) != 2:
                continue
            label = labels.get(pdb_id)
            if label is None and not allow_missing_labels:
                raise ValueError(
                    "Missing label for {}. Pass --allow-missing-labels for inference graphs.".format(pdb_id)
                )
            entries.append(
                Entry(
                    pdb_id=pdb_id,
                    ligand_chains=parse_chain_list(groups[0]),
                    receptor_chains=parse_chain_list(groups[1]),
                    label=label,
                )
            )
    return entries
