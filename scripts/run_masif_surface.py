#!/usr/bin/env python
"""Generate one complete external partner surface from a local PDB file."""
from __future__ import absolute_import, print_function

import argparse
import importlib.util
import os
import runpy
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", help="Entry such as 1WEJ_HL")
    parser.add_argument("--raw-pdb-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pdb-output-dir")
    parser.add_argument("--context-chains", default=None,
                        help="Restrict the staged complex to these chain IDs")
    parser.add_argument(
        "--disable-ply-iface",
        action="store_true",
        help=("Skip MaSIF optional bound-complex iface vertex label. "
              "The downstream patch builder derives the interface independently."))
    parser.add_argument(
        "--masif-root", required=True,
        help="Path to a MaSIF checkout containing source/ and data/masif_ppi_search/.")
    return parser.parse_args()


def stage_first_model(source, destination, context_chains=None):
    lines = source.read_text(errors="replace").splitlines(True)
    model_count = sum(line[:6].strip() == "MODEL" for line in lines)
    if model_count > 1:
        output = []
        in_first = False
        for line in lines:
            record = line[:6].strip()
            if record == "MODEL":
                if in_first:
                    break
                in_first = True
                continue
            if record == "ENDMDL" and in_first:
                break
            if not in_first:
                output.append(line)
            else:
                output.append(line)
        output.append("END\n")
    else:
        output = lines

    if context_chains:
        allowed = set(context_chains)
        filtered = []
        for line in output:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM", "ANISOU", "TER"}:
                chain_id = line[21:22].strip()
                if chain_id not in allowed:
                    continue
            filtered.append(line)
        output = filtered
    destination.write_text("".join(output))
    return model_count


def main():
    args = parse_args()
    masif_root = Path(args.masif_root).resolve()
    source_dir = masif_root / "source"
    work_dir = masif_root / "data" / "masif_ppi_search"
    raw_pdb_dir = Path(args.raw_pdb_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    pdb_output_dir = Path(
        args.pdb_output_dir or output_dir.parent / (output_dir.name + "_pdbs")
    ).resolve()
    pdb_id = args.entry.rsplit("_", 1)[0]
    raw_pdb = raw_pdb_dir / (pdb_id + ".pdb")
    if not raw_pdb.exists():
        raise FileNotFoundError(raw_pdb)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_dir.mkdir(parents=True, exist_ok=True)

    scratch_root = Path(
        os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.TemporaryDirectory(
        prefix="sgnet_surface_{}_".format(args.entry), dir=str(scratch_root))

    staged_raw_dir = Path(scratch.name) / "raw_pdb"
    staged_raw_dir.mkdir(parents=True, exist_ok=True)
    staged_raw = staged_raw_dir / (pdb_id + ".pdb")
    model_count = stage_first_model(
        raw_pdb, staged_raw, context_chains=args.context_chains)

    sys.path.insert(0, str(source_dir))
    from default_config.masif_opts import masif_opts
    masif_opts["tmp_dir"] = scratch.name
    masif_opts["raw_pdb_dir"] = str(staged_raw_dir) + os.sep
    masif_opts["ply_chain_dir"] = str(output_dir) + os.sep
    masif_opts["pdb_chain_dir"] = str(pdb_output_dir) + os.sep
    masif_opts["ply_file_template"] = str(output_dir / "{}_{}.ply")
    if args.disable_ply_iface:
        masif_opts["compute_iface"] = False

    local_patch_root = Path(__file__).resolve().parents[1] / "masif_patch"
    patch_path = local_patch_root / "computeMSMS_robust.py"
    spec = importlib.util.spec_from_file_location(
        "triangulation.computeMSMS", patch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["triangulation.computeMSMS"] = module

    apbs_patch_path = local_patch_root / "computeAPBS_safe.py"
    previous_masif_root = os.environ.get("SGNET_MASIF_ROOT")
    os.environ["SGNET_MASIF_ROOT"] = str(masif_root)
    apbs_spec = importlib.util.spec_from_file_location(
        "triangulation.computeAPBS", apbs_patch_path)
    apbs_module = importlib.util.module_from_spec(apbs_spec)
    apbs_spec.loader.exec_module(apbs_module)
    sys.modules["triangulation.computeAPBS"] = apbs_module
    script = source_dir / "data_preparation" / "01-pdb_extract_and_triangulate.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    fallback_dir = output_dir.parent / "fallbacks"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_marker = fallback_dir / (args.entry + ".apbs.txt")
    msms_marker = fallback_dir / (args.entry + ".msms.txt")
    model_marker = fallback_dir / (args.entry + ".model.txt")
    context_marker = fallback_dir / (args.entry + ".context.txt")
    iface_marker = fallback_dir / (args.entry + ".iface.txt")
    for marker in (fallback_marker, msms_marker, model_marker, context_marker,
                   iface_marker):
        if marker.exists():
            marker.unlink()

    if args.context_chains:
        context_marker.write_text(
            "source={}\ncontext_chains={}\npolicy=manifest partner union before protonation\n".format(
                raw_pdb, args.context_chains))

    if model_count > 1:
        model_marker.write_text(
            "source={}\nmodel_count={}\npolicy=first model before protonation\n".format(
                raw_pdb, model_count))

    if args.disable_ply_iface:
        iface_marker.write_text(
            "source={}\npolicy=skip optional MaSIF bound-complex iface label\n"
            "downstream=interface derived by patch builder\n".format(raw_pdb))

    previous_marker = os.environ.get("SGNET_APBS_FALLBACK_MARKER")
    previous_msms_marker = os.environ.get("SGNET_MSMS_FALLBACK_MARKER")
    os.environ["SGNET_APBS_FALLBACK_MARKER"] = str(fallback_marker)
    os.environ["SGNET_MSMS_FALLBACK_MARKER"] = str(msms_marker)
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(str(work_dir))
        sys.argv = [str(script), args.entry]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(str(previous_cwd))
        if previous_masif_root is None:
            os.environ.pop("SGNET_MASIF_ROOT", None)
        else:
            os.environ["SGNET_MASIF_ROOT"] = previous_masif_root
        if previous_marker is None:
            os.environ.pop("SGNET_APBS_FALLBACK_MARKER", None)
        else:
            os.environ["SGNET_APBS_FALLBACK_MARKER"] = previous_marker
        if previous_msms_marker is None:
            os.environ.pop("SGNET_MSMS_FALLBACK_MARKER", None)
        else:
            os.environ["SGNET_MSMS_FALLBACK_MARKER"] = previous_msms_marker
        scratch.cleanup()

    expected = output_dir / (args.entry + ".ply")
    if not expected.exists():
        raise FileNotFoundError("MaSIF did not produce {}".format(expected))
    print(expected, flush=True)


if __name__ == "__main__":
    main()
