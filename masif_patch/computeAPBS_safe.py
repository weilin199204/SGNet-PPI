"""APBS wrapper that records and zero-fills only unrecoverable electrostatics."""
from __future__ import absolute_import, print_function

import importlib.util
import os
from pathlib import Path

import numpy as np

MASIF_ROOT = os.environ.get("SGNET_MASIF_ROOT")
if not MASIF_ROOT:
    raise RuntimeError(
        "SGNET_MASIF_ROOT must point to the MaSIF checkout before this module is loaded")
ORIGINAL = Path(MASIF_ROOT).resolve() / "source" / "triangulation" / "computeAPBS.py"
if not ORIGINAL.is_file():
    raise FileNotFoundError(ORIGINAL)
_spec = importlib.util.spec_from_file_location("_masif_computeAPBS_original", str(ORIGINAL))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def computeAPBS(vertices, pdb_file, tmp_file_base):
    try:
        return _module.computeAPBS(vertices, pdb_file, tmp_file_base)
    except Exception as exc:
        marker = os.environ.get("SGNET_APBS_FALLBACK_MARKER")
        message = "{}: {}".format(type(exc).__name__, exc)
        if marker:
            path = Path(marker)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("pdb_file={}\nerror={}\npolicy=zero electrostatic channel\n".format(
                pdb_file, message))
        print("[APBS fallback] {} -> zero electrostatic channel".format(message),
              flush=True)
        return np.zeros((len(vertices),), dtype=np.float64)
