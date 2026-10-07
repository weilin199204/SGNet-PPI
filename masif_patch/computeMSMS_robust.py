"""MSMS wrapper that retains disconnected molecular fragments safely."""

import glob
import os
import random
from subprocess import PIPE, Popen

import numpy as np
from sklearn.neighbors import KDTree

from default_config.global_vars import msms_bin
from default_config.masif_opts import masif_opts
from input_output.read_msms import read_msms
from triangulation.xyzrn import output_pdb_as_xyzrn


def _deduplicate_xyzrn_lines(lines):
    """Remove coincident atom spheres that make legacy MSMS abort."""
    unique = {}
    order = []
    for line in lines:
        fields = line.split()
        key = tuple(round(float(value), 4) for value in fields[:3])
        if key not in unique:
            unique[key] = line
            order.append(key)
    return [unique[key] for key in order], len(lines) - len(order)


def _find_molecular_components(xyzrn_path):
    with open(xyzrn_path) as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise RuntimeError("xyzrn input contains no atoms")
    lines, duplicate_count = _deduplicate_xyzrn_lines(lines)

    spheres = np.asarray(
        [[float(value) for value in line.split()[:4]] for line in lines],
        dtype=np.float64,
    )
    coordinates = spheres[:, :3]
    radii = spheres[:, 3]
    parent = np.arange(len(lines), dtype=np.int64)

    def find(index):
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(source, target):
        source_root = find(source)
        target_root = find(target)
        if source_root != target_root:
            parent[target_root] = source_root

    # Covalent bonds are much shorter than nonbonded contacts. The scaled
    # cutoff also retains disulfides and hydrogens added by Reduce.
    neighborhoods = KDTree(coordinates).query_radius(coordinates, r=2.35)
    for source, targets in enumerate(neighborhoods):
        for target_raw in targets:
            target = int(target_raw)
            if target <= source:
                continue
            distance = np.linalg.norm(coordinates[source] - coordinates[target])
            cutoff = 0.62 * (radii[source] + radii[target]) + 0.05
            if distance <= cutoff:
                union(source, target)

    components = {}
    for atom_id in range(len(lines)):
        components.setdefault(find(atom_id), []).append(atom_id)
    atom_groups = sorted(
        components.values(), key=lambda ids: (-len(ids), ids[0])
    )
    return lines, atom_groups, duplicate_count


def _read_area(path):
    areas = {}
    if not os.path.exists(path):
        return areas
    with open(path) as handle:
        next(handle, None)
        for line in handle:
            fields = line.split()
            if len(fields) >= 4:
                areas[fields[3]] = fields[1]
    return areas


def _run_component(input_path, output_base, density, probe):
    args = [
        msms_bin,
        "-density", str(density),
        "-hdensity", str(density),
        "-probe", str(probe),
        "-if", input_path,
        "-of", output_base,
        "-af", output_base,
    ]
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = stdout.decode("utf-8", errors="replace").strip()[-2000:]
        raise RuntimeError(
            "MSMS failed with code {}: {}".format(process.returncode, detail)
        )
    part = read_msms(output_base)
    if len(part[0]) == 0 or len(part[1]) == 0:
        raise RuntimeError("MSMS produced an empty molecular surface")
    return part, _read_area(output_base + ".area")


def _merge_parts(parts):
    vertices = []
    faces = []
    normals = []
    names = []
    offset = 0
    for part_vertices, part_faces, part_normals, part_names in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        normals.append(part_normals)
        names.extend(part_names)
        offset += len(part_vertices)
    return (
        np.concatenate(vertices, axis=0),
        np.concatenate(faces, axis=0),
        np.concatenate(normals, axis=0),
        names,
    )


def _remove_outputs(file_base):
    for path in glob.glob(file_base + "*"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _write_audit(lines):
    marker = os.environ.get("SGNET_MSMS_FALLBACK_MARKER")
    if not marker or not lines:
        return
    with open(marker, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def computeMSMS(pdb_file, protonate=True):
    if not protonate:
        raise ValueError("pdb2xyzrn is deprecated; protonate must be True")
    randnum = random.randint(1, 10000000)
    file_base = masif_opts["tmp_dir"] + "/msms_" + str(randnum)
    out_xyzrn = file_base + ".xyzrn"
    output_pdb_as_xyzrn(pdb_file, out_xyzrn)

    audit = ["pdb_file={}".format(pdb_file),
             "probe_density_schedule=1.5,1.4,1.6,1.2 x 3.0,2.0,1.0"]
    try:
        lines, atom_groups, duplicate_count = _find_molecular_components(out_xyzrn)
        if duplicate_count:
            audit.append("coincident_atom_spheres_removed={}".format(duplicate_count))
        parts = []
        areas = {}
        skipped = []
        retried = []
        for component_id, atom_ids in enumerate(atom_groups):
            if len(atom_ids) < 3:
                skipped.append((component_id, len(atom_ids), "fewer than three atoms"))
                continue
            component_input = "{}_mol_{:04d}.xyzrn".format(file_base, component_id)
            with open(component_input, "w") as handle:
                handle.writelines(lines[atom_id] for atom_id in atom_ids)
            part = None
            component_areas = None
            errors = []
            for probe in (1.5, 1.4, 1.6, 1.2):
                for density in (3.0, 2.0, 1.0):
                    component_base = "{}_mol_{:04d}_p{}_d{}".format(
                        file_base, component_id, str(probe).replace(".", "p"),
                        str(density).replace(".", "p"))
                    try:
                        part, component_areas = _run_component(
                            component_input, component_base, density, probe)
                        if probe != 1.5 or density != 3.0:
                            retried.append(
                                (component_id, len(atom_ids), probe, density))
                        break
                    except Exception as exc:
                        errors.append("probe {} density {}: {}".format(
                            probe, density, exc))
                if part is not None:
                    break
            if part is None:
                detail = " | ".join(errors)
                if len(atom_ids) <= 64:
                    skipped.append((component_id, len(atom_ids), detail))
                    continue
                raise RuntimeError(
                    "component {} ({} atoms) failed all density retries: {}".format(
                        component_id, len(atom_ids), detail))
            parts.append(part)
            areas.update(component_areas)
        if duplicate_count or skipped or retried:
            audit.append("density_retries={}".format(retried))
            audit.append("skipped_components={}".format(skipped))
            _write_audit(audit)
            print("MSMS robust fallback: {}".format("; ".join(audit[2:])), flush=True)
        if not parts:
            raise RuntimeError("MSMS produced no valid molecular components")
        vertices, faces, normals, names = _merge_parts(parts)
        return vertices, faces, normals, names, areas
    finally:
        _remove_outputs(file_base)
