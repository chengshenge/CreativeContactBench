"""Conservative rigid-contact rasterization; no calibrated skin-pressure claim.

Each solver contact belongs to exactly one face per contacted finger. Its full
force on that finger is decomposed in the face's fixed orthonormal basis, then
deposited by bilinear cloud-in-cell (CIC). Cells are non-overlapping 2 mm bins;
force / bin area is a simulated bin-average traction, not soft-pad stress.
Genesis stores contact normal towards A: force_a=-force_b, compressive direction
is +normal on A and -normal on B. Quaternion order is explicitly w,x,y,z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

VERSION = "solver_grid_traction_v1"


def rotation_wxyz(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Quaternion must be finite wxyz[4]")
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"Quaternion is not unit length: {norm}")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def surface_specs(finger_name, cell_size_m=.002):
    if finger_name not in ("left_finger", "right_finger"):
        raise ValueError("Finger name must be left_finger or right_finger")
    side = 1 if finger_name == "left_finger" else -1
    specs = []
    for face, origin, normal, tangent_v, extent in (
        ("inner", [0, -side*.00015, 0], [0, side, 0], [0, 0, 1], [-.011, .011, 0, .054]),
        ("distal", [0, 0, .05385], [0, 0, -1], [0, side, 0], [-.011, .011, 0, .028]),
    ):
        nu = (extent[1]-extent[0])/cell_size_m
        nv = (extent[3]-extent[2])/cell_size_m
        if cell_size_m <= 0 or abs(nu-round(nu)) > 1e-8 or abs(nv-round(nv)) > 1e-8:
            raise ValueError("cell_size_m must divide both declared surface extents")
        u_edges = np.linspace(extent[0], extent[1], round(nu)+1)
        v_edges = np.linspace(extent[2], extent[3], round(nv)+1)
        area = np.outer(np.diff(v_edges), np.diff(u_edges))
        specs.append(dict(name=finger_name+"_"+face, finger=finger_name, face=face,
            origin_local_m=np.array(origin, dtype=float), compression_local=np.array(normal, dtype=float),
            tangent_u_local=np.array([1., 0., 0.]), tangent_v_local=np.array(tangent_v, dtype=float),
            u_edges_m=u_edges, v_edges_m=v_edges, cell_area_m2=area,
            normal_force_N=np.zeros_like(area), shear_u_force_N=np.zeros_like(area), shear_v_force_N=np.zeros_like(area)))
    return specs


def _cic_weights(u, v, spec):
    """Boundary bins absorb their own half-cell; never relocate outside points."""
    ue, ve = spec["u_edges_m"], spec["v_edges_m"]
    if not (ue[0]-1e-10 <= u <= ue[-1]+1e-10 and ve[0]-1e-10 <= v <= ve[-1]+1e-10):
        raise ValueError("Point outside declared sensing extent")
    fu = (u-ue[0])/(ue[1]-ue[0])-.5
    fv = (v-ve[0])/(ve[1]-ve[0])-.5
    iu, iv = int(np.floor(fu)), int(np.floor(fv))
    du, dv = fu-iu, fv-iv
    weights = {}
    for offset_u, wu in [(0, 1-du), (1, du)]:
        for offset_v, wv in [(0, 1-dv), (1, dv)]:
            key = (int(np.clip(iv+offset_v, 0, len(ve)-2)), int(np.clip(iu+offset_u, 0, len(ue)-2)))
            weights[key] = weights.get(key, 0.) + wu*wv
    if not np.isclose(sum(weights.values()), 1., atol=1e-12):
        raise AssertionError("CIC weights do not sum to one")
    return weights


def project_snapshot(snapshot, *, cell_size_m=.002, require_complete=True,
                     max_surface_distance_m=.002, min_normal_alignment=2**-.5,
                     force_atol_N=1e-5, force_rtol=1e-5):
    """Project snapshot['contacts'] and exactly two snapshot['fingers'].

    contacts: position/normal/force_a/force_b[N,3], link_a/link_b[N], optional
    valid_mask[N]. fingers: name, link_index, position_m[3], quat_wxyz[4],
    optional net_force_world_N[3]. Returned arrays are indexed [v,u].
    Incomplete coverage is retained in the result and rejected when requested;
    use require_complete=False for diagnostic inspection of a failed snapshot.
    """
    raw = snapshot["contacts"]
    contacts = {key: np.asarray(raw[key], dtype=float).reshape(-1, 3)
                for key in ("position", "normal", "force_a", "force_b")}
    n = len(contacts["position"])
    for key in ("link_a", "link_b"):
        links = np.asarray(raw[key], dtype=float).reshape(-1)
        if not np.isfinite(links).all() or not np.all(links == links.astype(int)):
            raise ValueError("Link indices must be finite integers")
        contacts[key] = links.astype(int)
    if any(len(value) != n or not np.isfinite(value).all() for value in contacts.values()):
        raise ValueError("Contact fields have inconsistent lengths or nonfinite values")
    valid = np.asarray(raw.get("valid_mask", np.ones(n, dtype=bool)), dtype=bool).reshape(-1)
    if len(valid) != n:
        raise ValueError("Invalid valid_mask length")
    fingers = snapshot["fingers"]
    if len(fingers) != 2 or {f["name"] for f in fingers} != {"left_finger", "right_finger"}:
        raise ValueError("Exactly one left_finger and one right_finger required")
    if len({int(f["link_index"]) for f in fingers}) != 2:
        raise ValueError("Finger global link indices must be distinct")
    surfaces, assignments, unassigned, finger_qa = {}, [], [], {}
    max_reaction_error = float(np.abs(contacts["force_a"][valid]+contacts["force_b"][valid]).max(initial=0))
    reaction_ok = np.allclose(contacts["force_a"][valid], -contacts["force_b"][valid], atol=force_atol_N, rtol=force_rtol)
    normal_norms = np.linalg.norm(contacts["normal"], axis=1)
    if np.any(np.abs(normal_norms[valid]-1) > 1e-4):
        raise ValueError("Contact normals must be unit vectors")
    negative_contact_normal_count = 0
    for finger in fingers:
        name, idx = finger["name"], int(finger["link_index"])
        pos = np.asarray(finger["position_m"], dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("Invalid finger position")
        rot = rotation_wxyz(finger["quat_wxyz"])
        specs = surface_specs(name, cell_size_m)
        force_input_world = np.zeros(3)
        abs_force_input = 0.
        input_moment_local = np.zeros(3)
        for ci in np.flatnonzero(valid & ((contacts["link_a"] == idx) | (contacts["link_b"] == idx))):
            is_a = contacts["link_a"][ci] == idx
            world_force = contacts["force_a" if is_a else "force_b"][ci]
            local_force = rot.T @ world_force
            local_point = rot.T @ (contacts["position"][ci]-pos)
            contact_normal = rot.T @ (contacts["normal"][ci] * (1 if is_a else -1))
            contact_normal /= np.linalg.norm(contact_normal)
            contact_normal_force = float(local_force @ contact_normal)
            if contact_normal_force < -force_atol_N:
                negative_contact_normal_count += 1
            force_input_world += world_force
            abs_force_input += np.linalg.norm(world_force)
            input_moment_local += np.cross(local_point, local_force)
            candidates = []
            for spec in specs:
                relative = local_point-spec["origin_local_m"]
                alignment = float(contact_normal @ spec["compression_local"])
                distance = abs(float(relative @ spec["compression_local"]))
                u, v = float(relative @ spec["tangent_u_local"]), float(relative @ spec["tangent_v_local"])
                ue, ve = spec["u_edges_m"], spec["v_edges_m"]
                inside = ue[0]-1e-10 <= u <= ue[-1]+1e-10 and ve[0]-1e-10 <= v <= ve[-1]+1e-10
                if alignment >= min_normal_alignment and distance <= max_surface_distance_m and inside:
                    candidates.append((alignment, -distance, spec["name"], spec, u, v))
            if not candidates:
                unassigned.append(dict(contact_index=int(ci), finger=name, reason="outside_or_unsupported_face",
                                       local_position_m=local_point.tolist(), force_world_N=world_force.tolist(),
                                       compression_normal_local=contact_normal.tolist()))
                continue
            # Unique assignment: largest normal alignment, then nearest plane, then stable name.
            _, _, _, spec, u, v = sorted(candidates, key=lambda x: x[:3], reverse=True)[0]
            components = np.array([local_force @ spec[k] for k in ("compression_local", "tangent_u_local", "tangent_v_local")])
            if components[0] < -force_atol_N:
                unassigned.append(dict(contact_index=int(ci), finger=name, reason="tensile_face_projection",
                                       local_position_m=local_point.tolist(), force_world_N=world_force.tolist()))
                continue
            components[0] = max(0., components[0])  # Only sub-tolerance roundoff can be removed.
            weights = _cic_weights(u, v, spec)
            for (iv, iu), weight in weights.items():
                for key, component in zip(("normal_force_N", "shear_u_force_N", "shear_v_force_N"), components):
                    spec[key][iv, iu] += component*weight
            assignments.append(dict(contact_index=int(ci), finger=name, surface=spec["name"],
                local_position_m=local_point.tolist(), uv_m=[u, v], full_force_local_N=local_force.tolist(),
                solver_contact_normal_force_N=contact_normal_force, face_components_N=components.tolist(),
                bins=[dict(v=iv, u=iu, weight=float(w)) for (iv, iu), w in sorted(weights.items())]))
        reconstructed_local, reconstructed_moment = np.zeros(3), np.zeros(3)
        for spec in specs:
            spec["pressure_kPa"] = spec["normal_force_N"]/spec["cell_area_m2"]/1000
            spec["shear_u_kPa"] = spec["shear_u_force_N"]/spec["cell_area_m2"]/1000
            spec["shear_v_kPa"] = spec["shear_v_force_N"]/spec["cell_area_m2"]/1000
            force = sum(spec[k][..., None]*spec[b] for k, b in (
                ("normal_force_N", "compression_local"), ("shear_u_force_N", "tangent_u_local"), ("shear_v_force_N", "tangent_v_local")))
            reconstructed_local += force.sum(axis=(0, 1))
            uu, vv = np.meshgrid((spec["u_edges_m"][:-1]+spec["u_edges_m"][1:])/2,
                                 (spec["v_edges_m"][:-1]+spec["v_edges_m"][1:])/2)
            centers = spec["origin_local_m"]+uu[..., None]*spec["tangent_u_local"]+vv[..., None]*spec["tangent_v_local"]
            reconstructed_moment += np.cross(centers, force).sum(axis=(0, 1))
            spec["finger_position_world_m"] = pos.copy()
            spec["finger_rotation_local_to_world"] = rot.copy()
            surfaces[spec["name"]] = spec
        reconstructed_world = rot @ reconstructed_local
        force_ok = bool(np.allclose(reconstructed_world, force_input_world, atol=force_atol_N, rtol=force_rtol))
        sensor_error = None
        sensor_ok = True
        if "net_force_world_N" in finger:
            expected = np.asarray(finger["net_force_world_N"], dtype=float)
            if expected.shape != (3,) or not np.isfinite(expected).all():
                raise ValueError("Invalid net_force_world_N")
            sensor_error = float(np.max(np.abs(force_input_world-expected)))
            sensor_ok = bool(np.allclose(force_input_world, expected, atol=force_atol_N, rtol=force_rtol))
        # Point projection to the plane and clamping the edge half-cell can move
        # the moment arm. We report this explicitly; force conservation is exact.
        moment_error = float(np.linalg.norm(reconstructed_moment-input_moment_local))
        moment_bound = (max_surface_distance_m+cell_size_m/2**.5)*abs_force_input + force_atol_N
        finger_qa[name] = dict(input_force_world_N=force_input_world.tolist(),
            reconstructed_force_world_N=reconstructed_world.tolist(), max_force_error_N=float(np.max(np.abs(reconstructed_world-force_input_world))),
            force_conserved=force_ok, sensor_agreement=sensor_ok, sensor_max_error_N=sensor_error,
            point_moment_local_Nm=input_moment_local.tolist(), grid_moment_local_Nm=reconstructed_moment.tolist(),
            moment_discretization_error_Nm=moment_error, moment_discretization_bound_Nm=moment_bound,
            moment_within_discretization_bound=bool(moment_error <= moment_bound))
    finger_indices = {int(f["link_index"]) for f in fingers}
    unrelated = [int(i) for i in np.flatnonzero(valid) if contacts["link_a"][i] not in finger_indices and contacts["link_b"][i] not in finger_indices]
    qa = dict(complete_fingertip_coverage=not unassigned, no_nonfinger_contacts=not unrelated,
        action_reaction_ok=bool(reaction_ok), max_action_reaction_error_N=max_reaction_error,
        compressive_contact_normals=negative_contact_normal_count == 0,
        finite_maps=all(np.isfinite(s[k]).all() for s in surfaces.values() for k in ("pressure_kPa", "shear_u_kPa", "shear_v_kPa")),
        nonnegative_pressure=all(np.all(s["pressure_kPa"] >= 0) for s in surfaces.values()),
        force_conserved=all(f["force_conserved"] for f in finger_qa.values()),
        sensor_agreement=all(f["sensor_agreement"] for f in finger_qa.values()),
        moment_within_discretization_bound=all(f["moment_within_discretization_bound"] for f in finger_qa.values()),
        valid_contact_count=int(valid.sum()), assigned_finger_contact_count=len(assignments),
        unassigned_finger_contact_count=len(unassigned), nonfinger_contact_indices=unrelated, fingers=finger_qa)
    required = ("complete_fingertip_coverage", "no_nonfinger_contacts", "action_reaction_ok", "compressive_contact_normals",
                "finite_maps", "nonnegative_pressure", "force_conserved", "sensor_agreement", "moment_within_discretization_bound")
    qa["passed"] = all(qa[k] for k in required)
    result = dict(version=VERSION, units="kPa (simulated grid-average traction)",
        physical_interpretation="Rigid solver force divided by declared bin area; uncalibrated spatial discretization, not deformable skin stress.",
        rasterization="bilinear CIC within unique face; boundary half-cell mass stays in boundary bin; no cross-face smoothing",
        contact_normal_convention="Genesis normal points towards A; compression is +normal on A, -normal on B",
        pressure_definition="positive inward pad-normal component of full force on finger / bin area",
        cell_size_m=cell_size_m, max_surface_distance_m=max_surface_distance_m,
        min_normal_alignment=min_normal_alignment, surfaces=surfaces, assignments=assignments,
        unassigned=unassigned, qa=qa)
    if require_complete and not qa["passed"]:
        failed = [k for k in required if not qa[k]]
        raise ValueError("Projection QA failed: "+", ".join(failed)+"; rerun require_complete=False to inspect retained details")
    return result


def flatten_arrays(result):
    """Return non-object numpy arrays for np.savez_compressed(..., **arrays)."""
    return {name+"__"+key: value for name, spec in result["surfaces"].items()
            for key, value in spec.items() if isinstance(value, np.ndarray)}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    result = project_snapshot(json.loads(args.snapshot.read_text()), require_complete=False)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output_prefix)+".npz", **flatten_arrays(result))
    Path(str(args.output_prefix)+".json").write_text(json.dumps(jsonable(result), indent=2)+"\n")
    if not result["qa"]["passed"]:
        raise SystemExit("Projection written with FAILED QA; inspect JSON")


if __name__ == "__main__":
    main()
