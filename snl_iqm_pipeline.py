from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import stim
from qiskit import transpile

import iqm_qec_pipeline as base


# =============================================================================
# 0. SNAKES-AND-LADDERS IMPORTS AND IQM EMERALD GEOMETRY
# =============================================================================
#
# The Amazon Snakes-and-Ladders package provides the defect-adapted stabilizer
# construction. The IQM Emerald maps below are used for real hardware labels and
# for Emerald-inspired plotting. For synthetic distances larger than the real
# chip, the plot extends this same tilted lattice pattern.

SNL_REPO_URL = "https://github.com/amazon-science/snakes_and_ladders_adapting_the_surface_code_to_defects.git"
SNL_LOCAL_DIR = Path(__file__).parent / "snakes_and_ladders_adapting_the_surface_code_to_defects"


def _load_snl_package():
    candidates = [
        os.environ.get("SNL_REPO"),
        str(SNL_LOCAL_DIR),
        "/private/tmp/snakes_and_ladders_adapting_the_surface_code_to_defects",
    ]
    for candidate in candidates:
        if candidate and Path(candidate, "defects_module").exists():
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return

    if not SNL_LOCAL_DIR.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", SNL_REPO_URL, str(SNL_LOCAL_DIR)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            raise ImportError(
                "Could not find or download the Amazon Science Snakes-and-Ladders repository. "
                "Install git/network access, or clone it manually next to this notebook with:\n"
                f"git clone {SNL_REPO_URL}\n"
                "Alternatively set SNL_REPO to an existing checkout."
            ) from exc

    if SNL_LOCAL_DIR.exists() and Path(SNL_LOCAL_DIR, "defects_module").exists():
        sys.path.insert(0, str(SNL_LOCAL_DIR))
        return

    raise ImportError(
        "Snakes-and-Ladders checkout exists but does not contain defects_module. "
        f"Expected it at {SNL_LOCAL_DIR}."
    )


_load_snl_package()

from defects_module.base import PauliT, Pos, SuperStabilizer  # noqa: E402
from defects_module.code_library import RotatedSurfaceCode  # noqa: E402
from defects_module.defects import DefectiveSurfaceCode, Heuristics  # noqa: E402
from defects_module.utils import memory_program, standard_noise  # noqa: E402

CAL = base.CAL
COORD_TO_QB = base.COORD_TO_QB
IQM_EMERALD_POS = base.IQM_EMERALD_POS
QB_TO_COORD = base.QB_TO_COORD
read_calibration_metrics = base.read_calibration_metrics


def pos_xy(pos: Pos) -> tuple[float, float]:
    """IQM-Emerald-like plotting coordinate for a SnL lattice position."""
    qb = COORD_TO_QB.get(pos)
    if qb is not None:
        x, y = IQM_EMERALD_POS[qb]
        return x, -y
    return (float(pos.x) + (0.25 if pos.y % 2 else 0.0), -float(pos.y))


def pos_label(pos: Pos) -> str:
    qb = COORD_TO_QB.get(pos)
    return f"QB{qb}" if qb is not None else f"({pos.x},{pos.y})"


# =============================================================================
# 1. CALIBRATION-AVERAGE NOISE MODEL
# =============================================================================
#
# The synthetic protocol samples the same kind of Stim circuit that the hardware
# path submits, but with average IQM calibration-derived reset, measurement, and
# Clifford noise. This intentionally keeps the simulation compact and comparable
# across arbitrary larger synthetic lattices.


def calibration_average_noise(cal: str = CAL) -> dict[str, float]:
    oneq, meas, cz = [], [], []
    for obs in json.loads(Path(cal).read_text())["observations"]:
        if obs.get("invalid"):
            continue
        field = obs["dut_field"]
        value = float(obs["value"])
        qubits = re.findall(r"QB(\d+)", field)
        if len(qubits) == 1:
            if ("metrics.rb.prx" in field or "metrics.rb.clifford.xy_sx" in field) and ".fidelity" in field:
                oneq.append(value)
            if "metrics.ssro.measure" in field and ("error_0_to_1" in field or "error_1_to_0" in field):
                meas.append(value)
        elif len(qubits) >= 2 and ".cz." in field and ".fidelity" in field:
            cz.append(value)

    mean = lambda xs, default: float(np.mean(xs)) if xs else default
    p1 = 1 - mean(oneq, 0.999)
    p2 = 1 - mean(cz, 0.995)
    pm = mean(meas, 0.01)
    return {"idle": max(p1, 1e-6), "meas": max(pm, 1e-6), "reset": max(pm, 1e-6), "gate": max(p1, p2, 1e-6)}


def inject_average_noise(c: stim.Circuit, noise: dict[str, float]) -> stim.Circuit:
    """Return a copy of `c` with calibration-average noise inserted."""
    out = stim.Circuit()
    for ins in c.flattened():
        name, targets = ins.name, ins.targets_copy()
        if name in ("M", "MZ", "MR"):
            for t in targets:
                if t.is_qubit_target:
                    out.append("X_ERROR", [t.value], noise["meas"])
            out.append(ins)
            if name == "MR":
                for t in targets:
                    if t.is_qubit_target:
                        out.append("X_ERROR", [t.value], noise["reset"])
            continue

        out.append(ins)
        if name in ("R", "RZ"):
            for t in targets:
                if t.is_qubit_target:
                    out.append("X_ERROR", [t.value], noise["reset"])
        elif name in ("H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Z", "SQRT_Z_DAG"):
            for t in targets:
                if t.is_qubit_target:
                    out.append("DEPOLARIZE1", [t.value], noise["gate"])
        elif name in ("CX", "CNOT", "CZ"):
            for i in range(0, len(targets), 2):
                out.append("DEPOLARIZE2", [targets[i].value, targets[i + 1].value], noise["gate"])
    return out


# =============================================================================
# 2. DEFECT SELECTION AND SNAKES-AND-LADDERS PATCH CONSTRUCTION
# =============================================================================
#
# Hardware mode can first read the calibration/backend and mark unavailable
# qubits/couplers as baseline defects. Explicit and random defects are then
# added on top. The same final defect set is used for all rounds in a sweep,
# because the sweep should describe one defective device.


def parse_iqm_qubit(q) -> int:
    if isinstance(q, str):
        m = re.search(r"QB(\d+)", q)
        if not m:
            raise ValueError(f"Could not parse IQM qubit label {q!r}")
        return int(m.group(1))
    return int(q)


def parse_iqm_coupler(edge) -> tuple[int, int]:
    if isinstance(edge, str):
        qs = list(map(int, re.findall(r"QB(\d+)", edge)))
        if len(qs) != 2:
            raise ValueError(f"Could not parse IQM coupler label {edge!r}")
        return tuple(sorted(qs))
    a, b = edge
    return tuple(sorted((parse_iqm_qubit(a), parse_iqm_qubit(b))))


def is_boundary_pos(patch: RotatedSurfaceCode, pos: Pos) -> bool:
    xmin, xmax, ymin, ymax = patch.extent
    return pos.x in (xmin + 1, xmax - 1, xmin, xmax) or pos.y in (ymin + 1, ymax - 1, ymin, ymax)


def stabilizer_links(patch: RotatedSurfaceCode) -> list[tuple[Pos, Pos]]:
    links = set()
    for stabilizer in patch.stabilizers:
        for ancilla in stabilizer.ancilla:
            for data in stabilizer.data_qubits:
                links.add((ancilla, data))
    return sorted(links)


def iqm_couplers_from_backend(backend) -> set[tuple[int, int]]:
    """Return currently native backend couplers as IQM labels, e.g. (45, 46)."""
    couplers = set()
    for a, b in backend.coupling_map.get_edges():
        qa = int(re.search(r"QB(\d+)", backend.index_to_qubit_name(a)).group(1))
        qb = int(re.search(r"QB(\d+)", backend.index_to_qubit_name(b)).group(1))
        couplers.add(tuple(sorted((qa, qb))))
    return couplers


def calibration_defects_for_patch(
    patch: RotatedSurfaceCode,
    cal: str = CAL,
    backend_couplers: set[tuple[int, int]] | None = None,
    min_cz_fidelity: float | None = None,
    min_1q_fidelity: float | None = None,
    max_measurement_error: float | None = None,
    coupler_defect_mode: str = "unavailable",
) -> dict:
    """Convert IQM calibration/backend availability into SnL qubit/link defects.

    By default this only treats genuinely unavailable hardware as a defect:
    calibration-invalid qubits/couplers or couplers absent from the backend.
    Low fidelity is kept as usable hardware and is handled by layout scoring and
    the decoder noise model. Set threshold arguments and
    coupler_defect_mode="threshold" only when you intentionally want to remove
    low-quality components.
    """
    if coupler_defect_mode not in {"unavailable", "threshold"}:
        raise ValueError("coupler_defect_mode must be 'unavailable' or 'threshold'")

    oneq, meas, cz, invalid_qubits, invalid_couplers = read_calibration_metrics(cal)

    def bad_qubit(pos: Pos) -> bool:
        qb = COORD_TO_QB.get(pos)
        return (
            qb is None
            or qb in invalid_qubits
            or (min_1q_fidelity is not None and oneq.get(qb, 1.0) < min_1q_fidelity)
            or (max_measurement_error is not None and meas.get(qb, 0.0) > max_measurement_error)
        )

    data_defects = {p for p in patch.data_qubits if bad_qubit(p)}
    ancilla_defects = {p for p in patch.ancilla_qubits if bad_qubit(p)}
    link_defects = set()
    missing_backend_couplers = set()

    for ancilla, data in stabilizer_links(patch):
        qa, qd = COORD_TO_QB.get(ancilla), COORD_TO_QB.get(data)
        if qa is None or qd is None:
            link_defects.add((ancilla, data))
            continue
        edge = tuple(sorted((qa, qd)))
        unavailable = edge in invalid_couplers
        if backend_couplers is not None and edge not in backend_couplers:
            unavailable = True
            missing_backend_couplers.add(edge)
        below_threshold = min_cz_fidelity is not None and cz.get(edge, 1.0) < min_cz_fidelity
        if unavailable or (coupler_defect_mode == "threshold" and below_threshold):
            link_defects.add((ancilla, data))

    return {
        "data_defects": data_defects,
        "ancilla_defects": ancilla_defects,
        "link_defects": link_defects,
        "calibration_info": {
            "invalid_qubits": sorted(invalid_qubits),
            "invalid_couplers": sorted(invalid_couplers),
            "missing_backend_couplers": sorted(missing_backend_couplers),
            "thresholds": {
                "min_cz_fidelity": min_cz_fidelity,
                "min_1q_fidelity": min_1q_fidelity,
                "max_measurement_error": max_measurement_error,
                "coupler_defect_mode": coupler_defect_mode,
            },
        },
    }


def choose_defects(
    clean_patch: RotatedSurfaceCode,
    num_qubit_defects: int = 0,
    num_coupler_defects: int = 0,
    defect_qubits=None,
    defect_couplers=None,
    seed: int | None = None,
    base_data_defects=None,
    base_ancilla_defects=None,
    base_link_defects=None,
) -> dict:
    rng = np.random.default_rng(seed)
    explicit_qbs = [parse_iqm_qubit(q) for q in (defect_qubits or [])]
    explicit_edges = [parse_iqm_coupler(e) for e in (defect_couplers or [])]

    chosen_qubits = set(base_data_defects or set()) | set(base_ancilla_defects or set())
    link_defects = set(base_link_defects or set())
    for qb in explicit_qbs:
        if qb not in QB_TO_COORD:
            raise ValueError(f"QB{qb} is not in the real IQM Emerald map.")
        pos = QB_TO_COORD[qb]
        if pos not in set(clean_patch.data_qubits) | set(clean_patch.ancilla_qubits):
            raise ValueError(f"QB{qb} is outside this distance={clean_patch.distance}, sw_offset={clean_patch.sw_offset} patch.")
        chosen_qubits.add(pos)

    random_qubit_pool = [
        p for p in (set(clean_patch.data_qubits) | set(clean_patch.ancilla_qubits))
        if p not in chosen_qubits and not is_boundary_pos(clean_patch, p)
    ]
    if num_qubit_defects > len(random_qubit_pool):
        raise ValueError(f"Requested {num_qubit_defects} random qubit defects, but only {len(random_qubit_pool)} interior sites are available.")
    if num_qubit_defects:
        inds = rng.choice(len(random_qubit_pool), size=num_qubit_defects, replace=False)
        chosen_qubits |= {random_qubit_pool[int(i)] for i in inds}

    link_candidates = [
        link for link in stabilizer_links(clean_patch)
        if link[0] not in chosen_qubits and link[1] not in chosen_qubits
        and not is_boundary_pos(clean_patch, link[0]) and not is_boundary_pos(clean_patch, link[1])
    ]
    for qa, qb in explicit_edges:
        if qa not in QB_TO_COORD or qb not in QB_TO_COORD:
            raise ValueError(f"Coupler QB{qa}-QB{qb} is not in the real IQM Emerald map.")
        link = (QB_TO_COORD[qa], QB_TO_COORD[qb])
        rev = (link[1], link[0])
        if link in stabilizer_links(clean_patch):
            link_defects.add(link)
        elif rev in stabilizer_links(clean_patch):
            link_defects.add(rev)
        else:
            raise ValueError(f"Coupler QB{qa}-QB{qb} is not an ancilla-data stabilizer link in this patch.")

    remaining_links = [link for link in link_candidates if link not in link_defects]
    if num_coupler_defects > len(remaining_links):
        raise ValueError(f"Requested {num_coupler_defects} random coupler defects, but only {len(remaining_links)} interior links are available.")
    if num_coupler_defects:
        inds = rng.choice(len(remaining_links), size=num_coupler_defects, replace=False)
        link_defects |= {remaining_links[int(i)] for i in inds}

    data_defects = set(clean_patch.data_qubits) & chosen_qubits
    ancilla_defects = set(clean_patch.ancilla_qubits) & chosen_qubits
    return {
        "data_defects": data_defects,
        "ancilla_defects": ancilla_defects,
        "link_defects": link_defects,
        "defect_qubits": sorted(COORD_TO_QB[p] for p in chosen_qubits if p in COORD_TO_QB),
        "defect_couplers": sorted(tuple(sorted((COORD_TO_QB[a], COORD_TO_QB[b]))) for a, b in link_defects if a in COORD_TO_QB and b in COORD_TO_QB),
    }


def build_defective_patch(
    distance: int,
    num_qubit_defects: int = 0,
    num_coupler_defects: int = 0,
    defect_qubits=None,
    defect_couplers=None,
    seed: int | None = None,
    sw_offset: tuple[int, int] = (1, 1),
    use_calibration_defects: bool = False,
    cal: str = CAL,
    backend_couplers: set[tuple[int, int]] | None = None,
    min_cz_fidelity: float | None = None,
    min_1q_fidelity: float | None = None,
    max_measurement_error: float | None = None,
    coupler_defect_mode: str = "unavailable",
):
    clean_patch = RotatedSurfaceCode(distance, sw_offset=Pos(*sw_offset), vertical_logical=PauliT.Z)
    calibration_defects = {"data_defects": set(), "ancilla_defects": set(), "link_defects": set(), "calibration_info": {}}
    if use_calibration_defects:
        calibration_defects = calibration_defects_for_patch(
            clean_patch,
            cal=cal,
            backend_couplers=backend_couplers,
            min_cz_fidelity=min_cz_fidelity,
            min_1q_fidelity=min_1q_fidelity,
            max_measurement_error=max_measurement_error,
            coupler_defect_mode=coupler_defect_mode,
        )
    defects = choose_defects(
        clean_patch,
        num_qubit_defects,
        num_coupler_defects,
        defect_qubits,
        defect_couplers,
        seed,
        base_data_defects=calibration_defects["data_defects"],
        base_ancilla_defects=calibration_defects["ancilla_defects"],
        base_link_defects=calibration_defects["link_defects"],
    )
    _, damaged_patches = DefectiveSurfaceCode.make_defective_patches(
        clean_patch,
        defects["ancilla_defects"],
        defects["data_defects"],
        defects["link_defects"],
        repurpose_ancillas=True,
        add_padding=True,
        first_super_type=PauliT.X,
        use_heuristics=Heuristics(n_sol_max_per_cluster=10, n_sol_max=50, n_skip=5),
    )
    if not damaged_patches:
        raise ValueError("Snakes-and-Ladders did not find a valid patch for these defects.")
    damaged_patch = max(damaged_patches, key=lambda p: min(p.effective_distance))
    return {
        "clean_patch": clean_patch,
        "damaged_patch": damaged_patch,
        **defects,
        "calibration_info": calibration_defects["calibration_info"],
        "use_calibration_defects": use_calibration_defects,
        "sw_offset": sw_offset,
        "seed": seed,
    }


def patch_iqm_edges(patch) -> set[tuple[int, int]]:
    edges = set()
    for ancilla, data in stabilizer_links(patch):
        qa, qd = COORD_TO_QB.get(ancilla), COORD_TO_QB.get(data)
        if qa is not None and qd is not None:
            edges.add(tuple(sorted((qa, qd))))
    return edges


def patch_fidelity_score(patch_data: dict, cal: str = CAL, backend_couplers: set[tuple[int, int]] | None = None) -> float:
    """Higher is better; prioritize native, high-fidelity couplers and qubits."""
    oneq, meas, cz, invalid_qubits, invalid_couplers = read_calibration_metrics(cal)
    patch = patch_data["damaged_patch"]
    positions = set(patch.data_qubits) | set(patch.ancilla_qubits)
    if any(p not in COORD_TO_QB for p in positions):
        return -1e9

    qubit_terms = []
    for p in positions:
        qb = COORD_TO_QB[p]
        if qb in invalid_qubits:
            return -1e9
        qubit_terms.append(np.log(max(oneq.get(qb, 0.999), 1e-6)))
        qubit_terms.append(np.log(max(1 - meas.get(qb, 0.01), 1e-6)))

    edge_terms = []
    for edge in patch_iqm_edges(patch):
        if edge in invalid_couplers or edge not in cz:
            edge_terms.append(np.log(0.95))
            continue
        if backend_couplers is not None and edge not in backend_couplers:
            edge_terms.append(np.log(0.90))
            continue
        edge_terms.append(np.log(max(cz.get(edge, 0.995), 1e-6)))
    return float(np.mean(qubit_terms + edge_terms)) if qubit_terms or edge_terms else -1e9


def real_emerald_offsets_for_distance(distance: int) -> list[tuple[int, int]]:
    xs = [p.x for p in COORD_TO_QB]
    ys = [p.y for p in COORD_TO_QB]
    candidates = []
    for x in range(min(xs) - 1, max(xs) + 2):
        for y in range(min(ys) - 1, max(ys) + 2):
            patch = RotatedSurfaceCode(distance, sw_offset=Pos(x, y), vertical_logical=PauliT.Z)
            positions = set(patch.data_qubits) | set(patch.ancilla_qubits)
            if positions and all(p in COORD_TO_QB for p in positions):
                candidates.append((x, y))
    return candidates


def build_optimized_defective_patch(
    distance: int,
    num_qubit_defects: int = 0,
    num_coupler_defects: int = 0,
    defect_qubits=None,
    defect_couplers=None,
    seed: int | None = None,
    cal: str = CAL,
    backend_couplers: set[tuple[int, int]] | None = None,
    use_calibration_defects: bool = True,
    min_cz_fidelity: float | None = None,
    min_1q_fidelity: float | None = None,
    max_measurement_error: float | None = None,
    coupler_defect_mode: str = "unavailable",
) -> dict:
    """Scan real Emerald placements and keep the valid SnL patch with best score."""
    best = None
    failures = []
    for sw_offset in real_emerald_offsets_for_distance(distance):
        try:
            patch_data = build_defective_patch(
                distance=distance,
                num_qubit_defects=num_qubit_defects,
                num_coupler_defects=num_coupler_defects,
                defect_qubits=defect_qubits,
                defect_couplers=defect_couplers,
                seed=seed,
                sw_offset=sw_offset,
                use_calibration_defects=use_calibration_defects,
                cal=cal,
                backend_couplers=backend_couplers,
                min_cz_fidelity=min_cz_fidelity,
                min_1q_fidelity=min_1q_fidelity,
                max_measurement_error=max_measurement_error,
                coupler_defect_mode=coupler_defect_mode,
            )
            score = patch_fidelity_score(patch_data, cal=cal, backend_couplers=backend_couplers)
            rank = (min(patch_data["damaged_patch"].effective_distance), score)
            if best is None or rank > best[0]:
                best = (rank, patch_data)
        except Exception as exc:
            failures.append((sw_offset, str(exc)))
    if best is None:
        raise ValueError(f"No valid optimized SnL patch found for distance={distance}. First failures: {failures[:5]}")
    patch_data = best[1]
    patch_data["optimize_layout"] = True
    patch_data["layout_score"] = best[0][1]
    patch_data["layout_candidates_checked"] = len(real_emerald_offsets_for_distance(distance))
    return patch_data


def hardware_faulty_region_demo_kwargs(
    distance: int = 3,
    defect_qubits=(),
    defect_couplers=("QB24_QB32",),
    cal: str = CAL,
    use_calibration_defects: bool = False,
) -> dict:
    """Return a fixed real-Emerald SnL patch around the lower-left faulty region.

    This is meant for the hardware demo: it finds a real d=3 patch containing
    the QB24-QB32 coupler, then returns the exact kwargs to pass into
    `run_snl_round_sweep`. The adjacent qubits are kept usable; only the coupler is
    treated as unavailable unless `defect_qubits` is explicitly provided. We
    disable layout optimization afterwards so the experiment stays on this
    chosen physical part of the chip.
    """
    failures = []
    for sw_offset in real_emerald_offsets_for_distance(distance):
        try:
            patch_data = build_defective_patch(
                distance=distance,
                sw_offset=sw_offset,
                defect_qubits=list(defect_qubits),
                defect_couplers=list(defect_couplers),
                use_calibration_defects=use_calibration_defects,
                cal=cal,
            )
            return {
                "distance": distance,
                "sw_offset": sw_offset,
                "defect_qubits": list(defect_qubits),
                "defect_couplers": list(defect_couplers),
                "use_calibration_defects": use_calibration_defects,
                "optimize_layout": False,
                "patch_data": patch_data,
                "summary": {
                    "sw_offset": sw_offset,
                    "defect_qubits": patch_data["defect_qubits"],
                    "defect_couplers": patch_data["defect_couplers"],
                    "effective_distance": patch_data["damaged_patch"].effective_distance,
                },
            }
        except Exception as exc:
            failures.append((sw_offset, str(exc)))
    raise ValueError(f"No real Emerald d={distance} patch found for this faulty region. First failures: {failures[:5]}")


# =============================================================================
# 3. STIM CIRCUIT, QISKIT CONVERSION, HARDWARE/SYNTHETIC EXECUTION
# =============================================================================


def snl_memory_circuit(patch, rounds: int, basis: str = "Z") -> tuple[stim.Circuit, object]:
    pauli = PauliT.Z if basis.upper() == "Z" else PauliT.X
    physical = memory_program(patch, standard_noise(0), rounds=rounds, pauli=pauli)
    return physical.stim_circuit, physical


def circuit_qubits(c: stim.Circuit) -> set[int]:
    """Stim qubit indices that are actually touched by circuit operations."""
    used = set()
    for ins in c.flattened():
        for t in ins.targets_copy():
            if t.is_qubit_target:
                used.add(int(t.value))
    return used


def patch_mapping(patch, c: stim.Circuit | None = None, allow_missing: bool = False) -> dict[int, int]:
    positions = set(patch.data_qubits) | set(patch.ancilla_qubits)
    if c is not None:
        used = circuit_qubits(c)
        positions = {p for p in positions if int(p) in used}
    missing = [p for p in positions if p not in COORD_TO_QB]
    if missing:
        if allow_missing:
            warnings.warn(f"Skipping unmappable non-Emerald SnL positions in hardware mapping: {missing[:8]}")
            positions = {p for p in positions if p in COORD_TO_QB}
        else:
            raise ValueError(f"Hardware mode requires real Emerald positions; missing {missing[:8]}")
    return {int(p): COORD_TO_QB[p] for p in positions}


def stim_to_qiskit_mapped_skip_impossible(c: stim.Circuit, mapping: dict, backend=None):
    """Convert Stim to Qiskit while warning and skipping impossible hardware gates."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if not mapping:
        raise ValueError("No hardware-mappable qubits remain for this circuit.")
    qr, cr = QuantumRegister(max(mapping.values()) + 1, "q"), ClassicalRegister(c.num_measurements, "m")
    qc, meas, k = QuantumCircuit(qr, cr), [], 0
    hw_edges = {tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()} if backend is not None else None
    skipped = []

    def mapped(t):
        return mapping.get(int(t.value))

    for ins in c.flattened():
        name, t = ins.name, ins.targets_copy()
        if name in {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"} or name.endswith("ERROR") or name.startswith(("DEPOLARIZE", "PAULI_CHANNEL")):
            continue
        if name == "TICK":
            qc.barrier()
            continue
        if name == "I":
            continue
        if name in ("R", "RZ"):
            for x in t:
                qx = mapped(x)
                if qx is None:
                    skipped.append((name, (int(x.value),), "unmappable qubit"))
                else:
                    qc.reset(qr[qx])
            continue
        if name == "H":
            for x in t:
                qx = mapped(x)
                if qx is None:
                    skipped.append((name, (int(x.value),), "unmappable qubit"))
                else:
                    qc.h(qr[qx])
            continue
        if name in ("M", "MZ", "MR"):
            for x in t:
                qx = mapped(x)
                if qx is None:
                    skipped.append((name, (int(x.value),), "unmappable measurement left as classical 0"))
                else:
                    qc.measure(qr[qx], cr[k])
                    meas.append((int(x.value), qx, k))
                k += 1
            if name == "MR":
                for x in t:
                    qx = mapped(x)
                    if qx is None:
                        skipped.append(("R", (int(x.value),), "unmappable reset after MR"))
                    else:
                        qc.reset(qr[qx])
            continue
        if name in ("CX", "CNOT", "CZ"):
            for i in range(0, len(t), 2):
                qa, qb = mapped(t[i]), mapped(t[i + 1])
                stim_edge = (int(t[i].value), int(t[i + 1].value))
                if qa is None or qb is None:
                    skipped.append((name, stim_edge, "unmappable endpoint"))
                    continue
                if hw_edges is not None and tuple(sorted((qa, qb))) not in hw_edges:
                    skipped.append((name, stim_edge, "non-native backend coupler"))
                    continue
                if name == "CZ":
                    qc.cz(qr[qa], qr[qb])
                else:
                    qc.cx(qr[qa], qr[qb])
            continue
        raise NotImplementedError(name)

    if k != c.num_measurements:
        raise ValueError(f"Measurement mismatch after conversion: Qiskit has {k}, Stim has {c.num_measurements}.")
    if skipped:
        warnings.warn(f"Skipped {len(skipped)} impossible hardware operation(s). First skipped operations: {skipped[:8]}")
    return qc, meas, skipped


def synthetic_mapping(patch) -> dict[int, int]:
    positions = sorted(set(patch.data_qubits) | set(patch.ancilla_qubits))
    return {int(p): i for i, p in enumerate(positions)}


def run_one_round_count(
    patch_data: dict,
    rounds: int,
    shots: int,
    mode: str,
    basis: str,
    cal: str,
    backend=None,
    token: str | None = None,
    api_url: str | None = None,
    quantum_computer: str = "emerald",
    use_timeslot: bool = False,
):
    patch = patch_data["damaged_patch"]
    c, physical = snl_memory_circuit(patch, rounds, basis)
    noise = calibration_average_noise(cal)
    decoder_c = inject_average_noise(c, noise)

    raw_measurements = None
    job = result = counts = iqm_qc = None
    backend_mapping = None
    skipped_hardware_ops = []
    swap_count = None
    if mode == "synthetic":
        raw_measurements = decoder_c.compile_sampler().sample(shots).astype(bool)
        memory = base.stim_measurements_to_qiskit_memory(raw_measurements)
        qc = None
    elif mode == "hardware":
        if backend is None:
            with base._explicit_token_overrides_env(token):
                provider = base.IQMProvider(
                    api_url or os.environ.get("IQM_API_URL", "https://resonance.iqm.tech/"),
                    quantum_computer=quantum_computer,
                    token=token or os.environ["IQM_TOKEN"],
                )
                backend = provider.get_backend()
        mapping = patch_mapping(patch, c=c, allow_missing=True)
        backend_mapping = base.iqm_label_mapping_to_backend_indices(mapping, backend)
        qc, _, skipped_hardware_ops = stim_to_qiskit_mapped_skip_impossible(c, backend_mapping, backend=backend)
        if not skipped_hardware_ops:
            base.validate_backend_mapping(qc, c, backend_mapping, backend)
        else:
            warnings.warn("Backend validation against the full Stim circuit was skipped because impossible operations were intentionally omitted.")
        iqm_qc = transpile(qc, backend=backend, optimization_level=1, initial_layout=base.identity_initial_layout(qc))
        swap_count = iqm_qc.count_ops().get("swap", 0)
        job = backend.run(iqm_qc, shots=shots, use_timeslot=use_timeslot)
        result = job.result()
        memory, counts = base.result_to_memory(result)
    else:
        raise ValueError("mode must be 'synthetic' or 'hardware'")

    decoded = base.decode(c, decoder_c, memory)
    summary = {
        "mode": mode,
        "job_id": job.job_id() if job is not None else None,
        "distance": patch_data["clean_patch"].distance,
        "rounds": rounds,
        "shots": shots,
        "effective_distance": patch.effective_distance,
        "logical_error_probability": decoded["corrected_ler"],
        "uncorrected_logical_error_probability": decoded["uncorrected_ler"],
        "num_data_defects": len(patch_data["data_defects"]),
        "num_ancilla_defects": len(patch_data["ancilla_defects"]),
        "num_coupler_defects": len(patch_data["link_defects"]),
        "defect_qubits": patch_data["defect_qubits"],
        "defect_couplers": patch_data["defect_couplers"],
        "use_calibration_defects": patch_data.get("use_calibration_defects", False),
        "optimize_layout": bool(patch_data.get("optimize_layout", False)),
        "sw_offset": patch_data["sw_offset"],
        "layout_score": patch_data.get("layout_score"),
        "swap_count": swap_count,
        "skipped_hardware_operation_count": len(skipped_hardware_ops),
    }
    return {
        **patch_data,
        "stim_circuit": c,
        "physical_circuit": physical,
        "decoder_circuit": decoder_c,
        "noise_model": noise,
        "raw_measurements": raw_measurements,
        "memory": memory,
        "counts": counts,
        "decoded": decoded,
        "qiskit_circuit": qc,
        "iqm_circuit": iqm_qc,
        "backend_mapping": backend_mapping,
        "skipped_hardware_ops": skipped_hardware_ops,
        "job": job,
        "result": result,
        "summary": summary,
    }


# =============================================================================
# 4. VISUALISATION
# =============================================================================
#
# The plot shows the two alternating SnL check layers. Filled regions are gauge
# checks measured in that layer; dashed outlines are full superstabilizers.


def _plot_layer(result: dict, layer: str, ax):
    import math
    from matplotlib.patches import Polygon

    patch = result["damaged_patch"]
    data_defects = result["data_defects"]
    ancilla_defects = result["ancilla_defects"]
    active = set(patch.data_qubits) | set(patch.ancilla_qubits)
    visible = active | data_defects | ancilla_defects
    for d in data_defects | ancilla_defects:
        visible.update(Pos.neighbors(d))

    if layer == "first":
        stabilizers = list(getattr(patch, "undamaged_stabilizers", [])) + list(getattr(patch, "first_super", []))
        title = "first SnL layer + undamaged checks"
    else:
        stabilizers = list(getattr(patch, "undamaged_stabilizers", [])) + list(getattr(patch, "second_super", []))
        title = "second SnL layer + undamaged checks"

    def order(points, center):
        return sorted(points, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))

    def gauge_polygon(gauge):
        pts = [pos_xy(q) for q in gauge.data_qubits]
        if len(pts) == 2:
            pts = pts + [pos_xy(gauge.only_ancilla)]
        if len(pts) < 3:
            return []
        return order(pts, pos_xy(gauge.only_ancilla))

    def draw_super_boundary(gauges, color):
        edge_counts = defaultdict(int)
        edge_points = {}

        def key(point):
            return (round(float(point[0]), 6), round(float(point[1]), 6))

        for gauge in gauges:
            poly = gauge_polygon(gauge)
            if len(poly) < 3:
                continue
            for p0, p1 in zip(poly, poly[1:] + poly[:1]):
                edge_key = tuple(sorted((key(p0), key(p1))))
                edge_counts[edge_key] += 1
                edge_points[edge_key] = (p0, p1)

        for edge_key, count in edge_counts.items():
            if count != 1:
                continue
            p0, p1 = edge_points[edge_key]
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color=color,
                lw=2.6,
                linestyle=(0, (6, 4)),
                solid_capstyle="round",
                zorder=5,
            )

    ax.set_aspect("equal")
    ax.axis("off")

    used_links = set()
    for s in stabilizers:
        gauges = s.gauges if isinstance(s, SuperStabilizer) else [s]
        for g in gauges:
            for q in g.data_qubits:
                used_links.add(tuple(sorted((g.only_ancilla, q))))

    all_links = {tuple(sorted(link)) for link in stabilizer_links(result["clean_patch"]) if link[0] in visible and link[1] in visible}
    for a, b in sorted(all_links):
        x0, y0 = pos_xy(a)
        x1, y1 = pos_xy(b)
        is_used = tuple(sorted((a, b))) in used_links
        ax.plot([x0, x1], [y0, y1], color="#16a34a" if is_used else "#9be375", lw=2.2 if is_used else 1.0, alpha=0.8, zorder=0)
        ax.scatter([(x0 + x1) / 2], [(y0 + y1) / 2], marker="D", s=420 if is_used else 300, c="#16a34a" if is_used else "#8dda68", edgecolors="white", zorder=1)

    for s in stabilizers:
        gauges = s.gauges if isinstance(s, SuperStabilizer) else [s]
        color = "#4f7cff" if s.type == PauliT.X else "#ff5a5f"
        edge = "#1d4ed8" if s.type == PauliT.X else "#b91c1c"
        for g in gauges:
            pts = gauge_polygon(g)
            center = pos_xy(g.only_ancilla)
            if len(pts) >= 3:
                ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge, lw=1.8, alpha=0.2, zorder=2))
            x, y = center
            ax.text(x, y, "X" if s.type == PauliT.X else "Z", ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                    bbox={"boxstyle": "circle,pad=0.2", "facecolor": edge, "edgecolor": "white"}, zorder=6)

        if isinstance(s, SuperStabilizer):
            draw_super_boundary(gauges, edge)

    for p in sorted(visible):
        x, y = pos_xy(p)
        if p in data_defects or p in ancilla_defects:
            face, txt = "#4c1d95", "white"
            label = f"{pos_label(p)}\ndefect"
        elif p in active:
            face, txt = "#5fcf55", "white"
            label = pos_label(p)
        else:
            face, txt = "#f8fafc", "#94a3b8"
            label = pos_label(p)
        ax.scatter([x], [y], s=740, c=face, edgecolors="white", lw=2, zorder=4)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color=txt, zorder=7)

    ax.set_title(f"{title}\neffective distance={patch.effective_distance}", fontsize=11.5)


def plot_snl_lattice(result: dict, save_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    _plot_layer(result, "first", axes[0])
    _plot_layer(result, "second", axes[1])
    fig.suptitle("Snakes-and-Ladders stabilizers on IQM-Emerald-style lattice", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


# =============================================================================
# 5. PUBLIC PIPELINE
# =============================================================================


def run_snl_pipeline(
    distance: int,
    rounds: int,
    shots: int,
    mode: str = "synthetic",
    num_qubit_defects: int = 0,
    num_coupler_defects: int = 0,
    defect_qubits=None,
    defect_couplers=None,
    basis: str = "Z",
    cal: str = CAL,
    seed: int | None = None,
    sw_offset: tuple[int, int] = (1, 1),
    token: str | None = None,
    api_url: str | None = None,
    quantum_computer: str = "emerald",
    use_timeslot: bool = False,
    refresh_calibration: bool = True,
    use_calibration_defects: bool | None = None,
    optimize_layout: bool = False,
    min_cz_fidelity: float | None = None,
    min_1q_fidelity: float | None = None,
    max_measurement_error: float | None = None,
    coupler_defect_mode: str = "unavailable",
    show_plot: bool = True,
) -> dict:
    """Build, run, decode, and visualize a Snakes-and-Ladders defective surface code.

    Args:
        distance: Starting rotated-code distance.
        rounds: Surface-code check rounds.
        shots: Number of hardware/synthetic shots.
        mode: "synthetic" or "hardware".
        num_qubit_defects: Random interior unavailable qubits, in addition to explicit defects.
        num_coupler_defects: Random interior unavailable couplers, in addition to explicit couplers.
        defect_qubits: Optional explicit IQM qubits, e.g. [27, "QB34"].
        defect_couplers: Optional explicit IQM couplers, e.g. ["QB45_QB46", (36, 27)].
        use_calibration_defects: If None, defaults to True for hardware and False for synthetic.
        optimize_layout: If True, scan real Emerald placements and choose the best valid one.
    """
    if refresh_calibration:
        cal = base.refresh_calibration_if_needed(cal, api_url, token, quantum_computer)

    backend = None
    backend_couplers = None
    if mode == "hardware":
        with base._explicit_token_overrides_env(token):
            provider = base.IQMProvider(
                api_url or os.environ.get("IQM_API_URL", "https://resonance.iqm.tech/"),
                quantum_computer=quantum_computer,
                token=token or os.environ["IQM_TOKEN"],
            )
            backend = provider.get_backend()
        backend_couplers = iqm_couplers_from_backend(backend)

    if use_calibration_defects is None:
        use_calibration_defects = mode == "hardware"

    builder_kwargs = dict(
        distance=distance,
        num_qubit_defects=num_qubit_defects,
        num_coupler_defects=num_coupler_defects,
        defect_qubits=defect_qubits,
        defect_couplers=defect_couplers,
        seed=seed,
        cal=cal,
        backend_couplers=backend_couplers,
        use_calibration_defects=use_calibration_defects,
        min_cz_fidelity=min_cz_fidelity,
        min_1q_fidelity=min_1q_fidelity,
        max_measurement_error=max_measurement_error,
        coupler_defect_mode=coupler_defect_mode,
    )
    if optimize_layout:
        patch_data = build_optimized_defective_patch(**builder_kwargs)
    else:
        patch_data = build_defective_patch(sw_offset=sw_offset, **builder_kwargs)
    out = run_one_round_count(
        patch_data=patch_data,
        rounds=rounds,
        shots=shots,
        mode=mode,
        basis=basis,
        cal=cal,
        backend=backend,
        token=token,
        api_url=api_url,
        quantum_computer=quantum_computer,
        use_timeslot=use_timeslot,
    )
    out["calibration_path"] = cal
    if show_plot:
        out["figure"] = plot_snl_lattice(out)
    return out


def fit_snl_round_sweep(results: list[dict]) -> dict:
    """Fit per-round logical error rate from a sweep over round counts."""
    rounds = np.array([r["summary"]["rounds"] for r in results], dtype=float)
    corrected = np.array([r["summary"]["logical_error_probability"] for r in results], dtype=float)
    uncorrected = np.array([r["summary"]["uncorrected_logical_error_probability"] for r in results], dtype=float)
    shots = np.array([r["summary"]["shots"] for r in results], dtype=float)

    # The fit model is bounded by 0.5. Folding handles an inverted logical frame
    # without hiding the raw probabilities from the returned data.
    corrected_fit_prob = np.minimum(corrected, 1 - corrected)
    uncorrected_fit_prob = np.minimum(uncorrected, 1 - uncorrected)
    return {
        "rounds": rounds,
        "corrected_prob": corrected,
        "uncorrected_prob": uncorrected,
        "corrected_fit_prob": corrected_fit_prob,
        "uncorrected_fit_prob": uncorrected_fit_prob,
        "corrected_was_folded": bool(np.any(corrected > 0.5)),
        "uncorrected_was_folded": bool(np.any(uncorrected > 0.5)),
        "corrected_fit": base.fit_logical_error_rate(rounds, corrected_fit_prob, shots=shots),
        "uncorrected_fit": base.fit_logical_error_rate(rounds, uncorrected_fit_prob, shots=shots),
    }


def plot_snl_logical_error_rate(fit: dict, save_path: str | None = None):
    """Plot logical error probability vs rounds on a readable linear scale."""
    import matplotlib.pyplot as plt

    rounds = fit["rounds"]
    xs = np.linspace(rounds.min(), rounds.max(), 300)
    eps_c = fit["corrected_fit"]["epsilon_l"]
    eps_u = fit["uncorrected_fit"]["epsilon_l"]
    y_c = fit["corrected_fit_prob"]
    y_u = fit["uncorrected_fit_prob"]
    fit_y_c = base.logical_error_probability(xs, eps_c)
    fit_y_u = base.logical_error_probability(xs, eps_u)

    def set_relevant_ylim(ax, values, log_scale: bool):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if log_scale:
            vals = vals[vals > 0]
            if len(vals) == 0:
                ax.set_ylim(1e-5, 0.7)
                return
            lo, hi = float(np.min(vals)), float(np.max(vals))
            if np.isclose(lo, hi):
                lo, hi = lo / 1.4, hi * 1.4
            else:
                factor = max(1.15, (hi / lo) ** 0.12)
                lo, hi = lo / factor, hi * factor
            ax.set_ylim(max(lo, 1e-8), min(max(hi, 2e-8), 1.0))
            return

        if len(vals) == 0:
            ax.set_ylim(0, 0.7)
            return
        lo, hi = float(np.min(vals)), float(np.max(vals))
        span = hi - lo
        margin = max(0.015, 0.25 * span)
        ax.set_ylim(max(0.0, lo - margin), min(1.0, hi + margin))

    plotted_values = [*y_c, *y_u, *fit_y_c, *fit_y_u]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(rounds, y_u, "o", color="#94a3b8", label="uncorrected probability")
    ax.plot(xs, fit_y_u, "--", color="#64748b", label=f"uncorrected fit, eps_L={eps_u:.4g}")
    ax.plot(rounds, y_c, "o", color="#2563eb", label="MWPM corrected probability")
    ax.plot(xs, fit_y_c, "-", color="#2563eb", label=f"corrected fit, eps_L={eps_c:.4g}")
    ax.axhline(0.5, color="black", lw=1, alpha=0.25)
    ax.set_xlabel("surface-code rounds")
    ax.set_ylabel("logical error probability")
    ax.grid(True, alpha=0.3)
    set_relevant_ylim(ax, plotted_values, log_scale=False)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Snakes-and-Ladders LER extraction\nMWPM eps_L={eps_c:.4g}, uncorrected eps_L={eps_u:.4g}", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def run_snl_round_sweep(
    distance: int,
    round_values=(1, 2, 3, 5, 8, 10),
    shots: int = 5_000,
    mode: str = "synthetic",
    num_qubit_defects: int = 0,
    num_coupler_defects: int = 0,
    defect_qubits=None,
    defect_couplers=None,
    basis: str = "Z",
    cal: str = CAL,
    seed: int | None = None,
    sw_offset: tuple[int, int] = (1, 1),
    token: str | None = None,
    api_url: str | None = None,
    quantum_computer: str = "emerald",
    use_timeslot: bool = False,
    refresh_calibration: bool = True,
    use_calibration_defects: bool | None = None,
    optimize_layout: bool = False,
    min_cz_fidelity: float | None = None,
    min_1q_fidelity: float | None = None,
    max_measurement_error: float | None = None,
    coupler_defect_mode: str = "unavailable",
    show_lattice_plot: bool = True,
    show_error_plot: bool = True,
) -> dict:
    """Run the SnL pipeline for several rounds and fit/plot logical error rate.

    Arguments match run_snl_pipeline, except `round_values` replaces `rounds`.
    The same SnL-deformed patch and defect locations are reused for all rounds.
    """
    if refresh_calibration:
        cal = base.refresh_calibration_if_needed(cal, api_url, token, quantum_computer)

    backend = None
    backend_couplers = None
    if mode == "hardware":
        with base._explicit_token_overrides_env(token):
            provider = base.IQMProvider(
                api_url or os.environ.get("IQM_API_URL", "https://resonance.iqm.tech/"),
                quantum_computer=quantum_computer,
                token=token or os.environ["IQM_TOKEN"],
            )
            backend = provider.get_backend()
        backend_couplers = iqm_couplers_from_backend(backend)

    if use_calibration_defects is None:
        use_calibration_defects = mode == "hardware"

    builder_kwargs = dict(
        distance=distance,
        num_qubit_defects=num_qubit_defects,
        num_coupler_defects=num_coupler_defects,
        defect_qubits=defect_qubits,
        defect_couplers=defect_couplers,
        seed=seed,
        cal=cal,
        backend_couplers=backend_couplers,
        use_calibration_defects=use_calibration_defects,
        min_cz_fidelity=min_cz_fidelity,
        min_1q_fidelity=min_1q_fidelity,
        max_measurement_error=max_measurement_error,
        coupler_defect_mode=coupler_defect_mode,
    )
    if optimize_layout:
        patch_data = build_optimized_defective_patch(**builder_kwargs)
    else:
        patch_data = build_defective_patch(sw_offset=sw_offset, **builder_kwargs)

    results = []
    for rounds in round_values:
        out = run_one_round_count(
            patch_data=patch_data,
            rounds=int(rounds),
            shots=shots,
            mode=mode,
            basis=basis,
            cal=cal,
            backend=backend,
            token=token,
            api_url=api_url,
            quantum_computer=quantum_computer,
            use_timeslot=use_timeslot,
        )
        out["calibration_path"] = cal
        results.append(out)
        print(
            f"rounds={rounds} | "
            f"logical_error_probability={out['summary']['logical_error_probability']:.5g} | "
            f"uncorrected={out['summary']['uncorrected_logical_error_probability']:.5g} | "
            f"swap_count={out['summary']['swap_count']} | "
            f"job={out['summary']['job_id']}"
        )

    fit = fit_snl_round_sweep(results)
    error_figure = plot_snl_logical_error_rate(fit) if show_error_plot else None
    lattice_figure = plot_snl_lattice(results[0]) if show_lattice_plot else None
    return {
        "results": results,
        "fit": fit,
        "error_figure": error_figure,
        "lattice_figure": lattice_figure,
        "summary": {
            "distance": distance,
            "round_values": list(round_values),
            "shots": shots,
            "mode": mode,
            "effective_distance": results[0]["summary"]["effective_distance"],
            "defect_qubits": results[0]["summary"]["defect_qubits"],
            "defect_couplers": results[0]["summary"]["defect_couplers"],
            "use_calibration_defects": results[0]["use_calibration_defects"],
            "optimize_layout": bool(results[0].get("optimize_layout", False)),
            "sw_offset": results[0]["sw_offset"],
            "logical_error_rate": fit["corrected_fit"]["epsilon_l"],
            "uncorrected_logical_error_rate": fit["uncorrected_fit"]["epsilon_l"],
            "swap_count": results[0]["summary"]["swap_count"],
            "skipped_hardware_operation_count": results[0]["summary"]["skipped_hardware_operation_count"],
        },
    }


if __name__ == "__main__":
    result = run_snl_pipeline(distance=3, rounds=1, shots=50, mode="synthetic", num_qubit_defects=1)
    print(result["summary"])
