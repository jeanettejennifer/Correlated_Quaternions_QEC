from __future__ import annotations

import functools
import json, math, os, re, sys, warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import networkx as nx
import numpy as np
import pymatching
import stim
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.transpiler import CouplingMap

try:
    from iqm.qiskit_iqm import IQMProvider
except Exception:  # pragma: no cover
    from qiskit_iqm import IQMProvider
from iqm.iqm_client import IQMClient

CAL = "calibration_data/2026-06-05T06_19_42.975934Z.json"
STIM_DIR = Path("stim_files")

for _snl_candidate in (
    os.environ.get("SNL_REPO"),
    "/private/tmp/snakes_and_ladders_adapting_the_surface_code_to_defects",
    str(Path(__file__).parent / "snakes_and_ladders_adapting_the_surface_code_to_defects"),
):
    if _snl_candidate and Path(_snl_candidate, "defects_module").exists() and _snl_candidate not in sys.path:
        sys.path.insert(0, _snl_candidate)

try:
    from defects_module.base import Pos as _EmeraldPos
except Exception:  # pragma: no cover
    class _EmeraldPos:
        def __init__(self, x: int, y: int):
            self.x = int(x)
            self.y = int(y)

        def __hash__(self):
            return hash((self.x, self.y))

        def __eq__(self, other):
            return getattr(other, "x", None) == self.x and getattr(other, "y", None) == self.y

IQM_EMERALD_POS = {
    54: (2, 0), 51: (4, 0), 46: (6, 0), 39: (8, 0),
    53: (1.25, 1), 50: (3.25, 1), 45: (5.25, 1), 38: (7.25, 1), 31: (9.25, 1),
    52: (0, 2), 49: (2, 2), 44: (4, 2), 37: (6, 2), 30: (8, 2),
    48: (1.25, 3), 43: (3.25, 3), 36: (5.25, 3), 29: (7.25, 3), 22: (9.25, 3),
    47: (0, 4), 42: (2, 4), 35: (4, 4), 28: (6, 4), 21: (8, 4), 14: (10, 4),
    41: (1.25, 5), 34: (3.25, 5), 27: (5.25, 5), 20: (7.25, 5), 13: (9.25, 5),
    40: (0, 6), 33: (2, 6), 26: (4, 6), 19: (6, 6), 12: (8, 6), 7: (10, 6),
    32: (1.25, 7), 25: (3.25, 7), 18: (5.25, 7), 11: (7.25, 7), 6: (9.25, 7),
    24: (2, 8), 17: (4, 8), 10: (6, 8), 5: (8, 8), 2: (10, 8),
    23: (1.25, 9), 16: (3.25, 9), 9: (5.25, 9), 4: (7.25, 9), 1: (9.25, 9),
    15: (2, 10), 8: (4, 10), 3: (6, 10),
}


def iqm_xy_to_snl_pos(xy: tuple[float, float]):
    x, y = xy
    y = int(round(y))
    return _EmeraldPos(int(round(x if y % 2 == 0 else x - 0.25)), y)


COORD_TO_QB = {iqm_xy_to_snl_pos(xy): qb for qb, xy in IQM_EMERALD_POS.items()}
QB_TO_COORD = {qb: iqm_xy_to_snl_pos(xy) for qb, xy in IQM_EMERALD_POS.items()}


def read_calibration_metrics(calibration_path: str = CAL) -> tuple[dict, dict, dict, set, set]:
    """Return one-qubit fidelities, readout errors, CZ fidelities, invalid qubits, and invalid couplers."""
    observations = json.loads(Path(calibration_path).read_text()).get("observations", [])
    oneq, meas, invalid_qubits, invalid_couplers, best_cz = {}, {}, set(), set(), {}
    for obs in observations:
        field = obs.get("dut_field", "")
        qubits = list(map(int, re.findall(r"QB(\d+)", field)))
        if obs.get("invalid") and qubits:
            if len(qubits) == 1:
                invalid_qubits.update(qubits)
            else:
                invalid_couplers.add(tuple(sorted(qubits[:2])))
            continue
        if not qubits:
            continue
        value = float(obs["value"])
        if len(qubits) == 1:
            if ("metrics.rb.prx" in field or "metrics.rb.clifford.xy_sx" in field) and ".fidelity" in field:
                oneq.setdefault(qubits[0], []).append(value)
            if "metrics.ssro.measure" in field and ("error_0_to_1" in field or "error_1_to_0" in field):
                meas.setdefault(qubits[0], []).append(value)
        elif ".cz." in field and ".fidelity" in field:
            edge = tuple(sorted(qubits[:2]))
            priority = 0 if field.startswith("metrics.irb.cz") else 1
            if edge not in best_cz or (priority, -value) < (best_cz[edge][1], -best_cz[edge][0]):
                best_cz[edge] = (value, priority)
    mean = lambda xs: float(np.mean(xs))
    return (
        {q: mean(vals) for q, vals in oneq.items()},
        {q: mean(vals) for q, vals in meas.items()},
        {edge: val for edge, (val, _) in best_cz.items()},
        invalid_qubits,
        invalid_couplers,
    )


# ======================================================================# 0. CALIBRATION AND COMMON UTILITIES
# ======================================================================#
# This support section provides the current IQM calibration data, maps IQM
# qubit labels to backend indices, and keeps token handling compatible with the
# IQM client. The actual experiment flow starts in section 1 below.

@contextmanager
def _explicit_token_overrides_env(token: str | None):
    """Avoid IQM client errors when both token=... and IQM_TOKEN are set."""
    old = os.environ.pop("IQM_TOKEN", None) if token else None
    try:
        yield
    finally:
        if old is not None:
            os.environ["IQM_TOKEN"] = old


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def calibration_date(path: str = CAL):
    d = json.loads(Path(path).read_text())
    ts = _parse_timestamp(d.get("end_timestamp")) or _parse_timestamp(d.get("created_timestamp"))
    return ts.date() if ts else None


def find_todays_calibration(cal: str = CAL) -> str | None:
    """Return an existing calibration JSON from today, preferring newest first."""
    today = datetime.now(timezone.utc).date()
    directory = Path(cal).parent
    if not directory.exists():
        return None
    candidates = []
    for path in directory.glob("*.json"):
        try:
            if calibration_date(str(path)) == today:
                candidates.append(path)
        except Exception:
            continue
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def refresh_calibration_if_needed(
    cal: str = CAL,
    api_url: str | None = None,
    token: str | None = None,
    quantum_computer: str = "emerald",
    force: bool = False,
) -> str:
    """Download today's default IQM quality metrics if the saved calibration is stale."""
    today = datetime.now(timezone.utc).date()
    path = Path(cal)
    if path.exists() and not force and calibration_date(cal) == today:
        print(f"Loaded today's calibration data: {path}")
        return str(path)
    if not force:
        existing = find_todays_calibration(cal)
        if existing is not None:
            print(f"Loaded today's calibration data: {existing}")
            return existing

    url = api_url or os.environ.get("IQM_API_URL", "https://resonance.iqm.tech/")
    tok = token or os.environ.get("IQM_TOKEN")
    if not tok:
        print(f"No IQM_TOKEN set; using existing calibration data: {path}")
        return str(path)

    with _explicit_token_overrides_env(token):
        client = IQMClient(url, quantum_computer=quantum_computer, token=tok)
        metrics = client.get_quality_metric_set()
    data = metrics.model_dump(mode="json")
    ts = data.get("end_timestamp") or data.get("created_timestamp") or datetime.now(timezone.utc).isoformat()
    safe_ts = ts.replace(":", "_").replace("+00:00", "Z")
    out = path.parent / f"{safe_ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"Downloaded fresh calibration data: {out}")
    return str(out)


# ======================================================================# 1. STIM CIRCUIT
# ======================================================================#
# Build the rotated surface-code memory experiment in Stim. This ideal Stim
# circuit is the source of truth for qubit coordinates, measurement order,
# detector definitions, and the logical observable.

def stim_circuit(d: int, r: int, basis: str = "Z", noise: dict | None = None) -> stim.Circuit:
    n = noise or {}
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=r,
        before_round_data_depolarization=n.get("idle", 0),
        before_measure_flip_probability=n.get("meas", 0),
        after_reset_flip_probability=n.get("reset", 0),
        after_clifford_depolarization=n.get("gate", 0),
    )


def write_stim_file(c: stim.Circuit, distance: int, rounds: int, basis: str = "Z", stem: str | None = None) -> str:
    """Write a Stim circuit into stim_files/ and return the path."""
    STIM_DIR.mkdir(parents=True, exist_ok=True)
    name = stem or f"distance_{distance}_rounds_{rounds}_surface_code_{basis.lower()}"
    path = STIM_DIR / f"{name}.stim"
    path.write_text(str(c))
    return str(path)


# ======================================================================# 2. QISKIT
# ======================================================================#
# Convert the Stim circuit into a Qiskit circuit laid out on an optimized IQM
# Emerald patch. Mapping is chosen by embedding the Stim interaction graph into
# the calibrated Emerald CZ graph and maximizing the product of used CZ
# fidelities. For hardware execution, IQM labels like QB11 are converted to the
# backend's Qiskit integer indices before transpilation.

def twoq_edges(c: stim.Circuit) -> Counter[tuple[int, int]]:
    out = Counter()
    for ins in c.flattened():
        if ins.name in ("CX", "CZ"):
            t = ins.targets_copy()
            for i in range(0, len(t), 2):
                out[tuple(sorted((int(t[i].value), int(t[i + 1].value))))] += 1
    return out


def calibration_graph(path: str = CAL) -> nx.Graph:
    g, best = nx.Graph(), {}
    for o in json.loads(Path(path).read_text())["observations"]:
        if o.get("invalid"): continue
        f = o["dut_field"]
        if ".cz." not in f or ".fidelity" not in f: continue
        m = re.search(r"QB(\d+)__QB(\d+)\.fidelity", f)
        if not m: continue
        e = tuple(sorted(map(int, m.groups()))); val = float(o["value"])
        prio = 0 if f.startswith("metrics.irb.cz") else 1
        if e not in best or (prio, -val) < (best[e][1], -best[e][0]): best[e] = (val, prio, f)
    for (a, b), (fid, _, field) in best.items(): g.add_edge(a, b, fidelity=fid, field=field)
    return g


def best_mapping(c: stim.Circuit, cal: str = CAL) -> dict:
    """Return {stim_qubit_index: iqm_qb_label_number} for the best calibrated patch."""
    hw, edges = calibration_graph(cal), twoq_edges(c)
    pat = nx.Graph(); pat.add_nodes_from(map(int, c.get_final_qubit_coordinates())); pat.add_edges_from(edges)
    best = None
    for hw_to_stim in nx.algorithms.isomorphism.GraphMatcher(hw, pat).subgraph_isomorphisms_iter():
        m = {s: h for h, s in hw_to_stim.items()}
        score = sum(n * math.log(hw.edges[m[a], m[b]]["fidelity"]) for (a, b), n in edges.items())
        minfid = min(hw.edges[m[a], m[b]]["fidelity"] for a, b in edges)
        cand = (score, minfid, dict(sorted(m.items())))
        if best is None or cand[:2] > best[:2]: best = cand
    if best is None: raise ValueError("No calibrated embedding found")
    return best[2]


def backend_couplers_as_iqm_labels(backend) -> set[tuple[int, int]]:
    """Return native backend couplers as IQM labels, e.g. {(45, 46), ...}."""
    couplers = set()
    for a, b in backend.coupling_map.get_edges():
        qa = int(re.search(r"QB(\d+)", backend.index_to_qubit_name(a)).group(1))
        qb = int(re.search(r"QB(\d+)", backend.index_to_qubit_name(b)).group(1))
        couplers.add(tuple(sorted((qa, qb))))
    return couplers


def _stim_coord_points(c: stim.Circuit) -> dict[int, tuple[int, int]]:
    return {int(q): (int(round(x)), int(round(y))) for q, (x, y) in c.get_final_qubit_coordinates().items()}


def _d4_transforms(x: int, y: int) -> tuple[tuple[int, int], ...]:
    return ((x, y), (x, -y), (-x, y), (-x, -y), (y, x), (y, -x), (-y, x), (-y, -x))


def geometric_mappings(c: stim.Circuit) -> list[dict[int, int]]:
    """All rigid/reflected placements of the Stim patch inside the real Emerald map."""
    if not COORD_TO_QB:
        return []
    coords = _stim_coord_points(c)
    hw_positions = list(COORD_TO_QB)
    out, seen = [], set()
    for transform_index in range(8):
        transformed = {q: _d4_transforms(x, y)[transform_index] for q, (x, y) in coords.items()}
        q0, p0 = next(iter(transformed.items()))
        for hw_anchor in hw_positions:
            tx, ty = hw_anchor.x - p0[0], hw_anchor.y - p0[1]
            mapping = {}
            used_qbs = set()
            ok = True
            for q, (x, y) in transformed.items():
                probe = type(hw_anchor)(x + tx, y + ty)
                qb = COORD_TO_QB.get(probe)
                if qb is None or qb in used_qbs:
                    ok = False
                    break
                mapping[q] = qb
                used_qbs.add(qb)
            if ok:
                key = tuple(sorted(mapping.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(dict(sorted(mapping.items())))
    return out


def mapping_score(
    c: stim.Circuit,
    mapping: dict[int, int],
    cal: str = CAL,
    backend_couplers: set[tuple[int, int]] | None = None,
) -> tuple[float, int, int, float]:
    """Score a mapping; higher is better, but missing hardware gates are allowed."""
    if read_calibration_metrics is None:
        hw = calibration_graph(cal)
        oneq, meas, cz, invalid_qubits, invalid_couplers = {}, {}, {tuple(sorted(e)): hw.edges[e].get("fidelity", 0.995) for e in hw.edges}, set(), set()
    else:
        oneq, meas, cz, invalid_qubits, invalid_couplers = read_calibration_metrics(cal)

    used_edges = twoq_edges(c)
    usable_edges, missing_edges = 0, 0
    terms = []
    min_cz = 1.0
    for stim_q, iqm_q in mapping.items():
        if iqm_q in invalid_qubits:
            terms.append(math.log(1e-6))
        else:
            terms.append(math.log(max(oneq.get(iqm_q, 0.999), 1e-6)))
            terms.append(math.log(max(1 - meas.get(iqm_q, 0.01), 1e-6)))

    for (a, b), count in used_edges.items():
        edge = tuple(sorted((mapping[a], mapping[b])))
        native = edge not in invalid_couplers and edge in cz and (backend_couplers is None or edge in backend_couplers)
        if native:
            usable_edges += count
            fid = max(cz.get(edge, 0.995), 1e-6)
            min_cz = min(min_cz, fid)
            terms.extend([math.log(fid)] * count)
        else:
            missing_edges += count
            terms.extend([math.log(1e-3)] * count)
    return (float(sum(terms)), usable_edges, -missing_edges, min_cz)


def optimized_mapping(
    c: stim.Circuit,
    cal: str = CAL,
    backend=None,
    allow_skipped_gates: bool = True,
) -> tuple[dict[int, int], dict]:
    """Choose a calibrated Emerald placement, allowing clipped gates if requested."""
    backend_couplers = backend_couplers_as_iqm_labels(backend) if backend is not None else None
    candidates = []

    try:
        exact = best_mapping(c, cal)
        candidates.append(("exact_graph", exact))
    except Exception as exc:
        exact_error = str(exc)
    else:
        exact_error = None

    for mapping in geometric_mappings(c):
        candidates.append(("geometry", mapping))

    if not candidates:
        raise ValueError("No Emerald placement candidates found for this Stim circuit.")

    ranked = []
    for source, mapping in candidates:
        score = mapping_score(c, mapping, cal=cal, backend_couplers=backend_couplers)
        if allow_skipped_gates or score[2] == 0:
            ranked.append((score, source, mapping))
    if not ranked:
        raise ValueError("No placement has all required native couplers.")

    score, source, mapping = max(ranked, key=lambda x: x[0])
    return mapping, {
        "mapping_source": source,
        "score": score[0],
        "usable_twoq_gate_count": score[1],
        "missing_twoq_gate_count": -score[2],
        "min_used_cz_fidelity": score[3],
        "candidate_count": len(candidates),
        "exact_graph_error": exact_error,
    }


def stim_to_qiskit_mapped(c: stim.Circuit, mapping: dict) -> tuple[QuantumCircuit, list[tuple[int, int, int]]]:
    qr, cr = QuantumRegister(max(mapping.values()) + 1, "q"), ClassicalRegister(c.num_measurements, "m")
    qc, meas, k = QuantumCircuit(qr, cr), [], 0
    q = lambda t: qr[mapping[int(t.value)]]
    for ins in c.flattened():
        name, t = ins.name, ins.targets_copy()
        if name in {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"} or name.endswith("ERROR") or name.startswith(("DEPOLARIZE", "PAULI_CHANNEL")): continue
        if name == "TICK": qc.barrier(); continue
        if name == "I": continue
        if name in ("R", "RZ", "MR"):
            if name == "MR":
                for x in t:
                    qc.measure(q(x), cr[k]); meas.append((int(x.value), mapping[int(x.value)], k)); k += 1
                for x in t:
                    qc.reset(q(x))
            else:
                for x in t: qc.reset(q(x))
            continue
        if name == "H":
            for x in t: qc.h(q(x))
        elif name in ("M", "MZ"):
            for x in t: qc.measure(q(x), cr[k]); meas.append((int(x.value), mapping[int(x.value)], k)); k += 1
        elif name in ("CX", "CNOT"):
            for i in range(0, len(t), 2): qc.cx(q(t[i]), q(t[i + 1]))
        elif name == "CZ":
            for i in range(0, len(t), 2): qc.cz(q(t[i]), q(t[i + 1]))
        else:
            raise NotImplementedError(name)
    return qc, meas


def stim_to_qiskit_mapped_tolerant(
    c: stim.Circuit,
    mapping: dict,
    backend=None,
    unavailable_backend_qubits: set[int] | None = None,
    unavailable_backend_couplers: set[tuple[int, int]] | None = None,
) -> tuple[QuantumCircuit, list[tuple[int, int, int]], list[tuple]]:
    """Convert Stim to Qiskit, warning and skipping impossible hardware operations."""
    unavailable_backend_qubits = unavailable_backend_qubits or set()
    unavailable_backend_couplers = {tuple(sorted(edge)) for edge in (unavailable_backend_couplers or set())}
    qr, cr = QuantumRegister(max(mapping.values()) + 1, "q"), ClassicalRegister(c.num_measurements, "m")
    qc, meas, skipped, k = QuantumCircuit(qr, cr), [], [], 0
    hw_edges = {tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()} if backend is not None else None

    def mapped(t):
        q = mapping.get(int(t.value))
        if q in unavailable_backend_qubits:
            return None
        return q

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
                    skipped.append((name, (int(x.value),), "unavailable or unmapped qubit"))
                else:
                    qc.reset(qr[qx])
            continue
        if name == "H":
            for x in t:
                qx = mapped(x)
                if qx is None:
                    skipped.append((name, (int(x.value),), "unavailable or unmapped qubit"))
                else:
                    qc.h(qr[qx])
            continue
        if name in ("M", "MZ", "MR"):
            for x in t:
                qx = mapped(x)
                if qx is None:
                    skipped.append((name, (int(x.value),), "measurement left as classical 0"))
                else:
                    qc.measure(qr[qx], cr[k])
                    meas.append((int(x.value), qx, k))
                k += 1
            if name == "MR":
                for x in t:
                    qx = mapped(x)
                    if qx is None:
                        skipped.append(("R", (int(x.value),), "reset after MR skipped"))
                    else:
                        qc.reset(qr[qx])
            continue
        if name in ("CX", "CNOT", "CZ"):
            for i in range(0, len(t), 2):
                qa, qb = mapped(t[i]), mapped(t[i + 1])
                stim_edge = (int(t[i].value), int(t[i + 1].value))
                if qa is None or qb is None:
                    skipped.append((name, stim_edge, "unavailable or unmapped endpoint"))
                    continue
                if tuple(sorted((qa, qb))) in unavailable_backend_couplers:
                    skipped.append((name, stim_edge, "explicitly unavailable backend coupler"))
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
        raise ValueError(f"Measurement mismatch: Qiskit has {k}, Stim has {c.num_measurements}.")
    if skipped:
        warnings.warn(f"Skipped {len(skipped)} impossible hardware operation(s). First skipped operations: {skipped[:8]}")
    return qc, meas, skipped


def iqm_label_mapping_to_backend_indices(mapping: dict[int, int], backend) -> dict[int, int]:
    """Convert {stim_q: IQM QB label number} into {stim_q: Qiskit backend index}."""
    return {stim_q: backend.qubit_name_to_index(f"QB{qb}") for stim_q, qb in mapping.items()}


def unavailable_backend_qubits_from_calibration(cal: str, backend) -> set[int]:
    """Backend indices for qubits explicitly marked invalid in calibration data."""
    if read_calibration_metrics is None:
        return set()
    _, _, _, invalid_qubits, _ = read_calibration_metrics(cal)
    unavailable = set()
    for qb in invalid_qubits:
        try:
            unavailable.add(backend.qubit_name_to_index(f"QB{qb}"))
        except Exception:
            continue
    return unavailable


def validate(qc: QuantumCircuit, c: stim.Circuit, mapping: dict, cal: str = CAL):
    hw_edges = set(tuple(sorted(e)) for e in calibration_graph(cal).edges)
    assert qc.num_clbits == c.num_measurements
    assert all(tuple(sorted((mapping[a], mapping[b]))) in hw_edges for a, b in twoq_edges(c))
    cm = CouplingMap(sorted(hw_edges))
    tqc = transpile(qc, coupling_map=cm, basis_gates=["cz", "h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "id", "reset", "measure"], optimization_level=1, initial_layout=list(range(qc.num_qubits)))
    assert tqc.count_ops().get("swap", 0) == 0


def validate_backend_mapping(qc: QuantumCircuit, c: stim.Circuit, backend_mapping: dict, backend):
    """Check that the backend-index circuit uses real backend couplers without swaps."""
    if qc.num_clbits != c.num_measurements:
        raise ValueError(f"Measurement mismatch: Qiskit has {qc.num_clbits}, Stim has {c.num_measurements}.")
    hw_edges = {tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()}
    missing = []
    for a, b in twoq_edges(c):
        edge = tuple(sorted((backend_mapping[a], backend_mapping[b])))
        if edge not in hw_edges:
            missing.append({
                "stim_edge": (a, b),
                "backend_edge": edge,
                "iqm_edge": tuple(backend.index_to_qubit_name(q) for q in edge),
            })
    if missing:
        raise ValueError(f"Circuit uses {len(missing)} non-native backend coupler(s): {missing[:10]}")


def identity_initial_layout(qc: QuantumCircuit) -> list[int]:
    """Pin q[i] to hardware qubit i after building a physically indexed circuit."""
    return list(range(qc.num_qubits))


# ======================================================================# 2b. OPTIMIZED QISKIT MAPPING AND CONVERSION
# ======================================================================#
# Drop-in replacements for the Section 2 functions with the ``optimized_``
# prefix. Key improvements:
#   - calibration_graph is parsed once and cached (lru_cache).
#   - The Stim circuit is flattened once; the instruction list is reused by
#     edge extraction, Qiskit conversion, and validation.
#   - best_mapping pre-prunes low-fidelity edges before isomorphism search.
#   - stim_to_qiskit_mapped skips TICK barriers to reduce Qiskit overhead.
#   - validate checks edge membership directly instead of re-transpiling.


@functools.lru_cache(maxsize=8)
def optimized_calibration_graph(path: str = CAL) -> nx.Graph:
    """Cached version of calibration_graph — parses the JSON only once per path."""
    g, best = nx.Graph(), {}
    for o in json.loads(Path(path).read_text())["observations"]:
        if o.get("invalid"):
            continue
        f = o["dut_field"]
        if ".cz." not in f or ".fidelity" not in f:
            continue
        m = re.search(r"QB(\d+)__QB(\d+)\.fidelity", f)
        if not m:
            continue
        e = tuple(sorted(map(int, m.groups())))
        val = float(o["value"])
        prio = 0 if f.startswith("metrics.irb.cz") else 1
        if e not in best or (prio, -val) < (best[e][1], -best[e][0]):
            best[e] = (val, prio, f)
    for (a, b), (fid, _, field) in best.items():
        g.add_edge(a, b, fidelity=fid, field=field)
    return g


def optimized_flatten_circuit(c: stim.Circuit) -> list:
    """Flatten a Stim circuit once and return the instruction list for reuse."""
    return list(c.flattened())


def optimized_twoq_edges(instructions: list) -> Counter[tuple[int, int]]:
    """Extract two-qubit gate edges from a pre-flattened instruction list.

    Unlike ``twoq_edges`` this avoids re-flattening the Stim circuit.
    """
    out: Counter[tuple[int, int]] = Counter()
    for ins in instructions:
        if ins.name in ("CX", "CZ"):
            t = ins.targets_copy()
            for i in range(0, len(t), 2):
                out[tuple(sorted((int(t[i].value), int(t[i + 1].value))))] += 1
    return out


def optimized_best_mapping(
    c: stim.Circuit,
    cal: str = CAL,
    min_fidelity: float = 0.90,
    *,
    _flat: list | None = None,
) -> dict:
    """Find the best hardware embedding with pre-pruned low-fidelity edges.

    Improvements over ``best_mapping``:
      1. Uses the cached calibration graph.
      2. Prunes hardware edges below *min_fidelity* before isomorphism search,
         drastically reducing the candidate space for larger distances.
      3. Accepts an optional pre-flattened instruction list to avoid redundant
         flattening.
    """
    hw = optimized_calibration_graph(cal)
    instructions = _flat if _flat is not None else optimized_flatten_circuit(c)
    edges = optimized_twoq_edges(instructions)

    # Build a pruned copy of the hardware graph.
    hw_pruned = nx.Graph(
        (u, v, d)
        for u, v, d in hw.edges(data=True)
        if d["fidelity"] >= min_fidelity
    )

    pat = nx.Graph()
    pat.add_nodes_from(map(int, c.get_final_qubit_coordinates()))
    pat.add_edges_from(edges)

    best = None
    for hw_to_stim in nx.algorithms.isomorphism.GraphMatcher(
        hw_pruned, pat
    ).subgraph_isomorphisms_iter():
        m = {s: h for h, s in hw_to_stim.items()}
        score = sum(
            n * math.log(hw.edges[m[a], m[b]]["fidelity"])
            for (a, b), n in edges.items()
        )
        minfid = min(hw.edges[m[a], m[b]]["fidelity"] for a, b in edges)
        cand = (score, minfid, dict(sorted(m.items())))
        if best is None or cand[:2] > best[:2]:
            best = cand

    if best is None:
        raise ValueError(
            f"No calibrated embedding found (min_fidelity={min_fidelity}). "
            "Try lowering min_fidelity."
        )
    return best[2]


# Pre-computed sets for fast instruction filtering in the converter.
_SKIP_NAMES = frozenset({
    "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS",
})
_SKIP_PREFIXES = ("DEPOLARIZE", "PAULI_CHANNEL")


def optimized_stim_to_qiskit_mapped(
    c: stim.Circuit,
    mapping: dict,
    *,
    _flat: list | None = None,
    emit_barriers: bool = False,
) -> tuple[QuantumCircuit, list[tuple[int, int, int]]]:
    """Convert a Stim circuit to a mapped Qiskit circuit.

    Improvements over ``stim_to_qiskit_mapped``:
      1. Accepts an optional pre-flattened instruction list (*_flat*) so the
         circuit is not re-flattened.
      2. TICK barriers are **skipped by default** (``emit_barriers=False``),
         which substantially reduces Qiskit circuit size and speeds up any
         downstream transpilation.  Set ``emit_barriers=True`` to restore the
         original behaviour for debugging.
    """
    instructions = _flat if _flat is not None else optimized_flatten_circuit(c)

    qr = QuantumRegister(max(mapping.values()) + 1, "q")
    cr = ClassicalRegister(c.num_measurements, "m")
    qc = QuantumCircuit(qr, cr)
    meas: list[tuple[int, int, int]] = []
    k = 0

    def q(t):
        return qr[mapping[int(t.value)]]

    for ins in instructions:
        name = ins.name

        # Skip metadata, noise, and annotation instructions.
        if name in _SKIP_NAMES or name.endswith("ERROR") or name.startswith(_SKIP_PREFIXES):
            continue

        t = ins.targets_copy()

        if name == "TICK":
            if emit_barriers:
                qc.barrier()
            continue

        if name in ("R", "RZ", "MR"):
            if name == "MR":
                for x in t:
                    qc.measure(q(x), cr[k])
                    meas.append((int(x.value), mapping[int(x.value)], k))
                    k += 1
                for x in t:
                    qc.reset(q(x))
            else:
                for x in t:
                    qc.reset(q(x))
            continue

        if name == "H":
            for x in t:
                qc.h(q(x))
        elif name in ("M", "MZ"):
            for x in t:
                qc.measure(q(x), cr[k])
                meas.append((int(x.value), mapping[int(x.value)], k))
                k += 1
        elif name in ("CX", "CNOT"):
            for i in range(0, len(t), 2):
                qc.cx(q(t[i]), q(t[i + 1]))
        elif name == "CZ":
            for i in range(0, len(t), 2):
                qc.cz(q(t[i]), q(t[i + 1]))
        else:
            raise NotImplementedError(name)

    return qc, meas


def optimized_validate(
    qc: QuantumCircuit,
    c: stim.Circuit,
    mapping: dict,
    cal: str = CAL,
    *,
    _flat: list | None = None,
) -> None:
    """Validate a mapped Qiskit circuit without re-transpiling.

    Improvements over ``validate``:
      1. Uses the cached calibration graph.
      2. Accepts a pre-flattened instruction list.
      3. Checks edge membership directly instead of running a full Qiskit
         transpile pass just to assert zero SWAPs.  Since every gate was
         already placed by ``optimized_stim_to_qiskit_mapped`` on a real
         hardware coupler, the transpile check is redundant.
    """
    hw_edges = frozenset(
        tuple(sorted(e)) for e in optimized_calibration_graph(cal).edges
    )
    instructions = _flat if _flat is not None else optimized_flatten_circuit(c)
    edges = optimized_twoq_edges(instructions)

    assert qc.num_clbits == c.num_measurements, (
        f"clbit count mismatch: circuit has {qc.num_clbits}, "
        f"expected {c.num_measurements}"
    )
    for (a, b) in edges:
        mapped_edge = tuple(sorted((mapping[a], mapping[b])))
        assert mapped_edge in hw_edges, (
            f"CZ edge ({mapping[a]}, {mapping[b]}) not in calibrated hardware couplers"
        )


def optimized_validate_backend_mapping(
    qc: QuantumCircuit,
    c: stim.Circuit,
    backend_mapping: dict,
    backend,
    *,
    _flat: list | None = None,
) -> None:
    """Validate backend-index mapping, reusing a pre-flattened instruction list."""
    assert qc.num_clbits == c.num_measurements
    hw_edges = frozenset(
        tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()
    )
    instructions = _flat if _flat is not None else optimized_flatten_circuit(c)
    edges = optimized_twoq_edges(instructions)
    for (a, b) in edges:
        mapped_edge = tuple(sorted((backend_mapping[a], backend_mapping[b])))
        assert mapped_edge in hw_edges, (
            f"CZ edge ({backend_mapping[a]}, {backend_mapping[b]}) not on backend"
        )


# ======================================================================# 3. IQM EMERALD (RESONANCE)
# ======================================================================#
# Submit the Qiskit circuit to IQM Resonance. Hardware runs use a pinned
# identity layout after converting selected IQM QB labels to backend indices.
# Synthetic runs sample from the calibrated Stim noise model but otherwise use
# the same mapping and decoder path.

def result_to_memory(result) -> tuple[list[str], dict | None]:
    """Get per-shot bitstrings from an IQM/Qiskit result, falling back to counts."""
    try:
        memory = result.get_memory()
        if isinstance(memory, str):
            memory = [memory]
        if memory:
            return list(memory), None
    except Exception:
        pass

    counts = result.get_counts()
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError("Expected one circuit result, but result.get_counts() returned multiple entries.")
        counts = counts[0]
    memory = []
    for bitstring, count in counts.items():
        memory.extend([bitstring] * int(count))
    return memory, counts


# ======================================================================# 4. SYNDROME EXTRACTION
# ======================================================================#
# Convert Qiskit/IQM bitstrings back into Stim measurement order, then use
# Stim's measurement-to-detector converter.

def qiskit_memory_to_array(memory: list[str], n: int) -> np.ndarray:
    return np.array([[int(b) for b in s.replace(" ", "")[::-1][:n]] for s in memory], dtype=bool)


def stim_measurements_to_qiskit_memory(measurements: np.ndarray) -> list[str]:
    """Format Stim-order sampled measurements like Qiskit/IQM bitstrings."""
    return ["".join(str(int(b)) for b in row.astype(np.uint8))[::-1] for row in measurements]


def extract_syndromes(c: stim.Circuit, memory: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (detector_events, observable_flips) from Qiskit/IQM memory."""
    return c.compile_m2d_converter().convert(measurements=qiskit_memory_to_array(memory, c.num_measurements), separate_observables=True)


# ======================================================================# 5. DECODER
# ======================================================================#
# Build the detector error model used by PyMatching from either average
# calibration-derived noise or qubit/coupler-specific calibration noise, then
# decode detector events into predicted logical observables.

def calibration_noise(mapping: dict, c: stim.Circuit, cal: str = CAL) -> dict:
    data = [o for o in json.loads(Path(cal).read_text())["observations"] if not o.get("invalid")]
    used, used_edges = set(mapping.values()), {tuple(sorted((mapping[a], mapping[b]))) for a, b in twoq_edges(c)}
    oneq, meas, cz = [], [], []
    for o in data:
        f, v = o["dut_field"], float(o["value"])
        qs = list(map(int, re.findall(r"QB(\d+)", f)))
        if len(qs) == 1 and qs[0] in used:
            if ("metrics.rb.prx" in f or "metrics.rb.clifford.xy_sx" in f) and ".fidelity" in f: oneq.append(v)
            if "metrics.ssro.measure" in f and ("error_0_to_1" in f or "error_1_to_0" in f): meas.append(v)
        if len(qs) >= 2 and tuple(sorted(qs[:2])) in used_edges and ".cz." in f and ".fidelity" in f: cz.append(v)
    mean = lambda xs, d: float(np.mean(xs)) if xs else d
    p1, p2, pm = 1 - mean(oneq, .999), 1 - mean(cz, .995), mean(meas, .01)
    return {"idle": max(p1, 1e-6), "meas": max(pm, 1e-6), "reset": max(pm, 1e-6), "gate": max(p1, p2, 1e-6)}


def calibration_error_maps(mapping: dict, c: stim.Circuit, cal: str = CAL) -> dict:
    data = [o for o in json.loads(Path(cal).read_text())["observations"] if not o.get("invalid")]
    used, used_edges = set(mapping.values()), {tuple(sorted((mapping[a], mapping[b]))) for a, b in twoq_edges(c)}
    oneq, meas, cz = {}, {}, {}

    for o in data:
        f, v = o["dut_field"], float(o["value"])
        qs = list(map(int, re.findall(r"QB(\d+)", f)))

        if len(qs) == 1 and qs[0] in used:
            q = qs[0]
            if ("metrics.rb.prx" in f or "metrics.rb.clifford.xy_sx" in f) and ".fidelity" in f:
                oneq.setdefault(q, []).append(1 - v)
            if "metrics.ssro.measure" in f and ("error_0_to_1" in f or "error_1_to_0" in f):
                meas.setdefault(q, []).append(v)

        if len(qs) >= 2:
            e = tuple(sorted(qs[:2]))
            if e in used_edges and ".cz." in f and ".fidelity" in f:
                cz.setdefault(e, []).append(1 - v)

    mean = lambda xs, d: float(np.mean(xs)) if xs else d
    avg = calibration_noise(mapping, c, cal)
    return {
        "oneq": {q: mean(v, avg["idle"]) for q, v in oneq.items()},
        "meas": {q: mean(v, avg["meas"]) for q, v in meas.items()},
        "reset": {q: mean(meas.get(q, []), avg["reset"]) for q in used},
        "cz": {e: mean(v, avg["gate"]) for e, v in cz.items()},
        "fallback": avg,
    }


def exact_calibrated_decoder_circuit(c: stim.Circuit, mapping: dict, cal: str = CAL) -> tuple[stim.Circuit, dict]:
    err = calibration_error_maps(mapping, c, cal)
    out = stim.Circuit()

    def iqm_q(stim_q: int) -> int:
        return mapping[int(stim_q)]

    def p1(stim_q: int) -> float:
        return max(err["oneq"].get(iqm_q(stim_q), err["fallback"]["idle"]), 1e-6)

    def pm(stim_q: int) -> float:
        return max(err["meas"].get(iqm_q(stim_q), err["fallback"]["meas"]), 1e-6)

    def pr(stim_q: int) -> float:
        return max(err["reset"].get(iqm_q(stim_q), err["fallback"]["reset"]), 1e-6)

    def p2(a: int, b: int) -> float:
        e = tuple(sorted((iqm_q(a), iqm_q(b))))
        return max(err["cz"].get(e, err["fallback"]["gate"]), 1e-6)

    for ins in c.flattened():
        name, t = ins.name, ins.targets_copy()

        if name in ("M", "MZ", "MR"):
            for x in t:
                if x.is_qubit_target: out.append("X_ERROR", [x.value], pm(x.value))
            out.append(ins)
            if name == "MR":
                for x in t:
                    if x.is_qubit_target: out.append("X_ERROR", [x.value], pr(x.value))
            continue

        out.append(ins)

        if name in ("R", "RZ"):
            for x in t:
                if x.is_qubit_target: out.append("X_ERROR", [x.value], pr(x.value))
        elif name in ("H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Z", "SQRT_Z_DAG"):
            for x in t:
                if x.is_qubit_target: out.append("DEPOLARIZE1", [x.value], p1(x.value))
        elif name in ("CX", "CNOT", "CZ"):
            for i in range(0, len(t), 2):
                out.append("DEPOLARIZE2", [t[i].value, t[i + 1].value], p2(t[i].value, t[i + 1].value))

    return out, err


def calibrated_decoder_circuit(c: stim.Circuit, mapping: dict, d: int, r: int, basis: str = "Z", cal: str = CAL, mode: str = "exact") -> tuple[stim.Circuit, dict]:
    if mode == "exact":
        return exact_calibrated_decoder_circuit(c, mapping, cal)
    if mode == "average":
        noise = calibration_noise(mapping, c, cal)
        return stim_circuit(d, r, basis, noise), noise
    raise ValueError("mode must be 'exact' or 'average'")


def add_synthetic_defect_noise(
    c: stim.Circuit,
    defect_qubits=(),
    defect_couplers=(),
    probability: float = 0.5,
) -> stim.Circuit:
    """Add strong depolarizing noise to selected Stim qubits/couplers."""
    defect_qubits = [int(q) for q in defect_qubits or ()]
    defect_couplers = {tuple(sorted((int(a), int(b)))) for a, b in (defect_couplers or ())}
    if not defect_qubits and not defect_couplers:
        return c
    out = stim.Circuit()
    for ins in c.flattened():
        name, targets = ins.name, ins.targets_copy()
        out.append(ins.name, ins.targets_copy(), ins.gate_args_copy())
        if name == "TICK" and defect_qubits:
            out.append("DEPOLARIZE1", defect_qubits, probability)
        elif name in ("CX", "CNOT", "CZ") and defect_couplers:
            for i in range(0, len(targets), 2):
                edge = tuple(sorted((int(targets[i].value), int(targets[i + 1].value))))
                if edge in defect_couplers:
                    out.append("DEPOLARIZE2", [edge[0], edge[1]], probability)
    return out


def decode(c: stim.Circuit, decoder_c: stim.Circuit, memory: list[str]) -> dict:
    dets, obs = extract_syndromes(c, memory)
    pred = pymatching.Matching.from_detector_error_model(decoder_c.detector_error_model(decompose_errors=True)).decode_batch(dets).astype(bool)
    obs = obs.astype(bool); err = np.logical_xor(pred, obs)
    return {"detectors": dets, "observable_flips": obs, "predictions": pred, "logical_errors": err,
            "corrected_ler": float(err.any(axis=1).mean()), "uncorrected_ler": float(obs.any(axis=1).mean())}


def audit_pipeline_result(out: dict) -> dict:
    """Compact run summary for plotting sweeps."""
    det_rates = out["decoded"]["detectors"].mean(axis=0)
    diagnostics = out.get("transpile_diagnostics") or {}
    return {
        "shots": len(out["memory"]),
        "swap_count": diagnostics.get("swap_count"),
        "corrected_ler": out["decoded"]["corrected_ler"],
        "uncorrected_ler": out["decoded"]["uncorrected_ler"],
        "mean_detector_firing_rate": float(det_rates.mean()),
        "max_detector_firing_rate": float(det_rates.max()),
        "hot_detector_fraction_gt_0p2": float((det_rates > 0.2).mean()),
    }


# ======================================================================# 6. LER
# ======================================================================#
# A single job returns a logical error probability after a fixed number of
# rounds. To report a logical error rate per round, fit probabilities from a
# sweep over round counts to:
#     P_error(n) = (1 - (1 - 2 epsilon_L)**n) / 2.

def logical_error_probability(rounds, epsilon_l: float):
    """Model P_error(n) = (1 - (1 - 2 epsilon_l)**n) / 2."""
    rounds = np.asarray(rounds, dtype=float)
    epsilon_l = np.asarray(epsilon_l, dtype=float)
    return 0.5 * (1 - (1 - 2 * epsilon_l) ** rounds)


def fit_logical_error_rate(rounds, probabilities, shots=None, grid_size: int = 200_000) -> dict:
    """Fit per-round logical error rate from logical error probabilities.

    The fitted model is 2 P_error = 1 - (1 - 2 epsilon_L)**rounds.
    """
    rounds = np.asarray(rounds, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if rounds.shape != probabilities.shape:
        raise ValueError("rounds and probabilities must have the same shape")
    if np.any((probabilities < 0) | (probabilities > 0.5)):
        raise ValueError("probabilities must be in [0, 0.5] for this fit")

    eps = np.linspace(0, 0.499999, grid_size)
    pred = logical_error_probability(rounds[None, :], eps[:, None])
    residual = pred - probabilities[None, :]
    if shots is not None:
        shots = np.asarray(shots, dtype=float)
        if shots.shape == ():
            shots = np.full_like(rounds, float(shots))
        sigma2 = np.maximum(probabilities * (1 - probabilities) / shots, 1 / (4 * shots**2))
        loss = ((residual**2) / sigma2[None, :]).sum(axis=1)
    else:
        loss = (residual**2).sum(axis=1)
    best = int(np.argmin(loss))
    epsilon_l = float(eps[best])
    fitted = logical_error_probability(rounds, epsilon_l)
    return {
        "epsilon_l": epsilon_l,
        "rounds": rounds.tolist(),
        "probabilities": probabilities.tolist(),
        "fitted_probabilities": fitted.tolist(),
        "loss": float(loss[best]),
    }


def run_pipeline(
    distance=3,
    rounds=None,
    shots=10,
    basis="Z",
    cal=CAL,
    api_url=None,
    token=None,
    quantum_computer="emerald",
    use_timeslot=False,
    decoder_noise="exact",
    run_mode="hardware",
    refresh_calibration=True,
    save_stim_file=True,
    optimize_layout: bool | None = None,
    allow_skipped_hardware_gates: bool = True,
    synthetic_defect_qubits=None,
    synthetic_defect_couplers=None,
    synthetic_defect_probability: float = 0.5,
    fixed_mapping: dict[int, int] | None = None,
    unavailable_iqm_couplers=None,
):
    """Run the QEC pipeline on IQM hardware or synthetic samples.

    Args:
        decoder_noise: "exact" for per-qubit/per-coupler calibration noise, or
            "average" for the compact four-parameter calibration model.
        run_mode: "hardware" submits to IQM. "synthetic" samples measurements
            from the selected calibrated Stim noise model instead.
        optimize_layout: If None, defaults to True on hardware and False for synthetic.
        allow_skipped_hardware_gates: If True, unavailable qubits/couplers are
            warned about and skipped in hardware submission instead of raising.
        synthetic_defect_qubits: Optional Stim qubit indices to damage in
            synthetic mode by adding depolarizing noise after each TICK.
        synthetic_defect_couplers: Optional Stim couplers/edges to damage in
            synthetic mode by adding depolarizing noise after matching CZ gates.
        fixed_mapping: Optional explicit {Stim qubit: IQM QB label} placement.
        unavailable_iqm_couplers: Optional IQM couplers to skip in hardware mode,
            e.g. ["QB24_QB32"] or [(24, 32)].
    """
    if run_mode not in {"hardware", "synthetic"}:
        raise ValueError("run_mode must be 'hardware' or 'synthetic'")
    rounds = rounds or distance
    if optimize_layout is None:
        optimize_layout = run_mode == "hardware"
    if refresh_calibration:
        cal = refresh_calibration_if_needed(cal, api_url, token, quantum_computer)
    c = stim_circuit(distance, rounds, basis)
    stim_file = write_stim_file(c, distance, rounds, basis) if save_stim_file else None

    raw_measurements = None
    iqm_qc = job = result = counts = None
    backend_mapping = None
    transpile_diagnostics = None
    skipped_hardware_ops = []
    backend = None
    if run_mode == "hardware":
        with _explicit_token_overrides_env(token):
            provider = IQMProvider(api_url or os.environ.get("IQM_API_URL", "https://resonance.iqm.tech/"), quantum_computer=quantum_computer, token=token or os.environ["IQM_TOKEN"])
            backend = provider.get_backend()

    unavailable_iqm_couplers = unavailable_iqm_couplers or []
    unavailable_iqm_couplers = {
        tuple(sorted(map(int, re.findall(r"QB(\d+)", edge)))) if isinstance(edge, str)
        else tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in unavailable_iqm_couplers
    }

    if fixed_mapping is not None:
        mapping = {int(k): int(v) for k, v in fixed_mapping.items()}
        layout_diagnostics = {
            "mapping_source": "fixed_mapping",
            "score": None,
            "usable_twoq_gate_count": int(sum(twoq_edges(c).values())),
            "missing_twoq_gate_count": None,
            "min_used_cz_fidelity": None,
            "candidate_count": None,
            "exact_graph_error": None,
        }
    elif optimize_layout:
        mapping, layout_diagnostics = optimized_mapping(c, cal=cal, backend=backend, allow_skipped_gates=allow_skipped_hardware_gates)
    elif run_mode == "synthetic":
        mapping = {int(q): int(q) for q in c.get_final_qubit_coordinates()}
        layout_diagnostics = {
            "mapping_source": "synthetic_identity",
            "score": None,
            "usable_twoq_gate_count": int(sum(twoq_edges(c).values())),
            "missing_twoq_gate_count": 0,
            "min_used_cz_fidelity": None,
            "candidate_count": None,
            "exact_graph_error": None,
        }
    else:
        mapping = best_mapping(c, cal)
        layout_diagnostics = {
            "mapping_source": "exact_graph",
            "score": None,
            "usable_twoq_gate_count": int(sum(twoq_edges(c).values())),
            "missing_twoq_gate_count": 0,
            "min_used_cz_fidelity": None,
            "candidate_count": None,
            "exact_graph_error": None,
        }

    decoder_c, decoder_noise_model = calibrated_decoder_circuit(c, mapping, distance, rounds, basis, cal, mode=decoder_noise)
    synthetic_defect_qubits = [int(q) for q in (synthetic_defect_qubits or [])]
    synthetic_defect_couplers = [tuple(sorted((int(a), int(b)))) for a, b in (synthetic_defect_couplers or [])]
    if synthetic_defect_qubits or synthetic_defect_couplers:
        if run_mode != "synthetic":
            raise ValueError("synthetic defects are only supported with run_mode='synthetic'")
        decoder_c = add_synthetic_defect_noise(
            decoder_c,
            defect_qubits=synthetic_defect_qubits,
            defect_couplers=synthetic_defect_couplers,
            probability=synthetic_defect_probability,
        )
        decoder_noise_model = {
            "base": decoder_noise_model,
            "synthetic_defect_qubits": synthetic_defect_qubits,
            "synthetic_defect_couplers": synthetic_defect_couplers,
            "synthetic_defect_probability": synthetic_defect_probability,
        }

    if run_mode == "hardware":
        backend_mapping = iqm_label_mapping_to_backend_indices(mapping, backend)
        unavailable_backend_qubits = unavailable_backend_qubits_from_calibration(cal, backend)
        unavailable_backend_couplers = {
            tuple(sorted((backend.qubit_name_to_index(f"QB{a}"), backend.qubit_name_to_index(f"QB{b}"))))
            for a, b in unavailable_iqm_couplers
        }
        if allow_skipped_hardware_gates:
            qc, meas_order, skipped_hardware_ops = stim_to_qiskit_mapped_tolerant(
                c,
                backend_mapping,
                backend=backend,
                unavailable_backend_qubits=unavailable_backend_qubits,
                unavailable_backend_couplers=unavailable_backend_couplers,
            )
            if skipped_hardware_ops:
                warnings.warn("Full backend validation was skipped because impossible operations were intentionally omitted.")
            else:
                validate_backend_mapping(qc, c, backend_mapping, backend)
        else:
            qc, meas_order = stim_to_qiskit_mapped(c, backend_mapping)
            validate_backend_mapping(qc, c, backend_mapping, backend)
        iqm_qc = transpile(qc, backend=backend, optimization_level=1, initial_layout=identity_initial_layout(qc))
        transpile_diagnostics = {
            "ideal_depth": qc.depth(),
            "transpiled_depth": iqm_qc.depth(),
            "swap_count": iqm_qc.count_ops().get("swap", 0),
            "initial_layout": identity_initial_layout(qc),
            "backend_mapping": backend_mapping,
            "backend_qubit_labels": {stim_q: backend.index_to_qubit_name(idx) for stim_q, idx in backend_mapping.items()},
            "layout_diagnostics": layout_diagnostics,
            "skipped_hardware_operation_count": len(skipped_hardware_ops),
        }
        job = backend.run(iqm_qc, shots=shots, use_timeslot=use_timeslot)
        result = job.result()
        memory, counts = result_to_memory(result)
    else:
        qc, meas_order = stim_to_qiskit_mapped(c, mapping)
        raw_measurements = decoder_c.compile_sampler().sample(shots).astype(bool)
        memory = stim_measurements_to_qiskit_memory(raw_measurements)

    decoded = decode(c, decoder_c, memory)
    summary = {
        "run_mode": run_mode,
        "job_id": job.job_id() if job is not None else None,
        "mapping": mapping,
        "decoder_noise": decoder_noise,
        "noise_model": decoder_noise_model,
        "layout_diagnostics": layout_diagnostics,
        "synthetic_defect_qubits": synthetic_defect_qubits,
        "synthetic_defect_couplers": synthetic_defect_couplers,
        "synthetic_defect_probability": synthetic_defect_probability if synthetic_defect_qubits or synthetic_defect_couplers else None,
        "unavailable_iqm_couplers": sorted(unavailable_iqm_couplers),
        "swap_count": transpile_diagnostics.get("swap_count") if transpile_diagnostics else None,
        "skipped_hardware_operation_count": len(skipped_hardware_ops),
        "shots_decoded": len(memory),
        "corrected_ler": decoded["corrected_ler"],
        "uncorrected_ler": decoded["uncorrected_ler"],
        "correction_improved": decoded["corrected_ler"] < decoded["uncorrected_ler"],
    }
    return {
        "calibration_path": cal,
        "stim_file": stim_file,
        "stim_circuit": c,
        "decoder_circuit": decoder_c,
        "mapping": mapping,
        "backend_mapping": backend_mapping,
        "decoder_noise": decoder_noise,
        "noise_model": decoder_noise_model,
        "run_mode": run_mode,
        "qiskit_circuit": qc,
        "iqm_circuit": iqm_qc,
        "transpile_diagnostics": transpile_diagnostics,
        "measurement_order": meas_order,
        "layout_diagnostics": layout_diagnostics,
        "skipped_hardware_ops": skipped_hardware_ops,
        "job": job,
        "result": result,
        "raw_measurements": raw_measurements,
        "memory": memory,
        "counts": counts,
        "decoded": decoded,
        "summary": summary,
    }


def fit_round_sweep(results: list[dict]) -> dict:
    """Fit per-round logical error rate from several `run_pipeline` results."""
    rounds = np.array([r["summary"].get("rounds", r["stim_circuit"].num_ticks) for r in results], dtype=float)
    corrected = np.array([r["summary"]["corrected_ler"] for r in results], dtype=float)
    uncorrected = np.array([r["summary"]["uncorrected_ler"] for r in results], dtype=float)
    shots = np.array([r["summary"]["shots_decoded"] for r in results], dtype=float)

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
        "corrected_fit": fit_logical_error_rate(rounds, corrected_fit_prob, shots=shots),
        "uncorrected_fit": fit_logical_error_rate(rounds, uncorrected_fit_prob, shots=shots),
    }


def plot_logical_error_rate(fit: dict, save_path: str | None = None):
    """Plot logical error probability vs rounds on a readable linear scale."""
    import matplotlib.pyplot as plt

    rounds = fit["rounds"]
    xs = np.linspace(rounds.min(), rounds.max(), 300)
    eps_c = fit["corrected_fit"]["epsilon_l"]
    eps_u = fit["uncorrected_fit"]["epsilon_l"]
    y_c = fit["corrected_fit_prob"]
    y_u = fit["uncorrected_fit_prob"]
    fit_y_c = logical_error_probability(xs, eps_c)
    fit_y_u = logical_error_probability(xs, eps_u)

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
    ax.set_title(f"Logical error rate extraction\nMWPM eps_L={eps_c:.4g}, uncorrected eps_L={eps_u:.4g}", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def run_round_sweep(
    distance: int = 3,
    round_values=(1, 2, 3, 5, 8, 10),
    shots: int = 5_000,
    basis: str = "Z",
    cal: str = CAL,
    api_url=None,
    token=None,
    quantum_computer: str = "emerald",
    use_timeslot: bool = False,
    decoder_noise: str = "exact",
    run_mode: str = "hardware",
    refresh_calibration: bool = True,
    save_stim_file: bool = True,
    optimize_layout: bool | None = None,
    allow_skipped_hardware_gates: bool = True,
    synthetic_defect_qubits=None,
    synthetic_defect_couplers=None,
    synthetic_defect_probability: float = 0.5,
    fixed_mapping: dict[int, int] | None = None,
    unavailable_iqm_couplers=None,
    show_error_plot: bool = True,
) -> dict:
    """Run the normal IQM QEC pipeline over several rounds and fit logical error rate."""
    if refresh_calibration:
        cal = refresh_calibration_if_needed(cal, api_url, token, quantum_computer)
    synthetic_defect_qubits = [int(q) for q in (synthetic_defect_qubits or [])]
    synthetic_defect_couplers = [tuple(sorted((int(a), int(b)))) for a, b in (synthetic_defect_couplers or [])]

    results = []
    for rounds in round_values:
        out = run_pipeline(
            distance=distance,
            rounds=int(rounds),
            shots=shots,
            basis=basis,
            cal=cal,
            api_url=api_url,
            token=token,
            quantum_computer=quantum_computer,
            use_timeslot=use_timeslot,
            decoder_noise=decoder_noise,
            run_mode=run_mode,
            refresh_calibration=False,
            save_stim_file=save_stim_file,
            optimize_layout=optimize_layout,
            allow_skipped_hardware_gates=allow_skipped_hardware_gates,
            synthetic_defect_qubits=synthetic_defect_qubits,
            synthetic_defect_couplers=synthetic_defect_couplers,
            synthetic_defect_probability=synthetic_defect_probability,
            fixed_mapping=fixed_mapping,
            unavailable_iqm_couplers=unavailable_iqm_couplers,
        )
        out["summary"]["rounds"] = int(rounds)
        results.append(out)
        s = out["summary"]
        print(
            f"rounds={rounds} | corrected_prob={s['corrected_ler']:.5g} | "
            f"uncorrected_prob={s['uncorrected_ler']:.5g} | "
            f"swap_count={s['swap_count']} | skipped_ops={s['skipped_hardware_operation_count']} | job={s['job_id']}"
        )

    fit = fit_round_sweep(results)
    error_figure = plot_logical_error_rate(fit) if show_error_plot else None
    return {
        "results": results,
        "fit": fit,
        "error_figure": error_figure,
        "summary": {
            "distance": distance,
            "round_values": list(round_values),
            "shots": shots,
            "run_mode": run_mode,
            "decoder_noise": decoder_noise,
            "logical_error_rate": fit["corrected_fit"]["epsilon_l"],
            "uncorrected_logical_error_rate": fit["uncorrected_fit"]["epsilon_l"],
            "mapping": results[0]["summary"]["mapping"],
            "layout_diagnostics": results[0]["summary"]["layout_diagnostics"],
            "synthetic_defect_qubits": synthetic_defect_qubits,
            "synthetic_defect_couplers": synthetic_defect_couplers,
            "synthetic_defect_probability": synthetic_defect_probability if synthetic_defect_qubits or synthetic_defect_couplers else None,
            "unavailable_iqm_couplers": results[0]["summary"]["unavailable_iqm_couplers"],
            "swap_count": results[0]["summary"]["swap_count"],
            "skipped_hardware_operation_count": results[0]["summary"]["skipped_hardware_operation_count"],
        },
    }


if __name__ == "__main__":
    run_pipeline(distance=3, rounds=3, shots=10, run_mode="synthetic")
