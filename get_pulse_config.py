"""Extract pulse configuration from IQM Resonance hardware for pulse_control.py.

Provides two main entry points:

- ``build_pulse_config_from_calibration(cal_path)`` — builds a PulseConfig
  entirely from a local calibration-data JSON file (no network required).
  This is what ``iqm_qec_pipeline.optimized_run_pipeline`` calls
  automatically.

- ``fetch_hardware_pulse_config()`` — connects to IQM Resonance, pulls the
  live calibration stash *and* compiles a test circuit through Pulla to
  extract hardware topology.  Use this when you want the most up-to-date
  parameters.

When run as a script (``python get_pulse_config.py``), it executes the
full online flow and saves the result to YAML + JSON.

Usage (offline, from pipeline)::

    from get_pulse_config import build_pulse_config_from_calibration
    cfg, info = build_pulse_config_from_calibration("calibration_data/....json")

Usage (online)::

    from get_pulse_config import fetch_hardware_pulse_config
    cfg, info = fetch_hardware_pulse_config()
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from pulse_control import PulseConfig

__all__ = [
    "build_pulse_config_from_calibration",
    "fetch_hardware_pulse_config",
    "extract_coherence_times",
]

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
IQM_SERVER_URL = "https://resonance.meetiqm.com"
DEVICE = "emerald"
_CONFIG_YAML = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _median_or_default(values: list[float], default: float) -> float:
    return statistics.median(values) if values else default


def _collect_from_observations(
    observations: list[dict], pattern: str,
) -> list[float]:
    """Collect ``value`` from calibration observations whose ``dut_field`` matches *pattern*."""
    values: list[float] = []
    for obs in observations:
        if obs.get("invalid"):
            continue
        field = obs.get("dut_field", "")
        if re.search(pattern, field) and obs.get("value") is not None:
            values.append(float(obs["value"]))
    return values


def _collect_from_stash(stash, pattern: str) -> list[float]:
    """Collect values from a live ``PullaStash`` object."""
    values: list[float] = []
    for field, obs in stash.observations.items():
        if re.search(pattern, field) and obs.value is not None:
            values.append(float(obs.value))
    return values


def _load_hw_timing(config_path: Path = _CONFIG_YAML) -> tuple[float, float]:
    """Return ``(cz_gate_time_ns, meas_duration_ns)`` from local config.yaml."""
    if config_path.exists():
        with open(config_path) as f:
            hw = yaml.safe_load(f).get("hardware", {})
        return hw.get("gate_2q_ns", 40.0), hw.get("meas_ns", 1000.0)
    return 40.0, 1000.0


def _build_config(
    t1_ns: float,
    t2_ns: float,
    t2_echo_ns: float,
    median_cz_fidelity: float,
    median_sq_fidelity: float,
    median_ro_fidelity: float,
    median_err_0to1: float,
    median_err_1to0: float,
    cz_gate_time: float,
    meas_duration: float,
) -> PulseConfig:
    """Map extracted calibration metrics into a PulseConfig."""
    # CZ leakage target from 1 - fidelity (first-order approximation)
    cz_leakage = max(1e-5, 1.0 - median_cz_fidelity)

    # Readout kick amplitude: higher error_1_to_0 → stronger kick
    readout_kick_amplitude = min(0.95, 0.60 + 0.4 * median_err_1to0 / 0.03)

    # Optimised readout duration scaled by fidelity
    if median_ro_fidelity > 0.99:
        readout_duration = min(meas_duration, 200.0)
    elif median_ro_fidelity > 0.98:
        readout_duration = min(meas_duration, 300.0)
    else:
        readout_duration = min(meas_duration, 500.0)

    return PulseConfig(
        sample_rate_ghz=1.0,
        # CZ gate
        cz_gate_time_ns=cz_gate_time,
        cz_amplitude=0.45,
        cz_slepian_order=1,
        cz_adiabatic_ramp_fraction=0.2,
        cz_leakage_target=cz_leakage,
        cz_net_zero=True,
        cz_net_zero_gap_ns=2.0,
        # Readout
        readout_duration_ns=readout_duration,
        readout_frequency_ghz=7.0,
        readout_amplitude=0.30,
        readout_kick_amplitude=readout_kick_amplitude,
        readout_kick_duration_ns=30.0,
        readout_ring_up_time_ns=20.0,
        clear_amplitude=0.60,
        clear_duration_ns=40.0,
        clear_phase_shift_rad=math.pi,
        # Reset
        reset_pi_amplitude=0.50,
        reset_pi_duration_ns=20.0,
        reset_feedback_latency_ns=100.0,
        reset_pump_amplitudes=(0.35, 0.25),
        reset_pump_duration_ns=80.0,
        reset_pump_frequencies_offset_ghz=(0.0, -0.22),
        # QEC cycle
        inter_gate_buffer_ns=4.0,
    )


# ---------------------------------------------------------------------------
# Public API — offline (local calibration JSON)
# ---------------------------------------------------------------------------

def extract_coherence_times(
    cal_path: str | Path,
) -> dict[str, float]:
    """Return median T1, T2, T2echo (in ns) from a calibration JSON file."""
    observations = json.loads(Path(cal_path).read_text())["observations"]
    t1 = _collect_from_observations(observations, r"characterization\.model\.QB\d+\.t1_time")
    t2 = _collect_from_observations(observations, r"characterization\.model\.QB\d+\.t2_time")
    t2e = _collect_from_observations(observations, r"characterization\.model\.QB\d+\.t2_echo_time")
    return {
        "t1_ns": _median_or_default(t1, 50e-6) * 1e9,
        "t2_ns": _median_or_default(t2, 30e-6) * 1e9,
        "t2_echo_ns": _median_or_default(t2e, 40e-6) * 1e9,
    }


def build_pulse_config_from_calibration(
    cal_path: str | Path,
    config_yaml: str | Path = _CONFIG_YAML,
) -> tuple[PulseConfig, dict[str, Any]]:
    """Build a PulseConfig from a local IQM calibration JSON file.

    No network connection is required.  This is the function imported
    by ``iqm_qec_pipeline.optimized_run_pipeline``.

    Returns
    -------
    (pulse_config, calibration_info)
        ``pulse_config`` is ready for use with ``pulse_control.py``.
        ``calibration_info`` is a dict with coherence times, fidelities,
        and error rates extracted from the calibration data.
    """
    observations = json.loads(Path(cal_path).read_text())["observations"]
    cz_gate_time, meas_duration = _load_hw_timing(Path(config_yaml))

    # --- Coherence (seconds → ns) ---
    t1_vals = _collect_from_observations(observations, r"characterization\.model\.QB\d+\.t1_time")
    t2_vals = _collect_from_observations(observations, r"characterization\.model\.QB\d+\.t2_time")
    t2e_vals = _collect_from_observations(observations, r"characterization\.model\.QB\d+\.t2_echo_time")
    t1_ns = _median_or_default(t1_vals, 50e-6) * 1e9
    t2_ns = _median_or_default(t2_vals, 30e-6) * 1e9
    t2_echo_ns = _median_or_default(t2e_vals, 40e-6) * 1e9

    # --- Gate fidelities ---
    cz_fid = _collect_from_observations(observations, r"metrics\.irb\.cz\.crf_acstarkcrf\.QB.*\.fidelity")
    sq_fid = _collect_from_observations(observations, r"metrics\.rb\.prx\.drag_crf_sx\.QB.*\.fidelity")
    median_cz = _median_or_default(cz_fid, 0.995)
    median_sq = _median_or_default(sq_fid, 0.999)

    # --- Readout ---
    e01 = _collect_from_observations(observations, r"metrics\.ssro\.measure\.constant\.QB\d+\.error_0_to_1")
    e10 = _collect_from_observations(observations, r"metrics\.ssro\.measure\.constant\.QB\d+\.error_1_to_0")
    ro_fid = _collect_from_observations(observations, r"metrics\.ssro\.measure\.constant\.QB\d+\.fidelity")
    median_ro = _median_or_default(ro_fid, 0.985)
    median_e01 = _median_or_default(e01, 0.005)
    median_e10 = _median_or_default(e10, 0.015)

    cfg = _build_config(
        t1_ns, t2_ns, t2_echo_ns,
        median_cz, median_sq, median_ro, median_e01, median_e10,
        cz_gate_time, meas_duration,
    )

    info = {
        "t1_ns": t1_ns,
        "t2_ns": t2_ns,
        "t2_echo_ns": t2_echo_ns,
        "median_cz_fidelity": median_cz,
        "median_1q_fidelity": median_sq,
        "median_readout_fidelity": median_ro,
        "median_err_0_to_1": median_e01,
        "median_err_1_to_0": median_e10,
        "num_qubits_calibrated": len(t1_vals),
        "num_cz_pairs_calibrated": len(cz_fid),
    }
    return cfg, info


# ---------------------------------------------------------------------------
# Public API — online (live IQM Resonance connection)
# ---------------------------------------------------------------------------

def fetch_hardware_pulse_config(
    server_url: str = IQM_SERVER_URL,
    device: str = DEVICE,
    config_yaml: str | Path = _CONFIG_YAML,
) -> tuple[PulseConfig, dict[str, Any]]:
    """Connect to IQM Resonance and build a PulseConfig from live calibration.

    Requires ``IQM_TOKEN`` in the environment or prompts interactively.

    Returns
    -------
    (pulse_config, calibration_info)
    """
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from iqm.qiskit_iqm import IQMProvider
    from iqm.pulla.pulla import Pulla
    from iqm.pulla.utils_qiskit import qiskit_to_pulla

    if "IQM_TOKEN" not in os.environ:
        os.environ["IQM_TOKEN"] = input("IQM Resonance token: ")

    print("[1/4] Connecting to IQM Resonance...")
    provider = IQMProvider(server_url, quantum_computer=device)
    backend = provider.get_backend()
    p = Pulla(server_url, quantum_computer=device)

    print("[2/4] Fetching calibration stash...")
    stash = p.get_calibration_stash()

    # Coherence
    t1_vals = _collect_from_stash(stash, r"characterization\.model\.QB\d+\.t1_time")
    t2_vals = _collect_from_stash(stash, r"characterization\.model\.QB\d+\.t2_time")
    t2e_vals = _collect_from_stash(stash, r"characterization\.model\.QB\d+\.t2_echo_time")
    t1_ns = _median_or_default(t1_vals, 50e-6) * 1e9
    t2_ns = _median_or_default(t2_vals, 30e-6) * 1e9
    t2_echo_ns = _median_or_default(t2e_vals, 40e-6) * 1e9

    # Fidelities
    cz_fid = _collect_from_stash(stash, r"metrics\.irb\.cz\.crf_acstarkcrf\.QB.*\.fidelity")
    sq_fid = _collect_from_stash(stash, r"metrics\.rb\.prx\.drag_crf_sx\.QB.*\.fidelity")
    median_cz = _median_or_default(cz_fid, 0.995)
    median_sq = _median_or_default(sq_fid, 0.999)

    # Readout
    e01 = _collect_from_stash(stash, r"metrics\.ssro\.measure\.constant\.QB\d+\.error_0_to_1")
    e10 = _collect_from_stash(stash, r"metrics\.ssro\.measure\.constant\.QB\d+\.error_1_to_0")
    ro_fid = _collect_from_stash(stash, r"metrics\.ssro\.measure\.constant\.QB\d+\.fidelity")
    median_ro = _median_or_default(ro_fid, 0.985)
    median_e01 = _median_or_default(e01, 0.005)
    median_e10 = _median_or_default(e10, 0.015)

    # Compile test circuit for topology
    print("[3/4] Compiling test circuit for topology...")
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cz(0, 1)
    qc_t = transpile(qc, backend=backend, optimization_level=3)
    circuits, compiler = qiskit_to_pulla(p, backend, qc_t)
    playlist, context = compiler.compile(circuits)
    run_props = playlist.additional_run_properties

    print("[4/4] Building PulseConfig...")
    cz_gate_time, meas_duration = _load_hw_timing(Path(config_yaml))
    cfg = _build_config(
        t1_ns, t2_ns, t2_echo_ns,
        median_cz, median_sq, median_ro, median_e01, median_e10,
        cz_gate_time, meas_duration,
    )

    info = {
        "t1_ns": t1_ns,
        "t2_ns": t2_ns,
        "t2_echo_ns": t2_echo_ns,
        "median_cz_fidelity": median_cz,
        "median_1q_fidelity": median_sq,
        "median_readout_fidelity": median_ro,
        "median_err_0_to_1": median_e01,
        "median_err_1_to_0": median_e10,
        "num_qubits_calibrated": len(t1_vals),
        "num_cz_pairs_calibrated": len(cz_fid),
        "hardware_topology": {
            "qubits": run_props.get("qubits", []),
            "couplers": run_props.get("couplers", []),
            "readout_components": run_props.get("readout_components", []),
        },
    }
    return cfg, info


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_YAML = Path(__file__).parent / "hardware_pulse_config.yaml"
    OUTPUT_JSON = Path(__file__).parent / "hardware_pulse_config.json"

    hardware_config, cal_info = fetch_hardware_pulse_config()

    # Serialise and save
    cfg_dict = asdict(hardware_config)
    for k, v in cfg_dict.items():
        if isinstance(v, tuple):
            cfg_dict[k] = list(v)

    output = {
        "device": DEVICE,
        "server_url": IQM_SERVER_URL,
        "calibration_summary": {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in cal_info.items()
            if k != "hardware_topology"
        },
        "hardware_topology": cal_info.get("hardware_topology", {}),
        "pulse_config": cfg_dict,
    }
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    t1 = cal_info["t1_ns"]
    t2 = cal_info["t2_ns"]
    t2e = cal_info["t2_echo_ns"]
    print("\n" + "=" * 60)
    print("  IQM Emerald Pulse Configuration Summary")
    print("=" * 60)
    print(f"\n--- Coherence ---")
    print(f"  T1 (median):      {t1/1e3:.1f} µs")
    print(f"  T2 (median):      {t2/1e3:.1f} µs")
    print(f"  T2 echo (median): {t2e/1e3:.1f} µs")
    print(f"\n--- Gate Fidelities ---")
    print(f"  1Q gate (median): {cal_info['median_1q_fidelity']:.4f}")
    print(f"  CZ gate (median): {cal_info['median_cz_fidelity']:.4f}")
    print(f"\n--- Readout ---")
    print(f"  Fidelity (median): {cal_info['median_readout_fidelity']:.4f}")
    print(f"  Err 0→1 (median):  {cal_info['median_err_0_to_1']:.4f}")
    print(f"  Err 1→0 (median):  {cal_info['median_err_1_to_0']:.4f}")
    print(f"\n--- PulseConfig ---")
    print(f"  CZ gate time:     {hardware_config.cz_gate_time_ns} ns")
    print(f"  Readout duration: {hardware_config.readout_duration_ns} ns")
    print(f"  Readout kick amp: {hardware_config.readout_kick_amplitude:.3f}")
    print(f"\nSaved to: {OUTPUT_YAML}")
    print(f"          {OUTPUT_JSON}")
    print(f"\nUsage:")
    print(f"  from get_pulse_config import build_pulse_config_from_calibration")
    print(f"  cfg, info = build_pulse_config_from_calibration('calibration_data/...')")
