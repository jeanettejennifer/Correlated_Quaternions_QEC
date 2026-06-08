"""Pulse-level control optimizations for IQM QEC pipeline.

All waveforms are returned as NumPy arrays sampled at a configurable DAC rate
(default 1 GS/s).  The helpers are designed to be composed into IQM pulse
schedules or used analytically (e.g. for Stim noise-model calibration).

Usage with the pipeline
-----------------------
>>> from pulse_control import (
...     PulseConfig,
...     optimized_cz_pulse, net_zero_cz_pulse,
...     optimized_readout_pulse, clear_pulse,
...     conditional_reset_pulse, unconditional_reset_pulse,
...     qec_cycle_schedule,
... )
>>> cfg = PulseConfig()                       # all defaults
>>> cz = net_zero_cz_pulse(cfg)               # bipolar CZ waveform
>>> ro, clr = optimized_readout_pulse(cfg)     # readout + CLEAR
>>> rst = unconditional_reset_pulse(cfg)       # multi-state drain
>>> schedule = qec_cycle_schedule(cfg)         # full cycle bundle
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class PulseConfig:
    """Central knob-set for all pulse optimizations.

    All times are in **nanoseconds** and amplitudes in **arbitrary DAC units**
    (normalised so that 1.0 ≈ full-scale).
    """

    # DAC / waveform sampling
    sample_rate_ghz: float = 1.0          # GS/s  →  1 sample per ns

    cz_gate_time_ns: float = 40.0         # total flux-pulse duration
    cz_amplitude: float = 0.45            # peak flux-pulse amplitude
    cz_slepian_order: int = 1             # DPSS order (0 = most concentrated)
    cz_adiabatic_ramp_fraction: float = 0.2  # fraction of gate time for ramps
    cz_leakage_target: float = 1e-4       # target |02⟩ leakage probability
    cz_net_zero: bool = True              # apply bipolar net-zero correction
    cz_net_zero_gap_ns: float = 2.0       # dead-time between bipolar lobes

    readout_duration_ns: float = 300.0    # total readout window
    readout_frequency_ghz: float = 7.0    # resonator drive frequency
    readout_amplitude: float = 0.30       # steady-state drive amplitude
    readout_kick_amplitude: float = 0.80  # initial overdrive amplitude
    readout_kick_duration_ns: float = 30.0  # kick transient length
    readout_ring_up_time_ns: float = 20.0   # cavity ring-up time constant
    clear_amplitude: float = 0.60         # CLEAR pulse amplitude
    clear_duration_ns: float = 40.0       # CLEAR pulse length
    clear_phase_shift_rad: float = math.pi  # 180° phase flip for destructive interference

    reset_pi_amplitude: float = 0.50      # conditional π-pulse amplitude
    reset_pi_duration_ns: float = 20.0    # π-pulse length
    reset_feedback_latency_ns: float = 100.0  # FPGA discrimination + routing
    reset_pump_amplitudes: tuple[float, ...] = (0.35, 0.25)  # |1⟩, |2⟩ pump drives
    reset_pump_duration_ns: float = 80.0  # unconditional pump window
    reset_pump_frequencies_offset_ghz: tuple[float, ...] = (0.0, -0.22)  # Δf from qubit freq

    # --- QEC cycle ---
    inter_gate_buffer_ns: float = 4.0     # dead time between operations

    @property
    def dt_ns(self) -> float:
        """Time step per sample in nanoseconds."""
        return 1.0 / self.sample_rate_ghz

    def n_samples(self, duration_ns: float) -> int:
        """Number of DAC samples for a given duration."""
        return max(1, int(round(duration_ns * self.sample_rate_ghz)))


# ---------------------------------------------------------------------------
# 1.  PULSE OPTIMIZATION
# ---------------------------------------------------------------------------

def _slepian_window(n: int, order: int = 1, half_bandwidth: float = 4.0) -> np.ndarray:
    """Discrete prolate spheroidal sequence (Slepian) window.

    The Slepian window maximally concentrates spectral energy within a narrow
    band, producing an ultra-smooth envelope that suppresses non-adiabatic
    transitions at the |11⟩↔|02⟩ avoided crossing.

    Parameters
    ----------
    n : int
        Window length in samples.
    order : int
        DPSS order (0 = most concentrated, higher = wider sidelobes).
    half_bandwidth : float
        Time–half-bandwidth product NW.
    """
    try:
        from scipy.signal.windows import dpss
        return dpss(n, half_bandwidth, Kmax=order + 1)[order]
    except ImportError:
        # Fallback: Kaiser window approximation with β chosen to mimic
        # the Slepian mainlobe width.
        beta = math.pi * math.sqrt(
            max((2 * half_bandwidth / n) ** 2 - (order + 0.5) ** 2 / n ** 2, 0.1)
        )
        return np.kaiser(n, beta)


def _adiabatic_ramp(n_ramp: int) -> np.ndarray:
    """Smooth adiabatic ramp using a raised-cosine (Hann) half-window.

    Guarantees dΦ/dt → 0 at the boundaries, preventing non-adiabatic
    transitions that cause leakage.
    """
    if n_ramp <= 1:
        return np.ones(1)
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / (n_ramp - 1)))


def optimized_cz_pulse(cfg: PulseConfig | None = None) -> np.ndarray:
    """Generate an adiabatic Slepian-shaped CZ flux pulse.

    The pulse smoothly tunes the qubit into the |11⟩↔|02⟩ avoided crossing,
    accumulates exactly a π conditional phase, and ramps back out.  The
    Slepian envelope suppresses spectral leakage, keeping population out of
    the |02⟩ state.

    Returns
    -------
    waveform : ndarray, shape (n_samples,)
        Flux-pulse amplitude vs. time.
    """
    cfg = cfg or PulseConfig()
    n = cfg.n_samples(cfg.cz_gate_time_ns)
    n_ramp = max(1, int(round(n * cfg.cz_adiabatic_ramp_fraction)))

    # Core Slepian envelope
    envelope = _slepian_window(n, order=cfg.cz_slepian_order)
    envelope = envelope / np.max(np.abs(envelope))  # normalise to unity

    # Apply adiabatic ramps to leading / trailing edges
    ramp_up = _adiabatic_ramp(n_ramp)
    ramp_down = ramp_up[::-1]
    envelope[:n_ramp] *= ramp_up
    envelope[-n_ramp:] *= ramp_down

    return cfg.cz_amplitude * envelope


def net_zero_cz_pulse(cfg: PulseConfig | None = None) -> np.ndarray:
    """Generate a bipolar net-zero CZ flux pulse.

    A positive lobe executes the CZ interaction, then a symmetric negative
    lobe cancels any long-lived transient distortions on the flux line.
    The integrated flux is zero, preventing slow drift from corrupting
    subsequent operations.

    Returns
    -------
    waveform : ndarray, shape (2 * n_gate + n_gap,)
        Bipolar flux-pulse waveform.
    """
    cfg = cfg or PulseConfig()
    positive_lobe = optimized_cz_pulse(cfg)

    if not cfg.cz_net_zero:
        return positive_lobe

    n_gap = cfg.n_samples(cfg.cz_net_zero_gap_ns)
    gap = np.zeros(n_gap)
    negative_lobe = -positive_lobe[::-1]

    # Fine-tune negative lobe amplitude so ∫Φ dt = 0 exactly
    area_pos = np.sum(positive_lobe)
    area_neg = np.sum(negative_lobe)
    if abs(area_neg) > 1e-12:
        negative_lobe *= -area_pos / area_neg

    return np.concatenate([positive_lobe, gap, negative_lobe])


def cz_pulse_diagnostics(waveform: np.ndarray, cfg: PulseConfig | None = None) -> dict:
    """Compute quality metrics for a CZ flux pulse.

    Returns
    -------
    dict with keys:
        net_flux          – integrated area (should be ≈ 0 for net-zero)
        peak_amplitude    – maximum absolute amplitude
        bandwidth_ghz     – 3 dB spectral bandwidth
        rise_time_ns      – 10 %–90 % rise time
        estimated_leakage – first-order leakage estimate from spectral tails
    """
    cfg = cfg or PulseConfig()
    dt = cfg.dt_ns
    n = len(waveform)

    net_flux = float(np.sum(waveform) * dt)
    peak = float(np.max(np.abs(waveform)))

    # Spectral analysis
    spectrum = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(n, d=dt)  # GHz
    peak_spec = np.max(spectrum)
    bw_mask = spectrum >= peak_spec / math.sqrt(2)
    bandwidth = float(freqs[bw_mask][-1] - freqs[bw_mask][0]) if bw_mask.any() else 0.0

    # Rise time (10%–90% of peak)
    abs_wf = np.abs(waveform)
    thresh_lo, thresh_hi = 0.1 * peak, 0.9 * peak
    idx_lo = np.searchsorted(abs_wf, thresh_lo)
    idx_hi = np.searchsorted(abs_wf, thresh_hi)
    rise_time = float((idx_hi - idx_lo) * dt)

    # First-order leakage estimate: energy outside the adiabatic bandwidth
    # approximated as spectral weight beyond 1/(2·gate_time).
    f_adiabatic = 0.5 / cfg.cz_gate_time_ns  # GHz
    leakage_energy = float(np.sum(spectrum[freqs > f_adiabatic] ** 2))
    total_energy = float(np.sum(spectrum ** 2))
    estimated_leakage = leakage_energy / total_energy if total_energy > 0 else 0.0

    return {
        "net_flux": net_flux,
        "peak_amplitude": peak,
        "bandwidth_ghz": bandwidth,
        "rise_time_ns": rise_time,
        "estimated_leakage": estimated_leakage,
    }


# ---------------------------------------------------------------------------
# 2.  READOUT
# ---------------------------------------------------------------------------

def _readout_kick_envelope(cfg: PulseConfig) -> np.ndarray:
    """Build the readout drive envelope with an initial high-amplitude kick.

    The kick forces photons into the resonator almost instantaneously,
    reducing the effective measurement latency.  After the transient spike
    the drive settles to the steady-state amplitude for integration.
    """
    n_total = cfg.n_samples(cfg.readout_duration_ns)
    n_kick = cfg.n_samples(cfg.readout_kick_duration_ns)
    n_ringup = cfg.n_samples(cfg.readout_ring_up_time_ns)

    envelope = np.full(n_total, cfg.readout_amplitude)

    # Overdrive spike at the start
    envelope[:n_kick] = cfg.readout_kick_amplitude

    # Smooth exponential transition from kick to steady-state
    if n_ringup > 0 and n_kick < n_total:
        n_trans = min(n_ringup, n_total - n_kick)
        t_trans = np.arange(n_trans) * cfg.dt_ns
        tau = cfg.readout_ring_up_time_ns / 3.0  # decay constant
        overshoot = cfg.readout_kick_amplitude - cfg.readout_amplitude
        envelope[n_kick:n_kick + n_trans] = (
            cfg.readout_amplitude + overshoot * np.exp(-t_trans / tau)
        )

    return envelope


def optimized_readout_pulse(
    cfg: PulseConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an optimized readout drive and a CLEAR cavity-depletion pulse.

    Returns
    -------
    readout_I : ndarray
        In-phase readout drive envelope (apply at ``readout_frequency_ghz``).
    clear_I : ndarray
        CLEAR pulse envelope to append immediately after the readout window.
        Its 180° phase shift creates destructive interference that rapidly
        empties the resonator.
    """
    cfg = cfg or PulseConfig()

    # Readout drive with kick overdrive
    readout_I = _readout_kick_envelope(cfg)

    # CLEAR pulse: phase-flipped drive to dump residual photons
    clear_I = clear_pulse(cfg)

    return readout_I, clear_I


def clear_pulse(cfg: PulseConfig | None = None) -> np.ndarray:
    """Generate a CLEAR (cavity-clearing) pulse.

    A short burst at the readout frequency but 180° out of phase, creating
    destructive interference that rapidly depletes the resonator of residual
    photons.  This prevents AC-Stark-shift–induced dephasing on data qubits
    in the next QEC cycle step.

    The envelope uses a smooth Tukey window to avoid ringing.
    """
    cfg = cfg or PulseConfig()
    n = cfg.n_samples(cfg.clear_duration_ns)
    t = np.arange(n) * cfg.dt_ns

    # Tukey (tapered cosine) window — 30 % taper
    alpha = 0.3
    window = np.ones(n)
    n_taper = int(alpha * n / 2)
    if n_taper > 0:
        taper = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        window[:n_taper] = taper
        window[-n_taper:] = taper[::-1]

    # Phase-flip: multiply by cos(π) = −1  (relative to readout phase)
    phase = np.cos(cfg.clear_phase_shift_rad)  # = −1.0
    return cfg.clear_amplitude * window * phase


def readout_pulse_diagnostics(
    readout_I: np.ndarray,
    clear_I: np.ndarray,
    cfg: PulseConfig | None = None,
) -> dict:
    """Compute quality metrics for the readout + CLEAR waveforms.

    Returns
    -------
    dict with keys:
        readout_duration_ns   – total readout window
        kick_peak             – overdrive amplitude
        clear_duration_ns     – CLEAR pulse length
        residual_photon_frac  – estimated fraction of photons remaining after CLEAR
        total_latency_ns      – readout + CLEAR combined
    """
    cfg = cfg or PulseConfig()
    dt = cfg.dt_ns
    ro_dur = len(readout_I) * dt
    clr_dur = len(clear_I) * dt

    # Residual photon estimate: ratio of CLEAR-cancelled energy to readout energy
    ro_energy = np.sum(readout_I ** 2)
    clr_energy = np.sum(clear_I ** 2)
    # The CLEAR pulse should dump ~(clr_energy / ro_energy) of the cavity
    residual = max(0.0, 1.0 - clr_energy / ro_energy) if ro_energy > 0 else 1.0

    return {
        "readout_duration_ns": ro_dur,
        "kick_peak": float(np.max(np.abs(readout_I))),
        "clear_duration_ns": clr_dur,
        "residual_photon_frac": float(residual),
        "total_latency_ns": ro_dur + clr_dur,
    }


# ---------------------------------------------------------------------------
# 3.  ACTIVE RESET
# ---------------------------------------------------------------------------

def conditional_reset_pulse(cfg: PulseConfig | None = None) -> dict:
    """Generate a conditional (feedback-based) reset π-pulse.

    The FPGA discriminates the ancilla state after readout.  If |1⟩ is
    detected, a calibrated DRAG-style π-pulse flips it to |0⟩.  The
    returned dictionary contains both the pulse waveform and timing
    metadata for the feedback controller.

    Returns
    -------
    dict with keys:
        pi_I, pi_Q    – in-phase and quadrature π-pulse envelopes (DRAG)
        delay_ns       – total latency before the pulse fires
        total_ns       – delay + pulse duration
    """
    cfg = cfg or PulseConfig()
    n = cfg.n_samples(cfg.reset_pi_duration_ns)
    t = np.arange(n) * cfg.dt_ns
    sigma = cfg.reset_pi_duration_ns / 4.0  # Gaussian width
    center = cfg.reset_pi_duration_ns / 2.0

    # Gaussian π-pulse (I channel)
    gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    gauss -= gauss[0]  # zero baseline
    gauss /= np.max(gauss)
    pi_I = cfg.reset_pi_amplitude * gauss

    # DRAG correction on Q channel to suppress leakage to |2⟩
    # Q(t) ∝ -dI/dt / anharmonicity;  we use a normalised derivative.
    drag_coeff = -0.5  # empirical DRAG scale (hardware-dependent)
    deriv = np.gradient(gauss, cfg.dt_ns)
    deriv /= np.max(np.abs(deriv)) if np.max(np.abs(deriv)) > 0 else 1.0
    pi_Q = cfg.reset_pi_amplitude * drag_coeff * deriv

    return {
        "pi_I": pi_I,
        "pi_Q": pi_Q,
        "delay_ns": cfg.reset_feedback_latency_ns,
        "total_ns": cfg.reset_feedback_latency_ns + cfg.reset_pi_duration_ns,
    }


def unconditional_reset_pulse(cfg: PulseConfig | None = None) -> np.ndarray:
    """Generate an unconditional multi-state reset (pump) waveform.

    This bypasses FPGA feedback entirely.  A multi-frequency drive
    simultaneously couples |1⟩ and leaked |2⟩ populations to the fast-
    decaying readout resonator, forcing the qubit to |0⟩ unconditionally.

    The composite waveform is the sum of individual pump tones, each shaped
    with a smooth Blackman window to prevent spectral splatter.

    Returns
    -------
    waveform : ndarray, shape (n_samples,)
        Real-valued baseband pump envelope (to be up-converted by hardware).
    """
    cfg = cfg or PulseConfig()
    n = cfg.n_samples(cfg.reset_pump_duration_ns)
    t = np.arange(n) * cfg.dt_ns

    # Blackman window for spectral containment
    window = np.blackman(n)

    waveform = np.zeros(n)
    for amp, df_ghz in zip(cfg.reset_pump_amplitudes, cfg.reset_pump_frequencies_offset_ghz):
        # Each tone: amplitude × window × cos(2π·Δf·t)
        tone = amp * window * np.cos(2.0 * np.pi * df_ghz * t)
        waveform += tone

    return waveform


def reset_pulse_diagnostics(cfg: PulseConfig | None = None) -> dict:
    """Compare conditional vs. unconditional reset strategies.

    Returns
    -------
    dict with keys:
        conditional_latency_ns    – total time for feedback reset
        unconditional_latency_ns  – total time for pump reset
        latency_saving_ns         – time saved by unconditional approach
        unconditional_bandwidth_ghz – spectral width of pump waveform
    """
    cfg = cfg or PulseConfig()
    cond = conditional_reset_pulse(cfg)

    pump = unconditional_reset_pulse(cfg)
    n = len(pump)
    spectrum = np.abs(np.fft.rfft(pump))
    freqs = np.fft.rfftfreq(n, d=cfg.dt_ns)
    peak_spec = np.max(spectrum)
    bw_mask = spectrum >= peak_spec / math.sqrt(2)
    bw = float(freqs[bw_mask][-1] - freqs[bw_mask][0]) if bw_mask.any() else 0.0

    cond_lat = cond["total_ns"]
    uncond_lat = cfg.reset_pump_duration_ns

    return {
        "conditional_latency_ns": cond_lat,
        "unconditional_latency_ns": uncond_lat,
        "latency_saving_ns": cond_lat - uncond_lat,
        "unconditional_bandwidth_ghz": bw,
    }


# ---------------------------------------------------------------------------
# 4.  QEC CYCLE SCHEDULE
# ---------------------------------------------------------------------------

@dataclass
class QECCycleSchedule:
    """Bundled pulse waveforms and timing for one complete QEC syndrome cycle.

    Attributes
    ----------
    cz_waveform : ndarray
        Optimized CZ flux pulse (net-zero if configured).
    readout_waveform : ndarray
        Readout drive envelope with kick overdrive.
    clear_waveform : ndarray
        CLEAR cavity-depletion pulse.
    reset_waveform : ndarray
        Active reset waveform (conditional π or unconditional pump).
    reset_mode : str
        ``"conditional"`` or ``"unconditional"``.
    timeline_ns : dict
        Start times of each operation within the cycle.
    total_cycle_ns : float
        Total cycle duration in nanoseconds.
    diagnostics : dict
        Merged diagnostics from all sub-pulses.
    """
    cz_waveform: np.ndarray
    readout_waveform: np.ndarray
    clear_waveform: np.ndarray
    reset_waveform: np.ndarray
    reset_mode: str
    timeline_ns: dict
    total_cycle_ns: float
    diagnostics: dict


def qec_cycle_schedule(
    cfg: PulseConfig | None = None,
    reset_mode: Literal["conditional", "unconditional"] = "unconditional",
) -> QECCycleSchedule:
    """Assemble a full QEC syndrome-extraction cycle from optimized pulses.

    The cycle order is:  CZ gates → readout → CLEAR → active reset.
    Buffer times are inserted between operations.

    Parameters
    ----------
    cfg : PulseConfig, optional
        Pulse parameters (defaults used if omitted).
    reset_mode : str
        ``"conditional"`` for FPGA-feedback π-pulse,
        ``"unconditional"`` for multi-state pump reset.

    Returns
    -------
    QECCycleSchedule
        Complete cycle with waveforms, timing, and diagnostics.
    """
    cfg = cfg or PulseConfig()
    buf = cfg.inter_gate_buffer_ns

    # --- Build waveforms ---
    cz = net_zero_cz_pulse(cfg) if cfg.cz_net_zero else optimized_cz_pulse(cfg)
    ro, clr = optimized_readout_pulse(cfg)

    if reset_mode == "conditional":
        rst_info = conditional_reset_pulse(cfg)
        rst = rst_info["pi_I"]  # use I-channel envelope
        rst_duration = rst_info["total_ns"]
    else:
        rst = unconditional_reset_pulse(cfg)
        rst_duration = cfg.reset_pump_duration_ns

    # --- Timeline ---
    t_cz = 0.0
    t_ro = t_cz + len(cz) * cfg.dt_ns + buf
    t_clr = t_ro + len(ro) * cfg.dt_ns
    t_rst = t_clr + len(clr) * cfg.dt_ns + buf
    t_end = t_rst + rst_duration + buf

    timeline = {
        "cz_start_ns": t_cz,
        "readout_start_ns": t_ro,
        "clear_start_ns": t_clr,
        "reset_start_ns": t_rst,
        "cycle_end_ns": t_end,
    }

    # --- Diagnostics ---
    diag = {}
    diag["cz"] = cz_pulse_diagnostics(cz, cfg)
    diag["readout"] = readout_pulse_diagnostics(ro, clr, cfg)
    diag["reset"] = reset_pulse_diagnostics(cfg)
    diag["total_cycle_ns"] = t_end

    return QECCycleSchedule(
        cz_waveform=cz,
        readout_waveform=ro,
        clear_waveform=clr,
        reset_waveform=rst,
        reset_mode=reset_mode,
        timeline_ns=timeline,
        total_cycle_ns=t_end,
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# 5.  NOISE MODEL INTEGRATION (bridge to iqm_qec_pipeline)
# ---------------------------------------------------------------------------

def pulse_aware_noise_params(
    cfg: PulseConfig | None = None,
    t1_ns: float = 30_000.0,
    t2_ns: float = 20_000.0,
) -> dict:
    """Derive Stim-compatible noise parameters from pulse-level timing.

    Maps the physical pulse durations onto the four noise channels used by
    ``iqm_qec_pipeline.stim_circuit()``:
      - **gate** : depolarisation per CZ gate, estimated from gate time / T₂.
      - **idle** : depolarisation during idle periods (buffer gaps).
      - **meas** : measurement bit-flip probability from readout SNR model.
      - **reset**: residual |1⟩ population after active reset.

    Parameters
    ----------
    cfg : PulseConfig
    t1_ns, t2_ns : float
        Qubit coherence times (energy relaxation and dephasing).

    Returns
    -------
    dict compatible with ``stim_circuit(..., noise=result)``.
    """
    cfg = cfg or PulseConfig()

    schedule = qec_cycle_schedule(cfg)
    cz_time = len(schedule.cz_waveform) * cfg.dt_ns
    idle_time = cfg.inter_gate_buffer_ns
    readout_time = schedule.diagnostics["readout"]["total_latency_ns"]

    # Gate error ≈ gate_time / T₂  (first-order dephasing during gate)
    p_gate = min(cz_time / t2_ns, 0.5)

    # Idle error ≈ buffer_time / T₂
    p_idle = min(idle_time / t2_ns, 0.5)

    # Measurement error: rough model — readout_time / T₁ gives decay during
    # measurement; combined with residual photon dephasing.
    residual_photon = schedule.diagnostics["readout"]["residual_photon_frac"]
    p_meas = min(readout_time / (2 * t1_ns) + 0.01 * residual_photon, 0.5)

    # Reset error: conditional reset has feedback fidelity ~99.5%;
    # unconditional pump reaches ~99.9%.
    if schedule.reset_mode == "unconditional":
        p_reset = max(1e-3, cfg.reset_pump_duration_ns / t1_ns)
    else:
        p_reset = max(5e-3, cfg.reset_feedback_latency_ns / t1_ns)

    return {
        "gate": max(p_gate, 1e-6),
        "idle": max(p_idle, 1e-6),
        "meas": max(p_meas, 1e-6),
        "reset": max(p_reset, 1e-6),
    }


# ---------------------------------------------------------------------------
# 6.  VISUALIZATION
# ---------------------------------------------------------------------------

def plot_qec_cycle(
    schedule: QECCycleSchedule | None = None,
    cfg: PulseConfig | None = None,
    save_path: str | None = "qec_cycle_pulses.png",
):
    """Plot all pulse waveforms in a single QEC cycle.

    Four vertically stacked panels: CZ gate, readout, CLEAR, and reset.
    """
    import matplotlib.pyplot as plt

    if schedule is None:
        schedule = qec_cycle_schedule(cfg)
    cfg = cfg or PulseConfig()
    dt = cfg.dt_ns

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Optimized QEC Cycle Pulses", fontsize=14, y=0.98)

    # CZ
    ax = axes[0]
    t_cz = np.arange(len(schedule.cz_waveform)) * dt
    ax.plot(t_cz, schedule.cz_waveform, color="#2563eb", linewidth=1.5)
    ax.fill_between(t_cz, schedule.cz_waveform, alpha=0.15, color="#2563eb")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Flux amplitude")
    nz_label = " (net-zero)" if cfg.cz_net_zero else ""
    ax.set_title(f"CZ Gate Pulse{nz_label}", fontsize=11)
    ax.set_xlabel("Time (ns)")

    # Readout
    ax = axes[1]
    t_ro = np.arange(len(schedule.readout_waveform)) * dt
    ax.plot(t_ro, schedule.readout_waveform, color="#16a34a", linewidth=1.5)
    ax.fill_between(t_ro, schedule.readout_waveform, alpha=0.12, color="#16a34a")
    ax.axhline(cfg.readout_amplitude, color="#16a34a", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Drive amplitude")
    ax.set_title("Readout Drive (with kick overdrive)", fontsize=11)
    ax.set_xlabel("Time (ns)")

    # CLEAR
    ax = axes[2]
    t_clr = np.arange(len(schedule.clear_waveform)) * dt
    ax.plot(t_clr, schedule.clear_waveform, color="#dc2626", linewidth=1.5)
    ax.fill_between(t_clr, schedule.clear_waveform, alpha=0.15, color="#dc2626")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Drive amplitude")
    ax.set_title("CLEAR Cavity-Depletion Pulse (180° phase)", fontsize=11)
    ax.set_xlabel("Time (ns)")

    # Reset
    ax = axes[3]
    t_rst = np.arange(len(schedule.reset_waveform)) * dt
    ax.plot(t_rst, schedule.reset_waveform, color="#9333ea", linewidth=1.5)
    ax.fill_between(t_rst, schedule.reset_waveform, alpha=0.12, color="#9333ea")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Drive amplitude")
    mode_label = schedule.reset_mode.capitalize()
    ax.set_title(f"Active Reset — {mode_label}", fontsize=11)
    ax.set_xlabel("Time (ns)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved QEC cycle plot to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 7.  CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PulseConfig()
    schedule = qec_cycle_schedule(cfg, reset_mode="unconditional")

    print("=== QEC Cycle Pulse Schedule ===")
    print(f"Total cycle time: {schedule.total_cycle_ns:.1f} ns")
    print()

    for section, diag in schedule.diagnostics.items():
        if isinstance(diag, dict):
            print(f"[{section}]")
            for k, v in diag.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
            print()

    print("Timeline:")
    for k, v in schedule.timeline_ns.items():
        print(f"  {k}: {v:.1f} ns")

    noise = pulse_aware_noise_params(cfg)
    print("\nStim-compatible noise parameters:")
    for k, v in noise.items():
        print(f"  {k}: {v:.6e}")

    # Generate plot if matplotlib is available
    try:
        plot_qec_cycle(schedule, cfg)
    except ImportError:
        print("\nmatplotlib not available — skipping plot.")
