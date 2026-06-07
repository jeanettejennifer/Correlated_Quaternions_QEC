from __future__ import annotations

import math
import os
import base64
from io import BytesIO
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from iqm_qec_pipeline import COORD_TO_QB, IQM_EMERALD_POS, QB_TO_COORD
from snl_iqm_pipeline import PauliT, Pos, SuperStabilizer


TWO_QUBIT_GATES = {"CX", "CNOT", "CZ"}
ONE_QUBIT_GATES = {"R", "RZ", "H", "M", "MZ", "MR", "X", "Y", "Z", "S", "S_DAG", "I"}


def _xy(qb: int) -> tuple[float, float]:
    x, y = IQM_EMERALD_POS[int(qb)]
    return float(x), -float(y)


def _emerald_edges() -> set[tuple[int, int]]:
    """Nearest-neighbour couplers from the Emerald coordinate map."""
    edges = set()
    for qa, pa in QB_TO_COORD.items():
        for qb, pb in QB_TO_COORD.items():
            if qa >= qb:
                continue
            dx, dy = abs(pa.x - pb.x), abs(pa.y - pb.y)
            if (dx, dy) in {(1, 1), (0, 2)}:
                x0, y0 = _xy(qa)
                x1, y1 = _xy(qb)
                if math.hypot(x1 - x0, y1 - y0) < 1.7:
                    edges.add(tuple(sorted((qa, qb))))
    return edges


def _result_mapping(result: dict) -> dict[int, int]:
    """Return {stim_qubit: IQM_QB_label}; works for normal and SnL results."""
    if result.get("mapping"):
        return {int(k): int(v) for k, v in result["mapping"].items()}

    patch = result.get("damaged_patch") or result.get("clean_patch")
    if patch is not None:
        positions = set(patch.data_qubits) | set(patch.ancilla_qubits)
        return {int(p): int(COORD_TO_QB[p]) for p in positions if p in COORD_TO_QB}

    raise ValueError("Could not infer IQM mapping from result. Expected `mapping` or SnL patch data.")


def used_iqm_qubits(result: dict) -> set[int]:
    return set(_result_mapping(result).values())


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotone chain hull for the black used-region outline."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _padded_outline(points: list[tuple[float, float]], padding: float = 0.45) -> list[tuple[float, float]]:
    hull = _convex_hull(points)
    if len(hull) < 3:
        return hull
    cx = float(np.mean([p[0] for p in hull]))
    cy = float(np.mean([p[1] for p in hull]))
    out = []
    for x, y in hull:
        vx, vy = x - cx, y - cy
        norm = math.hypot(vx, vy) or 1.0
        out.append((x + padding * vx / norm, y + padding * vy / norm))
    return out


def _parse_iqm_edge(edge) -> tuple[int, int]:
    if isinstance(edge, str):
        import re

        qs = [int(q) for q in re.findall(r"QB(\d+)", edge)]
        if len(qs) != 2:
            raise ValueError(f"Could not parse IQM coupler label {edge!r}")
        return tuple(sorted(qs))
    a, b = edge
    return tuple(sorted((int(a), int(b))))


def _result_defect_qubits(result: dict) -> set[int]:
    return {int(q) for q in result.get("summary", {}).get("defect_qubits", result.get("defect_qubits", []))}


def _result_defect_couplers(result: dict) -> set[tuple[int, int]]:
    edges = result.get("summary", {}).get("defect_couplers", result.get("defect_couplers", []))
    return {_parse_iqm_edge(edge) for edge in edges}


def used_iqm_couplers(result: dict) -> set[tuple[int, int]]:
    """Return IQM couplers used by the mapped code/stabilizer circuit."""
    patch = result.get("damaged_patch") or result.get("clean_patch")
    if patch is not None:
        edges = set()
        for stabilizer in patch.stabilizers:
            for ancilla in stabilizer.ancilla:
                for data in stabilizer.data_qubits:
                    qa = COORD_TO_QB.get(ancilla)
                    qd = COORD_TO_QB.get(data)
                    if qa is not None and qd is not None:
                        edges.add(tuple(sorted((int(qa), int(qd)))))
        return edges

    circuit = result.get("stim_circuit")
    mapping = _result_mapping(result)
    if circuit is None:
        return set()
    edges = set()
    for ins in circuit.flattened():
        if ins.name in TWO_QUBIT_GATES:
            targets = ins.targets_copy()
            for i in range(0, len(targets), 2):
                a, b = int(targets[i].value), int(targets[i + 1].value)
                if a in mapping and b in mapping:
                    edges.add(tuple(sorted((mapping[a], mapping[b]))))
    return edges


def _set_relevant_region(ax, qbs: set[int], pad: float = 1.2):
    if not qbs:
        return
    xs, ys = zip(*[_xy(qb) for qb in qbs if qb in IQM_EMERALD_POS])
    if not xs:
        return
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def draw_emerald_base(
    ax,
    used_qbs: set[int] | None = None,
    label_qubits: bool = True,
    used_couplers: set[tuple[int, int]] | None = None,
    defect_qbs: set[int] | None = None,
    defect_couplers: set[tuple[int, int]] | None = None,
    highlight_used_neighbor_couplers: bool = True,
):
    used_qbs = used_qbs or set()
    used_couplers = used_couplers or set()
    defect_qbs = defect_qbs or set()
    defect_couplers = defect_couplers or set()
    for qa, qb in sorted(_emerald_edges()):
        x0, y0 = _xy(qa)
        x1, y1 = _xy(qb)
        edge = tuple(sorted((qa, qb)))
        active = edge in used_couplers or (
            highlight_used_neighbor_couplers and qa in used_qbs and qb in used_qbs and not used_couplers
        )
        bad = edge in defect_couplers
        ax.plot(
            [x0, x1], [y0, y1],
            color="#dc2626" if bad else "#16a34a" if active else "#d9e2ec",
            lw=3.2 if bad else 1.9 if active else 0.9,
            linestyle=(0, (5, 3)) if bad else "solid",
            alpha=0.95 if bad or active else 0.75,
            zorder=5 if bad else 0,
        )
        ax.scatter([(x0 + x1) / 2], [(y0 + y1) / 2], marker="D", s=180 if active else 120,
                   c="#ef4444" if bad else "#16a34a" if active else "#b7e38f",
                   edgecolors="white", linewidths=0.8, alpha=0.95 if bad or active else 0.65,
                   zorder=6 if bad else 1)
        if bad:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, "x", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold", zorder=7)

    for qb, (x_raw, y_raw) in sorted(IQM_EMERALD_POS.items()):
        x, y = float(x_raw), -float(y_raw)
        active = qb in used_qbs
        bad = qb in defect_qbs
        ax.scatter(
            [x], [y],
            s=560 if active or bad else 360,
            c="#4c1d95" if bad else "#5fcf55" if active else "#f8fafc",
            edgecolors="white" if active or bad else "#cbd5e1",
            linewidths=2,
            zorder=8 if bad else 3,
        )
        if label_qubits or active or bad:
            ax.text(x, y, f"QB{qb}", ha="center", va="center", fontsize=7.5,
                    color="white" if active or bad else "#94a3b8", zorder=9 if bad else 4)


def plot_used_emerald_patch(
    result: dict,
    title: str = "Used IQM Emerald patch",
    save_path: str | None = None,
    label_qubits: bool = True,
):
    """Show the full Emerald lattice and outline the used code patch in black."""
    used = used_iqm_qubits(result)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.set_aspect("equal")
    ax.axis("off")
    draw_emerald_base(ax, used, label_qubits=label_qubits)

    outline = _padded_outline([_xy(qb) for qb in used])
    if len(outline) >= 3:
        xs, ys = zip(*(outline + [outline[0]]))
        ax.plot(xs, ys, color="black", lw=2.8, zorder=6)

    summary = result.get("summary", {})
    subtitle = []
    if "layout_diagnostics" in summary:
        diag = summary["layout_diagnostics"]
        subtitle.append(f"layout={diag.get('mapping_source')}")
        subtitle.append(f"missing gates={diag.get('missing_twoq_gate_count')}")
    if summary.get("defect_qubits") or summary.get("defect_couplers"):
        subtitle.append(f"defects q={summary.get('defect_qubits', [])}, c={summary.get('defect_couplers', [])}")
    ax.set_title(title + ("\n" + " | ".join(subtitle) if subtitle else ""), fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_snl_emerald_hardware_region(
    result: dict,
    title: str = "SnL hardware placement on IQM Emerald",
    show_full_lattice: bool = False,
    label_qubits: bool = True,
    save_path: str | None = None,
):
    """Show the actual Emerald region used by an SnL run and its defective coupler(s)."""
    used_qbs = used_iqm_qubits(result)
    used_edges = used_iqm_couplers(result)
    defect_qbs = _result_defect_qubits(result)
    defect_edges = _result_defect_couplers(result)

    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    ax.set_aspect("equal")
    ax.axis("off")
    draw_emerald_base(
        ax,
        used_qbs=used_qbs,
        label_qubits=label_qubits,
        used_couplers=used_edges,
        defect_qbs=defect_qbs,
        defect_couplers=defect_edges,
    )

    outline_qbs = set(used_qbs) | set(defect_qbs)
    for a, b in defect_edges:
        outline_qbs.update([a, b])
    outline = _padded_outline([_xy(qb) for qb in outline_qbs if qb in IQM_EMERALD_POS])
    if len(outline) >= 3:
        xs, ys = zip(*(outline + [outline[0]]))
        ax.plot(xs, ys, color="black", lw=3.0, zorder=10)

    if not show_full_lattice:
        _set_relevant_region(ax, outline_qbs, pad=1.25)

    summary = result.get("summary", {})
    defect_text = ", ".join(f"QB{a}-QB{b}" for a, b in sorted(defect_edges)) or "none"
    ax.set_title(
        f"{title}\n"
        f"used qubits={len(used_qbs)}, used couplers={len(used_edges)}, defective coupler(s): {defect_text}",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def stim_tick_layers(circuit, mapping: dict[int, int]) -> list[dict[str, list]]:
    """Group mapped Stim operations into TICK-separated layers on IQM qubits."""
    layers = []
    current = defaultdict(list)

    def flush():
        nonlocal current
        if current:
            layers.append({k: list(v) for k, v in current.items()})
            current = defaultdict(list)

    for ins in circuit.flattened():
        name = ins.name
        targets = ins.targets_copy()
        if name == "TICK":
            flush()
            continue
        if name in {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
            continue
        if name.endswith("ERROR") or name.startswith(("DEPOLARIZE", "PAULI_CHANNEL")):
            continue
        if name in TWO_QUBIT_GATES:
            for i in range(0, len(targets), 2):
                a, b = int(targets[i].value), int(targets[i + 1].value)
                if a in mapping and b in mapping:
                    current["two"].append((name, mapping[a], mapping[b], a, b))
            continue
        if name in ONE_QUBIT_GATES:
            for t in targets:
                if t.is_qubit_target and int(t.value) in mapping:
                    current["one"].append((name, mapping[int(t.value)], int(t.value)))
    flush()
    return layers


def _draw_gate_layer(ax, layer: dict, used: set[int], title: str, zoom_to_used: bool = False):
    ax.set_aspect("equal")
    ax.axis("off")
    draw_emerald_base(ax, used, label_qubits=False, highlight_used_neighbor_couplers=False)

    for name, qa, qb, _, _ in layer.get("two", []):
        x0, y0 = _xy(qa)
        x1, y1 = _xy(qb)
        ax.plot([x0, x1], [y0, y1], color="white", lw=8.5, alpha=0.98, zorder=8)
        ax.plot([x0, x1], [y0, y1], color="black", lw=5.2, alpha=0.98, zorder=9,
                path_effects=[pe.Stroke(linewidth=7.0, foreground="white"), pe.Normal()])
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "black", "edgecolor": "white", "linewidth": 1.2},
                zorder=10)

    by_qb = defaultdict(list)
    for name, qb, _ in layer.get("one", []):
        by_qb[qb].append(name)
    for qb, names in by_qb.items():
        x, y = _xy(qb)
        text = "/".join(dict.fromkeys(names))
        ax.text(x, y + 0.38, text, ha="center", va="center", fontsize=8.3, color="white",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "black", "edgecolor": "white", "linewidth": 1.2},
                path_effects=[pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()],
                zorder=11)
    if zoom_to_used:
        _set_relevant_region(ax, used, pad=1.05)
    ax.set_title(title, fontsize=10)


def plot_iqm_timeslice_layers(
    result: dict,
    layer_indices: list[int] | None = None,
    max_layers: int = 12,
    cols: int = 4,
    title: str = "Stim TICK layers on IQM Emerald layout",
    zoom_to_used: bool = True,
    save_path: str | None = None,
):
    """SVG-timeslice-style visualization of mapped Stim gates on Emerald geometry."""
    circuit = result["stim_circuit"]
    mapping = _result_mapping(result)
    layers = stim_tick_layers(circuit, mapping)
    if layer_indices is None:
        layer_indices = list(range(min(max_layers, len(layers))))
    else:
        layer_indices = [i for i in layer_indices if 0 <= i < len(layers)]

    rows = max(1, math.ceil(len(layer_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 4.25 * rows))
    axes = np.atleast_1d(axes).ravel()
    used = set(mapping.values())
    for ax, layer_i in zip(axes, layer_indices):
        layer = layers[layer_i]
        n1 = len(layer.get("one", []))
        n2 = len(layer.get("two", []))
        _draw_gate_layer(ax, layer, used, f"TICK layer {layer_i}: {n2} two-q, {n1} one-q", zoom_to_used=zoom_to_used)
    for ax in axes[len(layer_indices):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _gate_layer_png_data_uri(layer: dict, used: set[int], title: str | None = None, zoom_to_used: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    _draw_gate_layer(ax, layer, used, title or "")
    if zoom_to_used:
        _set_relevant_region(ax, used, pad=1.15)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _stim_timeslice_svg_data_uri(circuit, tick: int) -> str:
    try:
        svg = str(circuit.diagram("timeslice-svg", tick=tick))
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    except Exception as exc:
        msg = f"Stim timeslice failed for tick {tick}: {exc}"
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='480' height='80'>"
            "<rect width='100%' height='100%' fill='white'/>"
            f"<text x='12' y='42' font-family='monospace' font-size='14' fill='#991b1b'>{msg}</text>"
            "</svg>"
        )
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def show_single_round_tick_comparison(
    result: dict,
    tick_indices: list[int] | None = None,
    max_ticks: int = 8,
    title: str = "One surface-code round: mapped Emerald gates vs Stim timeslices",
    zoom_to_used: bool = True,
):
    """Return an HTML side-by-side comparison of Emerald layers and Stim SVG timeslices.

    The left column shows the operations from one Stim TICK layer after applying
    the IQM qubit mapping. The right column is Stim's own `timeslice-svg` for
    the same tick, so the two can be compared directly.
    """
    from IPython.display import HTML

    circuit = result["stim_circuit"]
    mapping = _result_mapping(result)
    layers = stim_tick_layers(circuit, mapping)
    if tick_indices is None:
        tick_indices = list(range(min(max_ticks, len(layers))))
    else:
        tick_indices = [i for i in tick_indices if 0 <= i < len(layers)]

    used = set(mapping.values())
    sections = []
    for tick in tick_indices:
        layer = layers[tick]
        n1 = len(layer.get("one", []))
        n2 = len(layer.get("two", []))
        meta = f"{n2} two-qubit, {n1} one-qubit"
        emerald_png = _gate_layer_png_data_uri(layer, used, None, zoom_to_used=zoom_to_used)
        stim_svg = _stim_timeslice_svg_data_uri(circuit, tick)
        sections.append(
            f"""
            <section class="tick-section">
              <div class="tick-title">Tick {tick}</div>
              <div class="tick-meta">{meta}</div>
              <div class="tick-grid">
                <div class="tick-panel">
                  <div class="panel-label">Hardware</div>
                  <img class="emerald-layer" src="{emerald_png}" />
                </div>
                <div class="tick-panel">
                  <div class="panel-label">Stim</div>
                  <div class="stim-svg-card"><img class="stim-layer" src="{stim_svg}" /></div>
                </div>
              </div>
            </section>
            """
        )

    html = f"""
    <style>
      .tick-compare-wrap {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .tick-compare-wrap h3 {{
        margin: 0 0 14px 0;
        font-weight: 600;
      }}
      .tick-section {{
        border-top: 1px solid #d1d5db;
        padding: 14px 0 18px 0;
        width: 100%;
      }}
      .tick-title {{
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: #111827;
      }}
      .tick-meta {{
        text-align: center;
        font-size: 12px;
        color: #4b5563;
        margin: 2px 0 8px 0;
      }}
      .tick-grid {{
        display: grid;
        grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
        gap: 18px;
        align-items: center;
        justify-items: center;
      }}
      .tick-panel {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
      }}
      .panel-label {{
        font-size: 14px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 6px;
        text-align: center;
      }}
      .emerald-layer {{
        width: 420px;
        max-width: 44vw;
        display: block;
      }}
      .stim-layer {{
        width: 420px;
        max-width: 44vw;
        height: auto;
        background: white;
        display: block;
      }}
      .stim-svg-card {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: white;
        padding: 10px;
        border: 1px solid #e5e7eb;
      }}
      @media (max-width: 900px) {{
        .tick-grid {{
          grid-template-columns: 1fr;
        }}
        .emerald-layer,
        .stim-layer {{
          max-width: 92vw;
        }}
      }}
    </style>
    <div class="tick-compare-wrap">
      <h3>{title}</h3>
      {''.join(sections)}
    </div>
    """
    return HTML(html)


def _pos_xy(pos: Pos) -> tuple[float, float]:
    qb = COORD_TO_QB.get(pos)
    if qb is not None:
        return _xy(qb)
    return float(pos.x) + (0.25 if pos.y % 2 else 0.0), -float(pos.y)


def _snl_pos_label(pos: Pos) -> str:
    qb = COORD_TO_QB.get(pos)
    return f"QB{qb}" if qb is not None else f"({pos.x},{pos.y})"


def _order_polygon(points: list[tuple[float, float]], center: tuple[float, float]) -> list[tuple[float, float]]:
    return sorted(points, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))


def _snl_stabilizer_links(patch) -> set[tuple[Pos, Pos]]:
    links = set()
    for stabilizer in patch.stabilizers:
        for ancilla in stabilizer.ancilla:
            for data in stabilizer.data_qubits:
                links.add(tuple(sorted((ancilla, data))))
    return links


def _snl_layer_stabilizers(patch, layer: str):
    if layer == "first":
        return list(getattr(patch, "undamaged_stabilizers", [])) + list(getattr(patch, "first_super", []))
    if layer == "second":
        return list(getattr(patch, "undamaged_stabilizers", [])) + list(getattr(patch, "second_super", []))
    raise ValueError("layer must be 'first' or 'second'")


def _draw_snl_layer(ax, result: dict, layer: str, show_full_context: bool = False):
    from matplotlib.patches import Polygon

    patch = result["damaged_patch"]
    clean_patch = result["clean_patch"]
    data_defects = set(result.get("data_defects", set()))
    ancilla_defects = set(result.get("ancilla_defects", set()))
    link_defects = {tuple(sorted(link)) for link in result.get("link_defects", set())}

    stabilizers = _snl_layer_stabilizers(patch, layer)
    active = set(patch.data_qubits) | set(patch.ancilla_qubits)
    visible = set(active) | data_defects | ancilla_defects
    for a, b in link_defects:
        visible.add(a)
        visible.add(b)
    if show_full_context:
        visible |= set(clean_patch.data_qubits) | set(clean_patch.ancilla_qubits)

    used_links = set()
    for stabilizer in stabilizers:
        gauges = stabilizer.gauges if isinstance(stabilizer, SuperStabilizer) else [stabilizer]
        for gauge in gauges:
            for q in gauge.data_qubits:
                used_links.add(tuple(sorted((gauge.only_ancilla, q))))

    ax.set_aspect("equal")
    ax.axis("off")

    # Light full Emerald-ish coupler background for this patch, then active layer links.
    clean_links = {link for link in _snl_stabilizer_links(clean_patch) if link[0] in visible and link[1] in visible}
    for a, b in sorted(clean_links):
        x0, y0 = _pos_xy(a)
        x1, y1 = _pos_xy(b)
        link = tuple(sorted((a, b)))
        is_used = link in used_links
        is_defect = link in link_defects
        if is_defect:
            color, lw, style, alpha = "#111827", 3.0, (0, (5, 4)), 0.95
        elif is_used:
            color, lw, style, alpha = "#16a34a", 2.4, "solid", 0.9
        else:
            color, lw, style, alpha = "#9be375", 1.1, "solid", 0.65
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, linestyle=style, alpha=alpha, zorder=1)
        if not is_defect:
            ax.scatter([(x0 + x1) / 2], [(y0 + y1) / 2], marker="D",
                       s=360 if is_used else 240, c="#16a34a" if is_used else "#8dda68",
                       edgecolors="white", linewidths=0.8, zorder=2, alpha=0.95 if is_used else 0.7)

    # Gauge plaquettes measured in this check layer.
    for stabilizer in stabilizers:
        gauges = stabilizer.gauges if isinstance(stabilizer, SuperStabilizer) else [stabilizer]
        color = "#4f7cff" if stabilizer.type == PauliT.X else "#ff5a5f"
        edge = "#1d4ed8" if stabilizer.type == PauliT.X else "#b91c1c"
        label = "X" if stabilizer.type == PauliT.X else "Z"

        for gauge in gauges:
            pts = [_pos_xy(q) for q in gauge.data_qubits]
            if len(pts) == 2:
                pts.append(_pos_xy(gauge.only_ancilla))
            center = _pos_xy(gauge.only_ancilla)
            if len(pts) >= 3:
                ax.add_patch(Polygon(
                    _order_polygon(pts, center),
                    closed=True,
                    facecolor=color,
                    edgecolor=edge,
                    lw=1.6,
                    alpha=0.20,
                    zorder=3,
                ))
            ax.text(
                center[0], center[1], label,
                ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                bbox={"boxstyle": "circle,pad=0.2", "facecolor": edge, "edgecolor": "white"},
                zorder=7,
            )

        # Dashed outline of the full superstabilizer support.
        if isinstance(stabilizer, SuperStabilizer):
            pts = [_pos_xy(q) for q in stabilizer.data_qubits] + [_pos_xy(a) for a in stabilizer.ancilla]
            if len(pts) >= 3:
                center = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
                ax.add_patch(Polygon(
                    _order_polygon(pts, center),
                    closed=True,
                    facecolor="none",
                    edgecolor=edge,
                    lw=2.6,
                    linestyle=(0, (6, 4)),
                    zorder=6,
                ))

    for p in sorted(visible):
        x, y = _pos_xy(p)
        is_defect = p in data_defects or p in ancilla_defects
        is_active = p in active
        if is_defect:
            face, text_color, label = "#4c1d95", "white", f"{_snl_pos_label(p)}\ndefect"
        elif is_active:
            face, text_color, label = "#5fcf55", "white", _snl_pos_label(p)
        else:
            face, text_color, label = "#f8fafc", "#94a3b8", _snl_pos_label(p)
        ax.scatter([x], [y], s=680 if is_active or is_defect else 420, c=face,
                   edgecolors="white" if is_active or is_defect else "#cbd5e1", lw=2, zorder=5)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.4, color=text_color, zorder=8)

    layer_name = "first SnL layer" if layer == "first" else "second SnL layer"
    ax.set_title(f"check round with {layer_name} plus undamaged stabilizers\n"
                 f"effective distance={patch.effective_distance}", fontsize=11)


def plot_snl_superstabilizer_rounds(
    result: dict,
    title: str = "Snakes-and-Ladders superstabilizer check rounds on IQM Emerald",
    show_full_context: bool = False,
    save_path: str | None = None,
):
    """Visualize the two alternating SnL superstabilizer measurement rounds.

    Pass a single SnL pipeline result, e.g.
    `snl_d3_faulty_coupler["results"][0]`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.8))
    _draw_snl_layer(axes[0], result, "first", show_full_context=show_full_context)
    _draw_snl_layer(axes[1], result, "second", show_full_context=show_full_context)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
