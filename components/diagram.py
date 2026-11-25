"""Architecture diagram: render_diagram(config) -> matplotlib.Figure.

Draws a vertical stack of labelled boxes (Input -> hidden layers -> Output)
with downward arrows. Layer types are colour-coded. No streamlit import --
returns a plain Figure for the caller to display with st.pyplot().
"""
from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

# ── Layout constants (data-coordinate units) ──────────────────────────────────
_BOX_W = 2.6
_BOX_H = 0.80   # taller than Phase 1 to fit CNN shape annotations
_GAP   = 0.48
_STEP  = _BOX_H + _GAP
_CX    = 1.60
_FIG_W = 3.80   # wider to fit longer CNN labels

# ── Colour palette ─────────────────────────────────────────────────────────────
_STYLE = {
    "input":   dict(face="#D6EAF8", edge="#2E86C1", lw=2.0),
    "hidden":  dict(face="#FDFEFE", edge="#AAB7B8", lw=1.4),  # Linear
    "conv":    dict(face="#FEF9E7", edge="#D4AC0D", lw=1.4),  # Conv2d
    "pool":    dict(face="#F9EBEA", edge="#C0392B", lw=1.4),  # MaxPool2d
    "flatten": dict(face="#EBF5FB", edge="#5DADE2", lw=1.4),  # Flatten
    "output":  dict(face="#D5F5E3", edge="#1E8449", lw=2.0),
}


def _build_rows(config: dict) -> list[tuple[str, str]]:
    """Return (kind, label) pairs ordered top-to-bottom."""
    from components.shapes import compute_shapes
    shapes = compute_shapes(config)
    rows: list[tuple[str, str]] = []

    # Input row
    if config.get("input_shape"):
        C, H, W = config["input_shape"]
        rows.append(("input", f"Input\n({C}x{H}x{W})"))
    else:
        rows.append(("input", f"Input\n({config['input_dim']} features)"))

    for i, layer in enumerate(config["hidden_layers"]):
        ltype     = layer.get("type", "Linear")
        out_shape = shapes[i + 1]

        if ltype == "Conv2d":
            bn        = " BN" if layer.get("batch_norm") else ""
            shape_str = (
                f"{out_shape[0]}x{out_shape[1]}x{out_shape[2]}"
                if out_shape else "invalid"
            )
            label = (
                f"Conv2d ({layer['out_channels']}@"
                f"{layer['kernel_size']}x{layer['kernel_size']}){bn}\n"
                f"{layer['activation']}  ->  {shape_str}"
            )
            rows.append(("conv", label))

        elif ltype == "MaxPool2d":
            shape_str = (
                f"{out_shape[0]}x{out_shape[1]}x{out_shape[2]}"
                if out_shape else "invalid"
            )
            label = (
                f"MaxPool2d ({layer['kernel_size']}x{layer['kernel_size']})\n"
                f"->  {shape_str}"
            )
            rows.append(("pool", label))

        elif ltype == "Flatten":
            flat  = out_shape[0] if out_shape else "?"
            label = f"Flatten\n->  {flat} units"
            rows.append(("flatten", label))

        else:  # Linear
            bn    = " + BN" if layer.get("batch_norm") else ""
            do    = (
                f" + Drop({layer['dropout_rate']:.2f})"
                if layer.get("dropout") and layer.get("dropout_rate", 0) > 0
                else ""
            )
            label = f"Linear ({layer['units']}){bn}{do}\n+ {layer['activation']}"
            rows.append(("hidden", label))

    # Output row
    if config["task"] == "binary":
        rows.append(("output", "Linear (1)\n+ Sigmoid"))
    else:
        rows.append(("output", f"Linear ({config['n_classes']})\n+ Softmax"))

    return rows


def render_diagram(config: dict) -> Figure:
    """Return a Figure showing the network architecture as a vertical flow."""
    rows = _build_rows(config)
    n    = len(rows)

    # Clamp height so very deep networks don't create enormous figures
    fig_h = min(max(n * _BOX_H + (n - 1) * _GAP + 1.4, 3.0), 20.0)

    fig = Figure(figsize=(_FIG_W, fig_h))
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, _FIG_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    top_y = fig_h - 0.65

    for i, (kind, label) in enumerate(rows):
        y_top = top_y - i * _STEP
        y_bot = y_top - _BOX_H
        y_ctr = (y_top + y_bot) / 2.0
        s     = _STYLE[kind]

        ax.add_patch(FancyBboxPatch(
            (_CX - _BOX_W / 2, y_bot), _BOX_W, _BOX_H,
            boxstyle="round,pad=0.06",
            linewidth=s["lw"],
            edgecolor=s["edge"],
            facecolor=s["face"],
            zorder=2,
        ))

        ax.text(
            _CX, y_ctr, label,
            ha="center", va="center",
            fontsize=8.0,
            fontweight="bold" if kind in ("input", "output") else "normal",
            linespacing=1.5,
            zorder=3,
        )

        if i > 0:
            prev_y_bot = top_y - (i - 1) * _STEP - _BOX_H
            ax.annotate(
                "",
                xy=(_CX, y_top + 0.03),
                xytext=(_CX, prev_y_bot - 0.03),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#666666",
                    lw=1.5,
                    mutation_scale=13,
                ),
                zorder=1,
            )

    fig.tight_layout(pad=0.2)
    return fig
