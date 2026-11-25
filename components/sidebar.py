"""Sidebar controls. Call render() inside a `with st.sidebar:` block.

Returns a config dict matching the Phase 2 schema.
This is the only component that imports streamlit.
"""
import torch
import streamlit as st

from datasets.registry import list as list_datasets, meta as dataset_meta, suggested as suggested_arch

_CUDA_AVAILABLE = torch.cuda.is_available()
_GPU_NAME       = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None

ACTIVATIONS  = ["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "Linear"]
CONV_ACT     = ["ReLU", "Tanh", "LeakyReLU", "Linear"]   # Sigmoid rarely used in conv
OPTIMIZERS   = ["Adam", "SGD", "Momentum"]
BATCH_SIZES  = ["full", 32, 64, 128]
LR_OPTIONS   = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
WD_OPTIONS   = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
LAYER_TYPES  = ["Linear", "Conv2d", "MaxPool2d", "Flatten"]

_DEFAULTS: dict[str, dict] = {
    "Linear":    {"type": "Linear",    "units": 64,  "activation": "ReLU",
                  "batch_norm": False, "dropout": False, "dropout_rate": 0.0},
    "Conv2d":    {"type": "Conv2d",    "out_channels": 16, "kernel_size": 3,
                  "activation": "ReLU", "batch_norm": False},
    "MaxPool2d": {"type": "MaxPool2d", "kernel_size": 2},
    "Flatten":   {"type": "Flatten"},
}


def _init_state() -> None:
    """Seed session state on first load."""
    if "layers" not in st.session_state:
        st.session_state["layers"] = [
            {"id": 0, "type": "Linear", "units": 16, "activation": "ReLU",
             "batch_norm": False, "dropout": False, "dropout_rate": 0.0},
            {"id": 1, "type": "Linear", "units": 8,  "activation": "ReLU",
             "batch_norm": False, "dropout": False, "dropout_rate": 0.0},
        ]
    if "_layer_counter" not in st.session_state:
        st.session_state["_layer_counter"] = 2
    if "_selected_layer" not in st.session_state:
        st.session_state["_selected_layer"] = None


def _check_ordering(layers: list) -> tuple[bool, str]:
    """Return (ok, message). Warnings are non-blocking — shown but don't prevent training."""
    flatten_seen = False
    prev_type    = None
    for layer in layers:
        ltype = layer.get("type", "Linear")
        if ltype == "Flatten":
            flatten_seen = True
        elif ltype in ("Conv2d", "MaxPool2d") and flatten_seen:
            return False, "Conv2d / MaxPool2d appears after Flatten — conv layers must come before Flatten."
        elif ltype == "Linear" and prev_type in ("Conv2d", "MaxPool2d"):
            return False, "Linear layer directly after Conv2d/MaxPool2d — insert a Flatten layer between them."
        prev_type = ltype
    return True, ""


def _serialize_layer(layer: dict) -> dict:
    """Return the config-schema representation of a layer (no 'id' key)."""
    ltype = layer.get("type", "Linear")
    if ltype == "Conv2d":
        return {
            "type":        "Conv2d",
            "out_channels": layer.get("out_channels", 16),
            "kernel_size":  layer.get("kernel_size", 3),
            "activation":   layer.get("activation", "ReLU"),
            "batch_norm":   layer.get("batch_norm", False),
        }
    if ltype == "MaxPool2d":
        return {"type": "MaxPool2d", "kernel_size": layer.get("kernel_size", 2)}
    if ltype == "Flatten":
        return {"type": "Flatten"}
    # Linear (default, also handles old-style dicts without "type")
    return {
        "type":         "Linear",
        "units":        layer.get("units", 16),
        "activation":   layer.get("activation", "ReLU"),
        "batch_norm":   layer.get("batch_norm", False),
        "dropout":      layer.get("dropout", False),
        "dropout_rate": layer.get("dropout_rate", 0.0),
    }


def render() -> dict:
    """Render sidebar widgets and return a config dict."""
    _init_state()

    # ── Dataset ──────────────────────────────────────────────────────────
    st.subheader("Dataset")
    names   = list_datasets()
    dataset = st.selectbox(
        "Dataset", names,
        index=names.index("moons"),
        label_visibility="collapsed",
    )
    info = dataset_meta(dataset)
    if info.input_shape:
        C, H, W  = info.input_shape
        dim_str  = f"Input: **{C}×{H}×{W}** · Classes: **{info.n_classes}**"
    else:
        dim_str  = f"Input dim: **{info.input_dim}** · Classes: **{info.n_classes}**"
    st.caption(f"{info.description}  \n{dim_str}")

    # ── Hidden layers ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Hidden Layers")

    if st.button("Use suggested architecture", use_container_width=True):
        arch = suggested_arch(dataset)
        nid  = st.session_state["_layer_counter"]
        st.session_state["layers"] = [{**l, "id": nid + i} for i, l in enumerate(arch)]
        st.session_state["_layer_counter"] = nid + len(arch)
        st.session_state["_selected_layer"] = None
        st.rerun()

    # Non-blocking ordering validation
    ordering_ok, ordering_msg = _check_ordering(st.session_state["layers"])
    if not ordering_ok:
        st.warning(ordering_msg)

    if not st.session_state["layers"]:
        st.warning("No hidden layers — add at least one.")

    # Deferred mutations — applied after the full layer loop
    to_remove = None   # int index
    to_move   = None   # ("up", int) | ("down", int)
    to_retype = None   # (int, str)  — replace layer at index with new type defaults

    layers = st.session_state["layers"]
    n      = len(layers)

    for i, layer in enumerate(layers):
        lid   = layer["id"]
        ltype = layer.get("type", "Linear")

        is_selected = layer["id"] == st.session_state.get("_selected_layer")
        with st.container(border=True):
            if is_selected:
                st.markdown(
                    '<div style="height:3px;background:#2E86C1;'
                    'border-radius:4px;margin-bottom:4px"></div>',
                    unsafe_allow_html=True,
                )
            # ── Header: number + type selector + ↑ ↓ × ─────────────────
            c_num, c_tsel, c_up, c_dn, c_rm = st.columns(
                [1.2, 3.2, 0.7, 0.7, 0.7], vertical_alignment="bottom",
            )
            with c_num:
                st.markdown(f"**Layer {i + 1}**")
            with c_tsel:
                new_ltype = st.selectbox(
                    "Layer type", LAYER_TYPES,
                    index=LAYER_TYPES.index(ltype),
                    key=f"ltype_{lid}",
                    label_visibility="collapsed",
                )
            with c_up:
                if st.button("↑", key=f"up_{lid}", disabled=(i == 0), help="Move up"):
                    to_move = ("up", i)
            with c_dn:
                if st.button("↓", key=f"dn_{lid}", disabled=(i == n - 1), help="Move down"):
                    to_move = ("down", i)
            with c_rm:
                if st.button("×", key=f"rm_{lid}", help="Remove layer"):
                    to_remove = i

            if new_ltype != ltype:
                to_retype = (i, new_ltype)

            # ── Type-specific parameters ─────────────────────────────────
            if ltype == "Flatten":
                st.caption("No parameters")

            elif ltype == "MaxPool2d":
                ks_opts = [2, 3]
                layer["kernel_size"] = st.selectbox(
                    "Kernel size", ks_opts,
                    index=ks_opts.index(layer.get("kernel_size", 2)),
                    key=f"ks_{lid}",
                )

            elif ltype == "Conv2d":
                c_ch, c_ks, c_act, c_bn = st.columns([2, 1, 2, 1], vertical_alignment="bottom")
                with c_ch:
                    layer["out_channels"] = st.number_input(
                        "Channels", min_value=1, max_value=512,
                        value=layer.get("out_channels", 16), key=f"ch_{lid}",
                    )
                with c_ks:
                    ks_opts = [3, 5]
                    layer["kernel_size"] = st.selectbox(
                        "K", ks_opts,
                        index=ks_opts.index(layer.get("kernel_size", 3)),
                        key=f"ks_{lid}",
                    )
                with c_act:
                    cur_act = layer.get("activation", "ReLU")
                    if cur_act not in CONV_ACT:
                        cur_act = "ReLU"
                    layer["activation"] = st.selectbox(
                        "Activation", CONV_ACT,
                        index=CONV_ACT.index(cur_act),
                        key=f"act_{lid}",
                    )
                with c_bn:
                    layer["batch_norm"] = st.checkbox(
                        "BN", value=layer.get("batch_norm", False), key=f"bn_{lid}",
                        help="Insert BatchNorm2d after the conv layer",
                    )

            else:  # Linear (also handles old-style dicts without "type")
                if "batch_norm" not in layer:
                    layer["batch_norm"] = False
                c_units, c_act, c_bn = st.columns([2, 2, 1], vertical_alignment="bottom")
                with c_units:
                    layer["units"] = st.number_input(
                        "Units", min_value=1, max_value=1024,
                        value=layer.get("units", 16), key=f"units_{lid}",
                    )
                with c_act:
                    layer["activation"] = st.selectbox(
                        "Activation", ACTIVATIONS,
                        index=ACTIVATIONS.index(layer.get("activation", "ReLU")),
                        key=f"act_{lid}",
                    )
                with c_bn:
                    layer["batch_norm"] = st.checkbox(
                        "BN", value=layer["batch_norm"], key=f"bn_{lid}",
                        help="Insert BatchNorm1d after the linear layer",
                    )
                # Dropout sub-row
                c_do_lbl, c_do_tog, c_do_rate = st.columns([1, 1, 3], vertical_alignment="center")
                with c_do_lbl:
                    st.caption("Dropout")
                with c_do_tog:
                    do_on = st.toggle(
                        "Enable dropout", value=layer.get("dropout", False),
                        key=f"do_{lid}", label_visibility="collapsed",
                    )
                    layer["dropout"] = do_on
                with c_do_rate:
                    if do_on:
                        layer["dropout_rate"] = st.slider(
                            "Rate", 0.0, 0.5,
                            value=float(layer.get("dropout_rate", 0.2)),
                            step=0.05, key=f"dr_{lid}",
                            label_visibility="collapsed",
                        )
                    else:
                        layer["dropout_rate"] = float(layer.get("dropout_rate", 0.0))

    # Apply deferred mutations in priority order: retype > move > remove
    if to_retype is not None:
        idx, new_type = to_retype
        old_id = layers[idx]["id"]
        layers[idx] = {**_DEFAULTS[new_type], "id": old_id}
        st.rerun()
    elif to_move is not None:
        direction, idx = to_move
        if direction == "up" and idx > 0:
            layers[idx], layers[idx - 1] = layers[idx - 1], layers[idx]
            st.session_state["_selected_layer"] = layers[idx - 1]["id"]
        elif direction == "down" and idx < n - 1:
            layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
            st.session_state["_selected_layer"] = layers[idx + 1]["id"]
        st.rerun()
    elif to_remove is not None:
        layers.pop(to_remove)
        st.rerun()

    # ── Add Layer ─────────────────────────────────────────────────────────
    c_type, c_add = st.columns([3, 2])
    with c_type:
        new_type = st.selectbox(
            "Layer type", LAYER_TYPES,
            label_visibility="collapsed", key="new_layer_type",
        )
    with c_add:
        if st.button("+ Add Layer", use_container_width=True):
            nid = st.session_state["_layer_counter"]
            st.session_state["_layer_counter"] += 1
            layers.append({**_DEFAULTS[new_type], "id": nid})
            st.session_state["_selected_layer"] = nid
            st.rerun()

    # ── Output layer (read-only) ──────────────────────────────────────────
    st.divider()
    st.subheader("Output (auto)")
    if info.task == "binary":
        st.caption("1 unit · Sigmoid · BCE Loss")
    else:
        st.caption(f"{info.n_classes} units · Softmax · CrossEntropy Loss")

    # ── Training ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Training")
    optimizer = st.selectbox("Optimizer", OPTIMIZERS)
    lr = st.select_slider(
        "Learning Rate",
        options=LR_OPTIONS,
        value=1e-3,
        format_func=lambda x: f"{x:.0e}",
        help="Step size per gradient update. Too high may cause divergence (NaN loss).",
    )
    epochs = st.number_input(
        "Epochs", min_value=1, max_value=5000, value=500, step=50,
        help="Number of full passes through the training data.",
    )
    batch_size = st.selectbox(
        "Batch Size", BATCH_SIZES,
        help='"full" uses all training samples per update. '
             "Mini-batches add noise that can help generalisation.",
    )
    weight_decay = st.select_slider(
        "L2 Weight Decay",
        options=WD_OPTIONS,
        value=0.0,
        format_func=lambda x: "off" if x == 0.0 else f"{x:.0e}",
        help="L2 regularization strength passed as weight_decay to the optimizer.",
    )
    seed = st.number_input(
        "Seed", min_value=0, max_value=99999, value=42,
        help="Seed for weight initialisation, DataLoader shuffle, and train/val split.",
    )

    # ── Early stopping ────────────────────────────────────────────────────
    es_on = st.toggle(
        "Early stopping", value=False,
        help="Stop training automatically when validation loss stops improving.",
    )
    es_patience  = 20
    es_min_delta = 0.0
    if es_on:
        c_esp, c_esd = st.columns(2)
        with c_esp:
            es_patience = st.number_input(
                "Patience", min_value=1, max_value=500, value=20, step=5,
                help="Epochs without improvement before stopping.",
            )
        with c_esd:
            _ES_DELTA = [0.0, 1e-4, 1e-3, 1e-2]
            es_min_delta = st.select_slider(
                "Min Δ", options=_ES_DELTA,
                format_func=lambda x: "0" if x == 0.0 else f"{x:.0e}",
                help="Minimum decrease in val loss to count as improvement.",
            )

    # ── Device ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Device")
    if _CUDA_AVAILABLE:
        use_gpu = st.toggle("Use GPU", value=True, help=f"Detected: {_GPU_NAME}")
        st.caption(f"{'GPU' if use_gpu else 'CPU'}  ·  {_GPU_NAME}")
        device = "cuda" if use_gpu else "cpu"
    else:
        st.toggle("Use GPU", value=False, disabled=True, help="No CUDA-capable GPU detected.")
        st.caption("CPU only  ·  no CUDA GPU detected")
        device = "cpu"

    return {
        "dataset":        dataset,
        "task":           info.task,
        "input_dim":      info.input_dim,
        "input_shape":    info.input_shape,
        "n_classes":      info.n_classes,
        "hidden_layers":  [_serialize_layer(l) for l in layers],
        "optimizer":      optimizer,
        "lr":             float(lr),
        "epochs":         int(epochs),
        "batch_size":     batch_size,
        "weight_decay":   float(weight_decay),
        "seed":           int(seed),
        "early_stopping": bool(es_on),
        "es_patience":    int(es_patience),
        "es_min_delta":   float(es_min_delta),
        "device":         device,
    }
