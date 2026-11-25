import copy
import json
import time
import threading

import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from components.sidebar import render as render_sidebar
from components.code_gen import generate_code
from components.diagram import render_diagram
from components.trainer import build_model, train_model
from components.visualizer import (
    plot_loss_curve,
    plot_decision_boundary,
    plot_confusion_matrix,
)
from datasets.registry import load as load_dataset, meta as dataset_meta

st.set_page_config(
    page_title="Neural Network Builder",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ▶ Train — the sole primary button on this page */
button[kind="primary"] {
    background-color: #27AE60 !important;
    border-color:     #27AE60 !important;
}
button[kind="primary"]:hover {
    background-color: #1E8449 !important;
    border-color:     #1E8449 !important;
}
button[kind="primary"]:disabled {
    background-color: #A9DFBF !important;
    border-color:     #A9DFBF !important;
    opacity: 0.7 !important;
}
/* ⏹ Stop — target the column (sibling scope) containing .red-btn-marker.
   Using stColumn, not stVerticalBlock, so the selector cannot bleed upward
   to an ancestor that also contains the Train button. */
[data-testid="stColumn"]:has(.red-btn-marker) button {
    background-color: #E74C3C !important;
    border-color:     #E74C3C !important;
    color: white !important;
}
[data-testid="stColumn"]:has(.red-btn-marker) button:hover {
    background-color: #C0392B !important;
    border-color:     #C0392B !important;
}
[data-testid="stColumn"]:has(.red-btn-marker) button:disabled {
    background-color: #FADBD8 !important;
    border-color:     #FADBD8 !important;
    color: #A93226 !important;
    opacity: 0.6 !important;
}
</style>
""", unsafe_allow_html=True)


def _config_changed(current: dict, last: dict) -> bool:
    """True when current config differs from the last trained config."""
    try:
        return (
            json.dumps(current, sort_keys=True, default=str)
            != json.dumps(last, sort_keys=True, default=str)
        )
    except (TypeError, ValueError):
        return True


# ── Session state defaults ────────────────────────────────────────────────────
for _key, _default in [
    ("model",            None),
    ("history",          None),
    ("last_config",      None),
    ("trained",          False),
    ("X_tr",             None),
    ("y_tr",             None),
    ("X_v",              None),
    ("y_v",              None),
    ("X_test",           None),
    ("y_test",           None),
    ("n_train",          None),
    ("n_val",            None),
    ("n_test",           None),
    # training thread state
    ("_training",        False),
    ("_training_thread", None),
    ("_stop_event",      None),
    ("_shared",          None),
    ("_total_epochs",    0),
    ("_pending_config",  None),
    ("_pending_data",    None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

with st.sidebar:
    config = render_sidebar()

# ── Top panel: diagram + code ─────────────────────────────────────────────────
st.title("Neural Network Builder")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Architecture Diagram")
    st.pyplot(render_diagram(config), use_container_width=True)
with col_right:
    with st.expander("Generated PyTorch Code", expanded=False):
        st.code(generate_code(config), language="python")

st.divider()

# ── Dataset & Training ────────────────────────────────────────────────────────
st.subheader("Dataset & Training")

info = dataset_meta(config["dataset"])
if info.input_shape:
    C, H, W = info.input_shape
    dim_str = f"{C}×{H}×{W} input"
else:
    dim_str = f"{info.input_dim} features"
st.caption(
    f"**{config['dataset']}**  —  {info.description}  "
    f"·  {dim_str}  ·  {info.n_classes} classes  ·  {info.task}"
)

if config["dataset"] == "mnist":
    st.info(
        "**MNIST note:** training 10 000 samples on CPU takes roughly 30 s per 10 epochs "
        "with Adam full-batch. Consider fewer epochs or mini-batches for quick experiments."
    )
elif config["dataset"] == "fashion_mnist":
    st.info(
        "**Fashion-MNIST note:** 10 000 samples, 1×28×28 images. "
        "A CNN with Conv→Pool→Flatten→Linear typically outperforms a plain MLP."
    )
elif config["dataset"] == "cifar10":
    st.warning(
        "**CIFAR-10 note:** 10 000 samples, 3×32×32 images — training is slower than Fashion-MNIST. "
        "Use mini-batches and a CNN for best results."
    )

# ── Button row ────────────────────────────────────────────────────────────────
is_training = st.session_state["_training"]

col_btn, col_stop, col_status = st.columns([1, 1, 4])
with col_btn:
    train_clicked = st.button(
        "▶ Train", type="primary", use_container_width=True,
        disabled=is_training,
    )
with col_stop:
    st.markdown('<div class="red-btn-marker"></div>', unsafe_allow_html=True)
    stop_clicked = st.button(
        "⏹ Stop", key="stop_btn", use_container_width=True,
        disabled=not is_training,
        help="Stop training after the current epoch completes.",
    )
    if stop_clicked and st.session_state["_stop_event"] is not None:
        st.session_state["_stop_event"].set()

with col_status:
    if is_training:
        shared = st.session_state["_shared"]
        ep     = shared.get("epoch", -1)
        total  = st.session_state["_total_epochs"]
        if ep >= 0 and shared["history"]["val_loss"]:
            st.info(
                f"Training… epoch **{ep + 1}/{total}** · "
                f"val loss {shared['history']['val_loss'][-1]:.4f} · "
                f"val acc {shared['history']['val_acc'][-1]:.1%}"
            )
        else:
            st.info("Starting training…")
    elif st.session_state["trained"]:
        h  = st.session_state["history"]
        lc = st.session_state["last_config"]
        epochs_trained = len(h["train_loss"])
        stop_note = (
            f" _(stopped at epoch {epochs_trained})_"
            if epochs_trained < lc["epochs"] else ""
        )
        st.success(
            f"Trained on **{lc['dataset']}**  ·  "
            f"train {st.session_state['n_train']} / "
            f"val {st.session_state['n_val']} / "
            f"test {st.session_state['n_test']}  ·  "
            f"val loss {h['val_loss'][-1]:.4f}  ·  "
            f"val acc {h['val_acc'][-1]:.1%}  ·  "
            f"{epochs_trained} epochs{stop_note}"
        )
    else:
        st.info("Configure the network in the sidebar, then click ▶ Train.")

# ── Start training (non-blocking) ────────────────────────────────────────────
if train_clicked and not is_training:
    if not config["hidden_layers"]:
        st.error("Add at least one hidden layer before training.")
    else:
        try:
            with st.spinner(f"Loading {config['dataset']}…"):
                X_all, y_all, X_test, y_test = load_dataset(
                    config["dataset"], seed=config["seed"]
                )
                X_tr, X_v, y_tr, y_v = train_test_split(
                    X_all, y_all,
                    test_size=0.2,
                    random_state=config["seed"],
                )

            stop_event   = threading.Event()
            total_epochs = config["epochs"]
            shared: dict = {
                "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
                "epoch":   -1,
                "done":    False,
                "error":   None,
                "model":   None,
            }
            cfg_copy = copy.deepcopy(config)

            def on_epoch(ep: int, metrics: dict) -> None:
                for k, v in metrics.items():
                    shared["history"][k].append(v)
                shared["epoch"] = ep

            def _train_thread() -> None:
                try:
                    mdl = build_model(cfg_copy)
                    mdl, hist = train_model(
                        mdl, cfg_copy, X_tr, y_tr, X_v, y_v,
                        on_epoch=on_epoch,
                        stop_event=stop_event,
                    )
                    shared["model"]   = mdl
                    shared["history"] = hist
                except (ValueError, RuntimeError) as exc:
                    shared["error"] = str(exc)
                except Exception as exc:
                    shared["error"] = f"Unexpected error: {exc}"
                finally:
                    shared["done"] = True

            thread = threading.Thread(target=_train_thread, daemon=True)
            st.session_state.update({
                "_training":        True,
                "_training_thread": thread,
                "_stop_event":      stop_event,
                "_shared":          shared,
                "_total_epochs":    total_epochs,
                "_pending_config":  cfg_copy,
                "_pending_data": {
                    "X_tr":    X_tr,  "y_tr":    y_tr,
                    "X_v":     X_v,   "y_v":     y_v,
                    "X_test":  X_test, "y_test": y_test,
                    "n_train": len(X_tr),
                    "n_val":   len(X_v),
                    "n_test":  len(X_test),
                },
            })
            thread.start()
            st.rerun()

        except Exception as exc:
            st.error(f"Error loading data: {exc}")

# ── Polling loop — runs on every rerun while _training is True ────────────────
if st.session_state["_training"]:
    shared = st.session_state["_shared"]
    total  = st.session_state["_total_epochs"]

    prog_bar  = st.empty()
    stat_text = st.empty()
    live_ch   = st.empty()

    ep   = shared.get("epoch", -1)
    hist = shared["history"]
    if ep >= 0 and hist["train_loss"]:
        prog_bar.progress(min(1.0, (ep + 1) / total))
        stat_text.markdown(
            f"**Epoch {ep + 1}/{total}** — "
            f"train loss: {hist['train_loss'][-1]:.4f}  ·  "
            f"val loss: {hist['val_loss'][-1]:.4f}  ·  "
            f"val acc: {hist['val_acc'][-1]:.1%}"
        )
        live_ch.pyplot(plot_loss_curve(hist), use_container_width=True)
    else:
        stat_text.info("Initialising model and data…")

    if not shared["done"]:
        time.sleep(0.4)
        st.rerun()
    else:
        prog_bar.empty()
        stat_text.empty()
        live_ch.empty()

        if shared["error"]:
            st.error(shared["error"])
            st.session_state["_training"] = False
            st.rerun()
        else:
            data = st.session_state["_pending_data"]
            st.session_state.update({
                "model":       shared["model"],
                "history":     shared["history"],
                "last_config": st.session_state["_pending_config"],
                "trained":     True,
                "X_tr":        data["X_tr"],
                "y_tr":        data["y_tr"],
                "X_v":         data["X_v"],
                "y_v":         data["y_v"],
                "X_test":      data["X_test"],
                "y_test":      data["y_test"],
                "n_train":     data["n_train"],
                "n_val":       data["n_val"],
                "n_test":      data["n_test"],
                "_training":   False,
            })
            st.rerun()

st.divider()

# ── Results ───────────────────────────────────────────────────────────────────
lc = st.session_state["last_config"]

if st.session_state["trained"] and lc is not None and _config_changed(config, lc):
    st.warning(
        "The current configuration differs from the last training run. "
        "Click **▶ Train** to update results."
    )

col_loss, col_viz = st.columns(2)

with col_loss:
    st.subheader("Loss & Accuracy")
    if st.session_state["trained"]:
        st.pyplot(
            plot_loss_curve(st.session_state["history"]),
            use_container_width=True,
        )
    else:
        st.info("Train the network to see results.")

with col_viz:
    if lc is not None and lc["input_dim"] == 2:
        st.subheader("Decision Boundary")
    else:
        st.subheader("Confusion Matrix")

    if st.session_state["trained"] and lc is not None:
        model = st.session_state["model"]
        if lc["input_dim"] == 2:
            X_all_viz = np.vstack([st.session_state["X_tr"], st.session_state["X_v"]])
            y_all_viz = np.concatenate([st.session_state["y_tr"], st.session_state["y_v"]])
            st.pyplot(
                plot_decision_boundary(model, X_all_viz, y_all_viz, lc),
                use_container_width=True,
            )
        else:
            st.pyplot(
                plot_confusion_matrix(
                    model,
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                    lc,
                    input_shape=lc.get("input_shape"),
                ),
                use_container_width=True,
            )
    else:
        st.info("Train the network to see results.")
