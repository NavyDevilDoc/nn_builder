"""Code generation: config dict -> complete, runnable PyTorch script string.

The generated code mirrors exactly what trainer.py executes:
  - Binary task  : Sigmoid output, BCELoss, y as float32 shape (N, 1)
  - Multiclass   : raw logit output, CrossEntropyLoss, y as long shape (N,)
  - Momentum opt : SGD with momentum=0.9
  - Mini-batch   : DataLoader over TensorDataset; eval always on full tensors
  - Image input  : flat (N, D) arrays reshaped to (N, C, H, W) before forward pass
  - weight_decay : emitted in optimizer call when non-zero
"""
from __future__ import annotations

_ACT = {
    "ReLU":      "nn.ReLU()",
    "Tanh":      "nn.Tanh()",
    "Sigmoid":   "nn.Sigmoid()",
    "LeakyReLU": "nn.LeakyReLU(0.01)",
    "Linear":    "nn.Identity()",
}


# ── Building blocks ───────────────────────────────────────────────────────────

def _seq_body(config: dict) -> str:
    """nn.Sequential layer lines, 12-space indent."""
    from components.shapes import compute_shapes
    is_image = config.get("input_shape") is not None
    shapes   = compute_shapes(config) if is_image else None
    lines    = []
    prev     = config["input_dim"]  # tabular width tracker

    for i, layer in enumerate(config["hidden_layers"]):
        ltype = layer.get("type", "Linear")

        if ltype == "Conv2d":
            in_c = shapes[i][0]
            lines.append(f"            nn.Conv2d({in_c}, {layer['out_channels']}, kernel_size={layer['kernel_size']}),")
            if layer.get("batch_norm"):
                lines.append(f"            nn.BatchNorm2d({layer['out_channels']}),")
            lines.append(f"            {_ACT[layer['activation']]},")

        elif ltype == "MaxPool2d":
            lines.append(f"            nn.MaxPool2d({layer['kernel_size']}),")

        elif ltype == "Flatten":
            lines.append( "            nn.Flatten(),")

        else:  # Linear
            in_units = shapes[i][0] if is_image else prev
            lines.append(f"            nn.Linear({in_units}, {layer['units']}),")
            if layer.get("batch_norm"):
                lines.append(f"            nn.BatchNorm1d({layer['units']}),")
            lines.append(f"            {_ACT[layer['activation']]},")
            if layer.get("dropout") and layer.get("dropout_rate", 0.0) > 0:
                lines.append(f"            nn.Dropout({layer['dropout_rate']}),")
            if not is_image:
                prev = layer["units"]

    # Output layer
    final_units = shapes[len(config["hidden_layers"])][0] if is_image else prev
    if config["task"] == "binary":
        lines.append(f"            nn.Linear({final_units}, 1),")
        lines.append( "            nn.Sigmoid(),")
    else:
        lines.append(f"            nn.Linear({final_units}, {config['n_classes']}),")
        lines.append( "            # CrossEntropyLoss applies log-softmax internally - no activation here")
    return "\n".join(lines)


def _criterion_line(config: dict) -> str:
    if config["task"] == "binary":
        return "    criterion = nn.BCELoss()"
    return "    criterion = nn.CrossEntropyLoss()"


def _optimizer_line(config: dict) -> str:
    opt, lr = config["optimizer"], config["lr"]
    wd      = config.get("weight_decay", 0.0)
    wd_str  = f", weight_decay={wd}" if wd > 0 else ""
    if opt == "Adam":
        return f"    optimizer = torch.optim.Adam(model.parameters(), lr={lr}{wd_str})"
    if opt == "Momentum":
        return f"    optimizer = torch.optim.SGD(model.parameters(), lr={lr}, momentum=0.9{wd_str})"
    return     f"    optimizer = torch.optim.SGD(model.parameters(), lr={lr}{wd_str})"


def _tensor_lines(config: dict) -> str:
    """Lines that build X / y tensors inside train(), 4-space indent."""
    if config["task"] == "binary":
        base = (
            "    X_tr = torch.tensor(X_train, dtype=torch.float32)\n"
            "    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)\n"
            "    X_v  = torch.tensor(X_val,   dtype=torch.float32)\n"
            "    y_v  = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)"
        )
    else:
        base = (
            "    X_tr = torch.tensor(X_train, dtype=torch.float32)\n"
            "    y_tr = torch.tensor(y_train, dtype=torch.long)\n"
            "    X_v  = torch.tensor(X_val,   dtype=torch.float32)\n"
            "    y_v  = torch.tensor(y_val,   dtype=torch.long)"
        )
    if config.get("input_shape") is not None:
        C, H, W = config["input_shape"]
        return base + (
            f"\n    # Reshape flat arrays to (N, C, H, W) for CNN input\n"
            f"    X_tr = X_tr.view(-1, {C}, {H}, {W})\n"
            f"    X_v  = X_v.view(-1,  {C}, {H}, {W})"
        )
    return base


def _epoch_loop(config: dict) -> str:
    """The full for-epoch block, 4-space base indent."""
    bs, epochs, task = config["batch_size"], config["epochs"], config["task"]

    if bs != "full":
        loader_setup = (
            f"    loader = DataLoader(\n"
            f"        TensorDataset(X_tr, y_tr), batch_size={bs}, shuffle=True,\n"
            f"    )\n\n"
        )
        inner = (
            f"        for X_batch, y_batch in loader:\n"
            f"            optimizer.zero_grad()\n"
            f"            out = model(X_batch)\n"
            f"            loss = criterion(out, y_batch)\n"
            f"            loss.backward()\n"
            f"            optimizer.step()"
        )
    else:
        loader_setup = ""
        inner = (
            f"        optimizer.zero_grad()\n"
            f"        out = model(X_tr)\n"
            f"        loss = criterion(out, y_tr)\n"
            f"        loss.backward()\n"
            f"        optimizer.step()"
        )

    if task == "binary":
        acc_tr = "            train_acc = ((train_out > 0.5).float() == y_tr).float().mean().item()"
        acc_v  = "            val_acc   = ((val_out   > 0.5).float() == y_v ).float().mean().item()"
    else:
        acc_tr = "            train_acc = (train_out.argmax(1) == y_tr).float().mean().item()"
        acc_v  = "            val_acc   = (val_out.argmax(1)   == y_v ).float().mean().item()"

    return (
        f"{loader_setup}"
        f"    for epoch in range({epochs}):\n"
        f"        model.train()\n"
        f"{inner}\n"
        f"\n"
        f"        model.eval()\n"
        f"        with torch.no_grad():\n"
        f"            train_out  = model(X_tr)\n"
        f"            train_loss = criterion(train_out, y_tr).item()\n"
        f"            val_out    = model(X_v)\n"
        f"            val_loss   = criterion(val_out,   y_v).item()\n"
        f"{acc_tr}\n"
        f"{acc_v}\n"
        f"\n"
        f'        history["train_loss"].append(train_loss)\n'
        f'        history["val_loss"].append(val_loss)\n'
        f'        history["train_acc"].append(train_acc)\n'
        f'        history["val_acc"].append(val_acc)'
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_code(config: dict) -> str:
    """Return a complete, runnable PyTorch script for the given config."""
    seed    = config["seed"]
    dataset = config["dataset"]

    bs_import = (
        "\nfrom torch.utils.data import DataLoader, TensorDataset"
        if config["batch_size"] != "full" else ""
    )

    return (
        f"import torch\n"
        f"import torch.nn as nn{bs_import}\n"
        f"import numpy as np\n"
        f"\n"
        f"torch.manual_seed({seed})\n"
        f"\n"
        f"\n"
        f"# -- Data --------------------------------------------------------------------\n"
        f"# from datasets.registry import load\n"
        f"# from sklearn.model_selection import train_test_split\n"
        f"#\n"
        f'# X_train, y_train, X_test, y_test = load("{dataset}", seed={seed})\n'
        f"# X_tr, X_v, y_tr_np, y_v_np = train_test_split(\n"
        f"#     X_train, y_train, test_size=0.2, random_state={seed}\n"
        f"# )\n"
        f"\n"
        f"\n"
        f"# -- Model -------------------------------------------------------------------\n"
        f"class Network(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.net = nn.Sequential(\n"
        f"{_seq_body(config)}\n"
        f"        )\n"
        f"\n"
        f"    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        f"        return self.net(x)\n"
        f"\n"
        f"\n"
        f"# -- Training ----------------------------------------------------------------\n"
        f"def train(\n"
        f"    model: Network,\n"
        f"    X_train: np.ndarray,\n"
        f"    y_train: np.ndarray,\n"
        f"    X_val:   np.ndarray,\n"
        f"    y_val:   np.ndarray,\n"
        f") -> dict:\n"
        f"{_criterion_line(config)}\n"
        f"{_optimizer_line(config)}\n"
        f"\n"
        f"{_tensor_lines(config)}\n"
        f"\n"
        f'    history = {{"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}}\n'
        f"\n"
        f"{_epoch_loop(config)}\n"
        f"\n"
        f"    return history\n"
        f"\n"
        f"\n"
        f"# -- Entry point -------------------------------------------------------------\n"
        f'if __name__ == "__main__":\n'
        f"    model = Network()\n"
        f"    print(model)\n"
        f"    # Load and split data, then:\n"
        f"    # history = train(model, X_tr, y_tr, X_v, y_v)\n"
    )
