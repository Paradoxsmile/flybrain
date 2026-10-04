"""GraphSAGE node classification (cell type / super class) from connectivity.

Usage: python experiments/b_gnn_celltypes/train_gnn.py [--config configs/default.yaml] [--synthetic]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from flybrain.loading import load_graph
from flybrain.splits import stratified_split
from flybrain.utils import get_device, log_run, make_run_dir, set_seed
from flybrain.viz import plot_training_curves

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"


class GraphSAGE(nn.Module):
    """Stack of SAGEConv layers with ReLU + dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, self.training)
        return self.convs[-1](x, edge_index)


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, mask: torch.Tensor) -> tuple[float, float]:
    """Return (accuracy, macro-F1) on the masked nodes."""
    model.eval()
    pred = model(data.x, data.edge_index).argmax(1)[mask].cpu()
    true = data.y[mask].cpu()
    return float((pred == true).float().mean()), float(f1_score(true, pred, average="macro"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test graph")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["synthetic"] = args.synthetic
    set_seed(cfg["seed"])
    device = get_device()

    data = load_graph(
        cfg["label_col"],
        synthetic=args.synthetic,
        seed=cfg["seed"],
        min_class_size=cfg.get("min_class_size", 1),
    )
    train, val, test = stratified_split(data.y, seed=cfg["seed"])
    mean, std = data.x[train].mean(0), data.x[train].std(0).clamp(min=1e-6)
    data.x = (data.x - mean) / std
    data = data.to(device)
    train, val, test = train.to(device), val.to(device), test.to(device)

    model = GraphSAGE(
        data.x.size(1), cfg["hidden_dim"], len(data.label_names), cfg["num_layers"], cfg["dropout"]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history, best_val, best_state = [], -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(data.x, data.edge_index)[train], data.y[train])
        loss.backward()
        opt.step()
        val_acc, _ = evaluate(model, data, val)
        history.append({"epoch": epoch, "loss": loss.item(), "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics: dict[str, float | int] = {"num_nodes": int(data.num_nodes), "device": str(device)}
    for name, mask in (("train", train), ("val", val), ("test", test)):
        acc, f1 = evaluate(model, data, mask)
        metrics[f"{name}_acc"], metrics[f"{name}_macro_f1"] = acc, f1

    run_dir = make_run_dir(cfg.get("run_name"), "gnn")
    log_run(run_dir, cfg, metrics)
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    plot_training_curves(hist, run_dir / "curves.png")
    print(f"[gnn] {metrics}\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
