"""GraphSAGE node classification (cell type / super class) from connectivity.

The model aggregates over presynaptic inputs and (if bidirectional) postsynaptic outputs,
optionally weighted by synapse count; see flybrain.models.DirectedSAGE.

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

from flybrain.features import degree_features, normalize_features
from flybrain.loading import load_graph
from flybrain.models import DirectedSAGE, MeanBuilder, class_weights, drop_edges
from flybrain.splits import make_split
from flybrain.utils import get_device, log_run, make_run_dir, set_seed, write_predictions
from flybrain.viz import plot_training_curves

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, adj: tuple, mask: torch.Tensor) -> tuple[float, float]:
    """Return (accuracy, macro-F1) on the masked nodes."""
    model.eval()
    pred = model(data.x, *adj).argmax(1)[mask].cpu()
    true = data.y[mask].cpu()
    return float((pred == true).float().mean()), float(f1_score(true, pred, average="macro"))


@torch.no_grad()
def save_predictions(model: nn.Module, data: Data, adj: tuple, masks: tuple, path: Path) -> None:
    """Write the model's predictions for every neuron (see flybrain.utils.write_predictions)."""
    model.eval()
    write_predictions(path, data, model(data.x, *adj).argmax(1).cpu().numpy(), masks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test graph")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--run-name", default=None, help="override config run_name")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set bidirectional=false --set seed=1 (repeatable)",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key not in cfg:
            p.error(f"--set expects KEY=VALUE with a key from {args.config.name}, got {item!r}")
        cfg[key] = yaml.safe_load(value)
    cfg["synthetic"] = args.synthetic
    set_seed(cfg["seed"])
    device = get_device()

    data = load_graph(
        cfg["label_col"],
        synthetic=args.synthetic,
        seed=cfg["seed"],
        min_class_size=cfg.get("min_class_size", 1),
    )
    train, val, test = make_split(data, cfg.get("split", "random"), seed=cfg["seed"])
    norm, side = cfg.get("feature_norm", "global"), getattr(data, "side", None)
    raw_x = data.x
    data.x = normalize_features(raw_x, norm, train, side)
    data = data.to(device)
    train_cpu = train
    train, val, test = train.to(device), val.to(device), test.to(device)

    bidirectional = cfg.get("bidirectional", True)
    ei, n, w = data.edge_index, data.num_nodes, data.edge_weight
    means = MeanBuilder(ei, n, w, cfg.get("edge_weight", "count"))
    adj = means(bidirectional)

    # augmentation: each epoch trains on a subgraph with randomly dropped edges, with degree
    # features recomputed from it (scaled like the full graph), so the model cannot rely on
    # exact degrees; evaluation always uses the full graph
    drop_rate = cfg.get("drop_edge", 0.0)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])

    def augmented() -> tuple[torch.Tensor, tuple]:
        keep = drop_edges(ei, n, drop_rate, gen)
        raw = degree_features(n, ei[:, keep].cpu(), w[keep].cpu())
        x = normalize_features(torch.from_numpy(raw).float(), norm, train_cpu, side, raw_x)
        return x.to(device), means(bidirectional, keep)

    num_classes = len(data.label_names)
    weight = class_weights(data.y[train], num_classes, cfg.get("class_weight", "none"))
    model = DirectedSAGE(
        data.x.size(1),
        cfg["hidden_dim"],
        num_classes,
        cfg["num_layers"],
        cfg["dropout"],
        bidirectional,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # select the epoch with the best validation macro-F1 (accuracy hides the rare classes)
    history, best_val, best_state = [], -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        x, train_adj = augmented() if drop_rate > 0 else (data.x, adj)
        out = model(x, *train_adj)
        loss = F.cross_entropy(out[train], data.y[train], weight=weight)
        loss.backward()
        opt.step()
        val_acc, val_f1 = evaluate(model, data, adj, val)
        history.append(
            {"epoch": epoch, "loss": loss.item(), "val_acc": val_acc, "val_macro_f1": val_f1}
        )
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics: dict[str, float | int] = {"num_nodes": int(data.num_nodes), "device": str(device)}
    for name, mask in (("train", train), ("val", val), ("test", test)):
        metrics[f"n_{name}"] = int(mask.sum())
    metrics["best_epoch"] = int(max(history, key=lambda h: h["val_macro_f1"])["epoch"])
    for name, mask in (("train", train), ("val", val), ("test", test)):
        acc, f1 = evaluate(model, data, adj, mask)
        metrics[f"{name}_acc"], metrics[f"{name}_macro_f1"] = acc, f1

    run_dir = make_run_dir(cfg.get("run_name"), "gnn")
    log_run(run_dir, cfg, metrics)
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    plot_training_curves(hist, run_dir / "curves.png")
    save_predictions(model, data, adj, (train, val, test), run_dir / "predictions.parquet")
    print(f"[gnn] {metrics}\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
