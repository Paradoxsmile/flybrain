"""GraphSAGE node classification (cell type / super class) from connectivity.

The model aggregates over presynaptic inputs and (if bidirectional) postsynaptic outputs,
optionally weighted by synapse count; see flybrain.models.DirectedSAGE.

Usage: python experiments/b_gnn_celltypes/train_gnn.py [--config configs/default.yaml] [--synthetic]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from flybrain.loading import load_graph
from flybrain.splits import make_split
from flybrain.training import evaluate, fit_gnn, predict
from flybrain.utils import get_device, log_run, make_run_dir, set_seed, write_predictions
from flybrain.viz import plot_training_curves

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"


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
        dataset=cfg.get("dataset", "fafb"),
        mcns_variant=cfg.get("mcns_variant", "density"),
    )
    train, val, test = make_split(data, cfg.get("split", "random"), seed=cfg["seed"])
    fit = fit_gnn(cfg, data, train, val, device)

    metrics: dict[str, float | int] = {"num_nodes": int(data.num_nodes), "device": str(device)}
    for name, mask in (("train", train), ("val", val), ("test", test)):
        metrics[f"n_{name}"] = int(mask.sum())
    metrics["best_epoch"] = int(max(fit.history, key=lambda h: h["val_macro_f1"])["epoch"])
    for name, mask in (("train", train), ("val", val), ("test", test)):
        acc, f1 = evaluate(fit.model, fit.x, fit.adj, fit.y, mask.to(device))
        metrics[f"{name}_acc"], metrics[f"{name}_macro_f1"] = acc, f1

    run_dir = make_run_dir(cfg.get("run_name"), "gnn")
    log_run(run_dir, cfg, metrics)
    hist = pd.DataFrame(fit.history)
    hist.to_csv(run_dir / "history.csv", index=False)
    plot_training_curves(hist, run_dir / "curves.png")
    pred = predict(fit.model, fit.x, fit.adj)
    write_predictions(run_dir / "predictions.parquet", data, pred, (train, val, test))
    print(f"[gnn] {metrics}\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
