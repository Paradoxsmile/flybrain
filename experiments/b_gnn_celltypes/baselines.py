"""Baselines that use the labels of other neurons, for comparison with the GNN.

majority:        the most frequent training class.
label_prop:      label spreading over the synapse graph; assumes partners share a label.
                 Optionally with class mass normalization against the pull of large classes.
partner_profile: logistic regression on the share of a neuron's input and output synapses per
                 partner class (partners with training labels only); the information that
                 connectivity-based cell typing uses.

Hyperparameters are chosen by validation macro-F1 (alpha and class mass normalization for
label_prop; C and class weighting for partner_profile); the choice is logged with the metrics.

Usage: python experiments/b_gnn_celltypes/baselines.py --method partner_profile
       [--label-col cell_type --min-class-size 20] [--split left_right] [--synthetic]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from flybrain.baselines import (
    class_mass_normalize,
    edge_homophily,
    label_propagation,
    partner_profiles,
    propagation_matrix,
)
from flybrain.loading import load_graph
from flybrain.splits import SPLITS, make_split
from flybrain.utils import log_run, make_run_dir, set_seed, write_predictions

METHODS = ("majority", "label_prop", "partner_profile")
LP_ALPHAS = (0.1, 0.3, 0.5, 0.8, 0.9, 0.99)
PROFILE_CS = (1.0, 10.0, 100.0)
PROFILE_CLASS_WEIGHTS = (None, "balanced")


def scores(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(accuracy, macro-F1) on the masked neurons."""
    return float((pred[mask] == y[mask]).mean()), float(
        f1_score(y[mask], pred[mask], average="macro")
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--label-col", default="super_class", choices=["super_class", "cell_type"])
    p.add_argument("--min-class-size", type=int, default=1, help="smaller classes are unlabeled")
    p.add_argument("--split", default="random", choices=SPLITS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test graph")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    data = load_graph(
        args.label_col,
        synthetic=args.synthetic,
        seed=args.seed,
        min_class_size=args.min_class_size,
    )
    masks = make_split(data, args.split, seed=args.seed)
    train, val, test = (m.numpy() for m in masks)
    y, k = data.y.numpy(), len(data.label_names)
    chosen: dict[str, float | str | None] = {}

    if args.method == "majority":
        pred = np.full(len(y), np.bincount(y[train], minlength=k).argmax())

    elif args.method == "label_prop":
        h, chance = edge_homophily(data.edge_index, data.edge_weight, data.y)
        chosen.update(edge_homophily=h, edge_homophily_chance=chance)
        s = propagation_matrix(data.edge_index, data.edge_weight, data.num_nodes)
        best = (-1.0, None, None)
        for alpha in LP_ALPHAS:
            f = label_propagation(s, data.y, masks[0], k, alpha)
            for cmn in (False, True):
                g = class_mass_normalize(f, data.y, masks[0]) if cmn else f
                cand = g.argmax(1).numpy()
                f1 = scores(y, cand, val)[1]
                print(f"[label_prop] alpha={alpha}, cmn={cmn}: val macro-F1 {f1:.4f}", flush=True)
                if f1 > best[0]:
                    best = (f1, (alpha, cmn), cand)
        _, (chosen["alpha"], chosen["class_mass_norm"]), pred = best

    else:  # partner_profile
        x = partner_profiles(data.edge_index, data.edge_weight, data.y, masks[0], k)
        best = (-1.0, None, None)
        for c, cw in itertools.product(PROFILE_CS, PROFILE_CLASS_WEIGHTS):
            clf = LogisticRegression(C=c, class_weight=cw, max_iter=1000)
            clf.fit(x[train], y[train])
            cand = clf.predict(x)
            f1 = scores(y, cand, val)[1]
            print(f"[partner_profile] C={c}, class_weight={cw}: val macro-F1 {f1:.4f}", flush=True)
            if f1 > best[0]:
                best = (f1, (c, cw), cand)
        _, (chosen["C"], chosen["class_weight"]), pred = best

    metrics: dict[str, float | int | str | None] = {"num_nodes": int(data.num_nodes), **chosen}
    for name, mask in (("train", train), ("val", val), ("test", test)):
        metrics[f"n_{name}"] = int(mask.sum())
        metrics[f"{name}_acc"], metrics[f"{name}_macro_f1"] = scores(y, pred, mask)

    run_dir = make_run_dir(args.run_name, f"baseline_{args.method}")
    log_run(run_dir, vars(args), metrics)
    write_predictions(run_dir / "predictions.parquet", data, pred, masks)
    print(f"[{args.method}] {metrics}\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
