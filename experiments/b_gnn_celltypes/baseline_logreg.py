"""Logistic regression on degree / synapse-count features (no graph structure beyond degrees).

Usage: python experiments/b_gnn_celltypes/baseline_logreg.py [--synthetic] [--label-col ...]
"""

from __future__ import annotations

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from flybrain.loading import load_graph
from flybrain.splits import stratified_split
from flybrain.utils import log_run, make_run_dir, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test graph")
    p.add_argument("--label-col", default="super_class", choices=["super_class", "cell_type"])
    p.add_argument("--min-class-size", type=int, default=1, help="smaller classes are unlabeled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    data = load_graph(
        args.label_col,
        synthetic=args.synthetic,
        seed=args.seed,
        min_class_size=args.min_class_size,
    )
    train, val, test = stratified_split(data.y, seed=args.seed)
    x, y = data.x.numpy(), data.y.numpy()

    scaler = StandardScaler().fit(x[train])
    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(scaler.transform(x[train]), y[train])

    metrics: dict[str, float | int] = {"num_nodes": int(data.num_nodes)}
    for name, mask in (("train", train), ("val", val), ("test", test)):
        pred = clf.predict(scaler.transform(x[mask]))
        metrics[f"{name}_acc"] = float(accuracy_score(y[mask], pred))
        metrics[f"{name}_macro_f1"] = float(f1_score(y[mask], pred, average="macro"))

    run_dir = make_run_dir(args.run_name, "logreg")
    log_run(run_dir, vars(args), metrics)
    print(f"[logreg] {metrics}\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
