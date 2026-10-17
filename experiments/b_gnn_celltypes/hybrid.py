"""GNN-seeded NTAC: label neurons that have no labeled partners.

NTAC needs labeled neurons in the same graph; the GNN does not, but is less accurate. Here the
GNN predictions of several seeds seed NTAC: neurons on which all GNNs agree become NTAC seeds
(together with any truly known labels), NTAC labels the rest. Nothing is tuned: seeds are the
unanimous predictions, NTAC runs with the package defaults (15 steps, learning rate 0.3).

Scenarios (GNN predictions are read from earlier runs):
  transfer           MCNS brain, `density` alignment, MCNS-normalized GNNs trained on FAFB
                     (results/transfer); no MCNS label is known
  <dataset>-<split>  hemisphere split within FAFB or MCNS (results/benchmark, benchmark_mcns);
                     labels of the training hemisphere are known, the other one is scored

Writes results/<out>/<scenario>/<label>/metrics.json and predictions.parquet, and summary.md.

Usage: python experiments/b_gnn_celltypes/hybrid.py [--scenarios transfer fafb-right_left]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from flybrain.loading import load_graph
from flybrain.ntac import agreement_seeds, ntac_from_predictions
from flybrain.splits import make_split
from flybrain.utils import REPO_ROOT, log_run, write_predictions

LABELS = {"super_class": 1, "cell_type": 20}  # min_class_size, as in the configs
SCENARIOS = ["transfer"] + [
    f"{d}-{s}" for d in ("fafb", "mcns") for s in ("left_right", "right_left")
]
BENCH = {"fafb": "benchmark", "mcns": "benchmark_mcns"}
NTAC_LR, NTAC_STEPS = 0.3, 15


def score(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(accuracy, macro-F1) on the masked neurons."""
    return float((pred[mask] == y[mask]).mean()), float(
        f1_score(y[mask], pred[mask], average="macro")
    )


def scenario_inputs(scenario: str, label: str, results):
    """(graph, known labels or -1, evaluation mask, GNN prediction files) of a scenario."""
    if scenario == "transfer":
        g = load_graph(label, min_class_size=LABELS[label], dataset="mcns", mcns_variant="density")
        files = sorted(
            (results / "transfer" / label / "gnn").glob("s*/predictions-density-target.parquet")
        )
        return g, np.full(g.num_nodes, -1), g.y.numpy() >= 0, files
    dataset, split = scenario.split("-")
    g = load_graph(label, min_class_size=LABELS[label], dataset=dataset)
    train, val, test = (m.numpy() for m in make_split(g, split, seed=0))
    y = g.y.numpy()
    known = np.where(train | val, y, -1)  # the whole labeled training hemisphere
    files = sorted(
        (results / BENCH[dataset] / label / split / "gnn").glob("s*/predictions.parquet")
    )
    return g, known, test, files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--labels", nargs="+", default=list(LABELS), choices=list(LABELS))
    p.add_argument("--out", default="hybrid")
    args = p.parse_args()

    results = REPO_ROOT / "results"
    rows = []
    for scenario in args.scenarios:
        for label in args.labels:
            g, known, ev, files = scenario_inputs(scenario, label, results)
            if len(files) < 2:
                print(
                    f"[hybrid] {scenario} {label}: needs >= 2 GNN predictions, found {len(files)}"
                )
                continue
            idx = {name: i for i, name in enumerate(g.label_names)}
            preds = []
            for f in files:
                df = pd.read_parquet(f)
                assert (df.root_id.to_numpy() == g.node_ids.numpy()).all(), (
                    f"{f}: node order differs"
                )
                preds.append(df.pred.map(idx).to_numpy(dtype=np.int64))
            preds = np.stack(preds)
            y = g.y.numpy()
            vote, agree = agreement_seeds(preds)
            labels, seeds = ntac_from_predictions(
                g.edge_index, g.edge_weight, known, preds, len(g.label_names), NTAC_LR, NTAC_STEPS
            )
            single = [score(pr, y, ev) for pr in preds]
            m = {
                "n_models": len(files),
                "gnn_acc": float(np.mean([s[0] for s in single])),
                "gnn_macro_f1": float(np.mean([s[1] for s in single])),
                "seed_coverage": float(agree[ev].mean()),
                "seed_acc": score(vote, y, ev & agree)[0],
            }
            m["vote_acc"], m["vote_macro_f1"] = score(vote, y, ev)
            m["hybrid_acc"], m["hybrid_macro_f1"] = score(labels, y, ev)
            m["hybrid_nonseed_acc"] = score(labels, y, ev & ~agree)[0]
            run_dir = results / args.out / scenario / label
            run_dir.mkdir(parents=True, exist_ok=True)
            log_run(
                run_dir,
                {
                    "scenario": scenario,
                    "label": label,
                    "files": [str(f) for f in files],
                    "ntac_lr": NTAC_LR,
                    "ntac_steps": NTAC_STEPS,
                },
                m,
            )
            none = torch.zeros(g.num_nodes, dtype=torch.bool)
            masks = (torch.from_numpy(seeds & ~ev), none, torch.from_numpy(ev))
            write_predictions(run_dir / "predictions.parquet", g, labels, masks)
            rows.append({"scenario": scenario, "label": label, **m})
            print(f"[hybrid] {scenario} {label}: {json.dumps(m)}", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        lines = ["Accuracy (%) / macro-F1 on the scored neurons. GNN: mean of single models.", ""]
        lines += [
            "| scenario | label | GNN | GNN vote | seeds (share, acc) | GNN + NTAC |",
            "|---" * 6 + "|",
        ]
        for r in df.itertuples():
            lines.append(
                f"| {r.scenario} | {r.label} | {100 * r.gnn_acc:.1f} / {r.gnn_macro_f1:.2f} | "
                f"{100 * r.vote_acc:.1f} / {r.vote_macro_f1:.2f} | "
                f"{100 * r.seed_coverage:.0f} %, {100 * r.seed_acc:.1f} | "
                f"**{100 * r.hybrid_acc:.1f} / {r.hybrid_macro_f1:.2f}** |"
            )
        (results / args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
