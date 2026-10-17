"""Controlled test of how synapse density limits transfer: FAFB against thinned copies of itself.

A thinned copy keeps each synapse with probability p and drops connections with fewer than 5
remaining synapses (flybrain.thinning); p = 0.4 gives about BANC's density, p = 0.6 about FAFB's
density relative to MCNS (Janelia's factor 0.581). All copies have the same neurons and labels,
so differences in accuracy come from density alone, not from individual or annotation
differences.

For every source density p_s, GNNs are trained on a thinned copy (realization seed 1; the full
graph for p_s = 1) with the label's config, one per seed (random split). Each is evaluated on
independently thinned target copies (realization seed 2) at every target density p_t, on its
test split, with the target's own feature normalization, without (rho 0) and with prior matching
(rho 1, sizes from the model's predictions on its training graph). p_t < p_s is a sparser target
(as BANC), p_t > p_s a denser one (as future, more complete reconstructions). For p_t > p_s the
target is also thinned down to the source density (retention p_s / p_t), a label-free way to
align a denser target. With --thin-aug p_min, the GNNs are instead trained with density
augmentation (flybrain.training: every epoch on a copy thinned to a random retention in
[p_min, 1]); --target-retentions sets the target densities separately.

Writes results/<out>/grid.csv (one row per label, p_s, p_t, strategy, rho, seed), stats.csv
(graph statistics per copy), the models (models/<label>/p<p_s>/s<seed>/model.pt; finished
models are reused) and summary.md (mean over seeds).

Usage: python experiments/b_gnn_celltypes/thinning.py [--labels cell_type] [--seeds 0 1 2]
       [--retentions 1 0.6 0.4 0.25] [--epochs N] [--synthetic] [--out thinning]
       [--thin-aug 0.25 --retentions 1 --target-retentions 1 0.6 0.4 0.25 --out thinning_aug]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from flybrain.adapt import prior_weights
from flybrain.features import normalize_features
from flybrain.loading import load_graph
from flybrain.splits import make_split
from flybrain.thinning import thin_graph
from flybrain.training import fit_gnn, graph_inputs, load_model
from flybrain.utils import REPO_ROOT, get_device, set_seed

HERE = Path(__file__).parent
CONFIGS = {
    "super_class": HERE / "configs/default.yaml",
    "cell_type": HERE / "configs/cell_type.yaml",
}
SOURCE_SEED, TARGET_SEED, ALIGN_SEED = 1, 2, 3  # thinning realizations


def copy(graph, p: float, seed: int):
    return graph if p == 1.0 else thin_graph(graph, p, seed=seed)


def stats(g) -> dict:
    src, dst = g.edge_index.numpy()
    w = g.edge_weight.numpy()
    n = g.num_nodes
    ind, outd = np.bincount(dst, minlength=n), np.bincount(src, minlength=n)
    return {
        "connections_per_neuron": len(w) / n,
        "synapses_per_neuron": float(w.sum() / n),
        "synapses_per_connection": float(w.mean()),
        "median_input_synapses": float(np.median(np.bincount(dst, w, n))),
        "median_input_partners": float(np.median(ind)),
        "isolated": float(((ind + outd) == 0).mean()),
    }


@torch.no_grad()
def probabilities(model, cfg, graph, x, device) -> np.ndarray:
    xx, adj = graph_inputs(graph, x, cfg, device)
    return torch.softmax(model(xx, *adj), dim=1).cpu().numpy()


def trained_model(label, p_s, seed, src, args, device):
    """Train (or reuse) the GNN for one source density and seed."""
    path = REPO_ROOT / "results" / args.out / "models" / label / f"p{p_s}" / f"s{seed}" / "model.pt"
    if path.exists():
        model, cfg, _ = load_model(path, device)
        return model, cfg
    cfg = yaml.safe_load(CONFIGS[label].read_text())
    cfg.update(seed=seed, retention=p_s, thin_aug=args.thin_aug)
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(seed)
    train, val, _ = make_split(src, "random", seed=seed)
    fit = fit_gnn(cfg, src, train, val, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"state_dict": fit.model.state_dict(), "cfg": cfg, "label_names": src.label_names}
    torch.save(state, path)
    print(f"[thinning] trained {label} p_s={p_s} s{seed}", flush=True)
    return fit.model.eval(), cfg


def run_label(label: str, args, device) -> tuple[list, list]:
    msize = yaml.safe_load(CONFIGS[label].read_text()).get("min_class_size", 1)
    full = load_graph(label, synthetic=args.synthetic, min_class_size=msize)
    sources = {p: copy(full, p, SOURCE_SEED) for p in args.retentions}
    targets = {p: copy(full, p, TARGET_SEED) for p in args.target_retentions or args.retentions}
    stat_rows = [
        {"label": label, "copy": role, "p": p, **stats(g)}
        for role, graphs in (("source", sources), ("target", targets))
        for p, g in graphs.items()
    ]
    y = full.y.numpy()
    rows = []
    for (p_s, src), seed in itertools.product(sources.items(), args.seeds):
        model, cfg = trained_model(label, p_s, seed, src, args, device)
        train, _, test = make_split(src, "random", seed=seed)
        t = test.numpy()
        ref = probabilities(model, cfg, src, normalize_features(src.x, "global", train), device)
        for p_t, tgt in targets.items():
            cases = {"as_is": tgt}
            if p_t > p_s:
                cases["thinned_to_source"] = thin_graph(tgt, p_s / p_t, seed=ALIGN_SEED)
            for strategy, g in cases.items():
                p = probabilities(model, cfg, g, normalize_features(g.x, "global"), device)
                weights = prior_weights(p, ref.mean(axis=0) * len(p))
                for rho in (0.0, 1.0):
                    pred = (p * weights**rho).argmax(axis=1)
                    rows.append(
                        {
                            "label": label,
                            "p_source": p_s,
                            "p_target": p_t,
                            "strategy": strategy,
                            "rho": rho,
                            "seed": seed,
                            "acc": float((pred[t] == y[t]).mean()),
                            "macro_f1": float(f1_score(y[t], pred[t], average="macro")),
                        }
                    )
        print(f"[thinning] evaluated {label} p_s={p_s} s{seed}", flush=True)
    return rows, stat_rows


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def summary_lines(grid: pd.DataFrame, stat: pd.DataFrame) -> list[str]:
    lines = ["Thinned FAFB copies: accuracy % on the test split, mean over seeds.", ""]
    mean = grid.groupby(["label", "strategy", "rho", "p_source", "p_target"]).acc.mean() * 100
    for label in grid.label.unique():
        for rho in (0.0, 1.0):
            m = mean[label, "as_is", rho].unstack("p_target")
            ps = sorted(m.columns, reverse=True)
            lines += [f"{label}, rho {rho:g} (rows: source density, columns: target density):", ""]
            sources = sorted(m.index, reverse=True)
            rows = [[f"{p_s:g}"] + [f"{m.loc[p_s, p]:.1f}" for p in ps] for p_s in sources]
            lines += table([r"p_s \ p_t"] + [f"{p:g}" for p in ps], rows) + [""]
        if "thinned_to_source" not in set(grid[grid.label == label].strategy):
            continue
        lines += [f"{label}, denser target thinned to the source density:", ""]
        rows = []
        for rho, p_s, p_t in mean[label, "thinned_to_source"].index:
            if rho == 0.0:
                cells = [
                    mean[label, s, r, p_s, p_t]
                    for r in (0.0, 1.0)
                    for s in ("as_is", "thinned_to_source")
                ]
                rows.append([f"{p_s:g}", f"{p_t:g}"] + [f"{c:.1f}" for c in cells])
        header = ["p_s", "p_t", "as is, rho 0", "thinned, rho 0", "as is, rho 1", "thinned, rho 1"]
        lines += table(header, rows) + [""]
    lines += ["Graph statistics of the copies:", ""]
    rows = [
        [f"{v:.3g}" if isinstance(v, float) else str(v) for v in r] for r in stat.values.tolist()
    ]
    return lines + table(list(stat.columns), rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", default=["cell_type"], choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--retentions", nargs="+", type=float, default=[1.0, 0.6, 0.4, 0.25])
    p.add_argument("--target-retentions", nargs="+", type=float, default=None)
    p.add_argument("--thin-aug", type=float, default=0.0, help="density augmentation p_min")
    p.add_argument("--epochs", type=int, default=None, help="override the config (for tests)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="thinning")
    args = p.parse_args()
    device = get_device()
    out_dir = REPO_ROOT / "results" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, stat_rows = [], []
    for label in args.labels:
        r, s = run_label(label, args, device)
        rows += r
        stat_rows += s
    grid, stat = pd.DataFrame(rows), pd.DataFrame(stat_rows)
    grid.to_csv(out_dir / "grid.csv", index=False)
    stat.to_csv(out_dir / "stats.csv", index=False)
    lines = summary_lines(grid, stat)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
