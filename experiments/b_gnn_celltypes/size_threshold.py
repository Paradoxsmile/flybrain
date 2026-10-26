"""Cell-type accuracy as a function of the minimum type size (FAFB neurons per type).

The cell-type experiments label only FAFB types with >= 20 neurons (434 types, 78 % of the
neurons; configs/cell_type.yaml, min_class_size). Here the threshold is varied (--min-sizes) and,
per threshold, with the cell_type config otherwise unchanged:
- within: GNNs trained on FAFB, random split (seeds --seeds) and left -> right split (train and
  validate on the left hemisphere, test on the right; seeds --lr-seeds), scored on the test split;
- label_free: the random-split GNNs (the same models transfer.py would train) mapped to MCNS with
  label_free.py's pipeline (prior matching, selection by type consistency, NTAC), the MCNS labels
  coded in the FAFB label space of the threshold.
Both are broken down with stratify.py's strata: region of the type (FAFB super class) and FAFB
type size, with size bins extended below 20.

Below a threshold of 5 (thousands of classes), the GNNs train with lean_training.py (chunked output
layer, same results) and the label-free stage runs with lean_label_free.py (lazy probabilities and
chunked prior matching; label_free.py's dense N x K float64 matrices would need about 9 GB per seed
at threshold 3). Both agree with the standard code up to float rounding.

Writes results/size_threshold/m<size>/within_<split>/s<seed>/{metrics.json,predictions.parquet}
(random split also model.pt, in label_free.py's layout under m<size>/models/cell_type/gnn/),
m<size>/label_free/cell_type/* (as label_free.py) and strata.csv, types.csv, table.md over all
finished runs. Finished runs are skipped. With --target banc (only the label_free stage) or
--select size_agreement, the label-free run goes to m<size>/label_free_<target>_<select> instead
(e.g. label_free_banc_size_agreement; docs/banc_rare_protocol.md) and is not in the report.
--target-retention / --retention-seed / --target-crop / --target-exclude degrade the target as in
label_free.py (development sandbox, docs/candidate_selection_plan.md), e.g.
m2/label_free_mcns_r0.35_size_agreement.

Usage: python experiments/b_gnn_celltypes/size_threshold.py [--min-sizes 20 10 5 2]
       [--stages within label_free report] [--seeds 0 1 2] [--lr-seeds 0] [--epochs 300]
       python experiments/b_gnn_celltypes/size_threshold.py --min-sizes 2 --stages label_free
           --target banc --select size_agreement
       python experiments/b_gnn_celltypes/size_threshold.py --min-sizes 2 --stages label_free
           --select size_agreement --target-retention 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import label_free
import lean_label_free
import lean_training
import numpy as np
import pandas as pd
import stratify
import torch
import yaml

from flybrain.features import normalize_features
from flybrain.loading import load_graph
from flybrain.splits import make_split
from flybrain.training import fit_gnn, load_model, predict
from flybrain.utils import REPO_ROOT, get_device, set_seed

HERE = Path(__file__).parent
OUT = REPO_ROOT / "results" / "size_threshold"
LABEL = "cell_type"
SPLITS = {"random": "seeds", "left_right": "lr_seeds"}
SIZE_BINS = [(2, 4), (5, 9), (10, 19), (20, 49), (50, 199), (200, 999), (1000, np.inf)]
LEAN_BELOW = 5  # below: lean_training / lean_label_free (same results, O(chunk x K) memory)
REUSE = {"models": "transfer2", "seeds": (0, 1, 2), "label_free": "label_free"}  # threshold 20
STAGE_NAMES = {"no_adaptation": "no adaptation", "selected": "prior matched", "refined": "+ NTAC"}


def size_bin(n: float) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return f"≥ {lo}" if hi == np.inf else f"{lo}–{hi}"
    return "1"


def fafb_types() -> pd.DataFrame:
    t = stratify.fafb_types()
    t["size_bin"] = t["size"].map(size_bin)
    return t


def reuse_within(seed: int) -> None:
    """At the default threshold (20), the random-split GNNs of transfer.py (results/transfer2,
    same config) are the within-FAFB models: only their test predictions are computed."""
    d = OUT / "m20" / "within_random" / f"s{seed}"
    path = REPO_ROOT / "results" / REUSE["models"] / LABEL / "gnn" / f"s{seed}" / "model.pt"
    model, cfg, names = load_model(path)
    data = load_graph(LABEL, min_class_size=20)
    assert names == data.label_names and cfg["seed"] == seed
    train, _, test = make_split(data, "random", seed=seed)
    x = normalize_features(data.x, "global", train)
    pred = label_free.probabilities(model, cfg, data, x).argmax(axis=1)
    write_within(d, data, pred, train, test, cfg, None, f"reused {path}")


def write_within(d, data, pred, train, test, cfg, fit, note: str = "") -> None:
    names = np.array(data.label_names, dtype=object)
    t = test.numpy()
    y = data.y.numpy()
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"root_id": data.node_ids.numpy()[t], "label": names[y[t]], "pred": names[pred[t]]}
    ).to_parquet(d / "predictions.parquet", index=False)
    metrics = {
        "min_size": cfg["min_class_size"],
        "split": cfg["split"],
        "seed": cfg["seed"],
        "n_types": len(names),
        "n_train": int(train.sum()),
        "n_test": int(t.sum()),
        "acc": float((pred[t] == y[t]).mean()),
        "epochs": cfg["epochs"],
        "note": note,
    }
    if fit is not None:
        metrics["best_epoch"] = int(max(fit.history, key=lambda h: h["val_macro_f1"])["epoch"])
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[size_threshold] within {metrics}", flush=True)


def run_within(min_size: int, split: str, seed: int, epochs: int | None) -> None:
    d = OUT / f"m{min_size}" / f"within_{split}" / f"s{seed}"
    if (d / "metrics.json").exists():
        return
    if min_size == 20 and split == "random" and seed in REUSE["seeds"] and not epochs:
        return reuse_within(seed)
    cfg = yaml.safe_load((HERE / "configs" / "cell_type.yaml").read_text())
    cfg.update(seed=seed, min_class_size=min_size, split=split)
    if epochs:
        cfg["epochs"] = epochs
    set_seed(seed)
    device = get_device()
    data = load_graph(LABEL, seed=seed, min_class_size=min_size)
    train, val, test = make_split(data, split, seed=seed)
    start = time.time()
    lean = min_size < LEAN_BELOW
    fit = (lean_training.fit_gnn_lean if lean else fit_gnn)(cfg, data, train, val, device)
    seconds = time.time() - start
    pred = (lean_training.predict_lean if lean else predict)(fit.model, fit.x, fit.adj)
    if split == "random":  # label_free.py's layout: <models>/<label>/gnn/s<seed>/model.pt
        m = OUT / f"m{min_size}" / "models" / LABEL / "gnn" / f"s{seed}"
        m.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": fit.model.state_dict(), "cfg": cfg, "label_names": data.label_names},
            m / "model.pt",
        )
    write_within(d, data, pred, train, test, cfg, fit, f"{seconds:.0f} s on {device}")


DEGRADE = {"retention": 1.0, "retention_seed": 2, "crop": 0.0, "exclude": ()}  # full target


def label_free_dir(target: str, select: str, degrade: dict | None = None) -> str:
    """m<size>/<this>: label_free for the full MCNS with type consistency, otherwise
    label_free_<target>[_r<retention>[_seed<s>]][_crop<f>][_x]_<select>."""
    d = {**DEGRADE, **(degrade or {})}
    name = f"label_free_{target}"
    if d["retention"] < 1:
        name += f"_r{d['retention']:g}"
        name += f"_seed{d['retention_seed']}" if d["retention_seed"] != 2 else ""
    name += f"_crop{d['crop']:g}" if d["crop"] else ""
    name += "_x" if d["exclude"] else ""
    if name == "label_free_mcns" and select == "consistency":
        return "label_free"
    return f"{name}_{select}"


def run_label_free(
    min_size: int, target: str = "mcns", select: str = "consistency", degrade: dict | None = None
) -> None:
    d = {**DEGRADE, **(degrade or {})}
    out_dir = OUT / f"m{min_size}" / label_free_dir(target, select, d)
    if (out_dir / LABEL / "selected.json").exists():
        return
    if min_size == 20 and out_dir.name == "label_free":
        # the paper's run (results/label_free), with its stages from stratify.py
        pred = stratify.labelings(REUSE["label_free"], "fafb", "mcns")
        sel = json.loads(
            (REPO_ROOT / "results" / REUSE["label_free"] / LABEL / "selected.json").read_text()
        )
        (out_dir / LABEL).mkdir(parents=True, exist_ok=True)
        pred.rename(columns={"pred_refined": "pred"}).to_parquet(
            out_dir / LABEL / "predictions.parquet", index=False
        )
        sel["note"] = f"reused results/{REUSE['label_free']}"
        (out_dir / LABEL / "selected.json").write_text(json.dumps(sel, indent=2))
        return
    label_free.LABELS[LABEL] = min_size  # the FAFB label space of this threshold
    args = argparse.Namespace(
        source="fafb",
        target=target,
        models=f"size_threshold/m{min_size}/models",
        synthetic=False,
        score_all=False,
        select=select,
        target_retention=d["retention"],
        retention_seed=d["retention_seed"],
        target_exclude=list(d["exclude"]),
        max_weight=None,
        source_retention=1.0,
        target_coverage=None,
        target_crop=d["crop"],
        prior="source",
        coverage=None,
    )
    start = time.time()
    if min_size < LEAN_BELOW:
        with lean_label_free.patched():
            sel = label_free.run_label(LABEL, args, out_dir)
    else:
        sel = label_free.run_label(LABEL, args, out_dir)
    sel["seconds"] = time.time() - start
    (out_dir / LABEL / "selected.json").write_text(json.dumps(sel, indent=2))


def runs(types: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-type tables and strata of every finished run."""
    all_t, all_s = [], []
    for d in sorted(p for p in OUT.glob("m*") if p.is_dir()):
        min_size = int(d.name[1:])
        for split in SPLITS:
            files = sorted((d / f"within_{split}").glob("s*/predictions.parquet"))
            for f in files:
                pred = pd.read_parquet(f).rename(columns={"pred": "pred_test"})
                t = stratify.per_type(pred, "test", types)
                meta = dict(min_size=min_size, setting=f"FAFB {split}", seed=int(f.parent.name[1:]))
                all_t.append(t.assign(stage="test", **meta))
                all_s.append(stratify.strata(t).assign(stage="test", **meta))
        f = d / "label_free" / LABEL / "predictions.parquet"
        if f.exists():
            pred = pd.read_parquet(f).rename(columns={"pred": "pred_refined"})
            pred = pred[pred.label.notna()]
            for stage in STAGE_NAMES:
                t = stratify.per_type(pred, stage, types)
                meta = dict(min_size=min_size, setting="FAFB → MCNS", seed=-1)
                all_t.append(t.assign(stage=stage, **meta))
                all_s.append(stratify.strata(t).assign(stage=stage, **meta))
    return pd.concat(all_t, ignore_index=True), pd.concat(all_s, ignore_index=True)


def table(s: pd.DataFrame, kind: str, order: list) -> list:
    """Rows: setting x threshold (mean over seeds), columns: strata (accuracy % / macro-F1)."""
    s = s[s.kind.isin([kind, "all"]) & s.stage.isin(["test", "refined"])]
    g = s.groupby(["setting", "min_size", "stratum"])
    m = g[["acc", "macro_f1", "n_neurons", "n_types"]].mean()
    sd = g.acc.std()
    n_seeds = g.seed.nunique()
    cols = ["all"] + order
    lines = ["| setting | min size | seeds | " + " | ".join(cols) + " |"]
    lines.append("|---" * (len(cols) + 3) + "|")
    settings = sorted(s.setting.unique(), key=lambda x: (x.startswith("FAFB →"), x))
    for setting in settings:
        for min_size in sorted(s[s.setting == setting].min_size.unique(), reverse=True):
            key = (setting, min_size)
            cells = []
            for c in cols:
                if key + (c,) not in m.index:
                    cells.append("–")
                    continue
                r = m.loc[key + (c,)]
                e = sd.get(key + (c,))
                pm = (
                    f" ± {100 * e:.1f}"
                    if e == e and e is not None and n_seeds[key + (c,)] > 1
                    else ""
                )
                cells.append(
                    f"{100 * r.acc:.1f}{pm} / {r.macro_f1:.2f} "
                    f"<sub>{int(r.n_types)}t, {int(r.n_neurons):,}n</sub>"
                )
            n = n_seeds[key + ("all",)] if setting != "FAFB → MCNS" else "3 (ens.)"
            lines.append(f"| {setting} | {min_size} | {n} | " + " | ".join(cells) + " |")
    return lines


def report() -> None:
    types = fafb_types()
    t, s = runs(types)
    t.to_csv(OUT / "types.csv", index=False)
    s.to_csv(OUT / "strata.csv", index=False)
    regions = list(stratify.REGION_STYLE)
    bins = [size_bin(lo) for lo, _ in SIZE_BINS]
    lines = [
        "Accuracy % (± std over seeds) / macro-F1 by minimum FAFB type size; small print: types",
        "and test neurons of the stratum. FAFB rows: test split (mean over seeds). FAFB → MCNS:",
        "label-free pipeline after NTAC, labeled MCNS neurons. Region = FAFB super class of the",
        "true type. Macro-F1 = mean per-type F1 over the stratum's true types (stratify.py).",
        "",
        "### By region",
        "",
    ]
    lines += table(s, "region", regions)
    lines += ["", "### By FAFB type size", ""]
    lines += table(s, "type size", bins)
    lf = s[(s.setting == "FAFB → MCNS") & (s.stratum == "all")]
    if not lf.empty:
        lines += ["", "### FAFB → MCNS by pipeline stage (all labeled neurons)", ""]
        lines += ["| min size | " + " | ".join(STAGE_NAMES.values()) + " |", "|---" * 4 + "|"]
        for min_size in sorted(lf.min_size.unique(), reverse=True):
            r = lf[lf.min_size == min_size].set_index("stage")
            cells = [f"{100 * r.acc[k]:.1f} / {r.macro_f1[k]:.2f}" for k in STAGE_NAMES]
            lines.append(f"| {min_size} | " + " | ".join(cells) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-sizes", type=int, nargs="+", default=[20, 10, 5, 2])
    p.add_argument(
        "--stages",
        nargs="+",
        default=["within", "label_free", "report"],
        choices=["within", "label_free", "report"],
    )
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="random split")
    p.add_argument("--lr-seeds", type=int, nargs="*", default=[0], help="left -> right split")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--target", default="mcns", choices=["mcns", "banc"], help="label_free stage")
    p.add_argument("--select", default="consistency", choices=["consistency", "size_agreement"])
    p.add_argument("--target-retention", type=float, default=1.0, help="label_free: thin target")
    p.add_argument("--retention-seed", type=int, default=2, help="thinning realization")
    p.add_argument("--target-crop", type=float, default=0.0, help="share of cropped neurons")
    p.add_argument("--target-exclude", nargs="*", default=[], help="cell types to remove")
    args = p.parse_args()
    degrade = {
        "retention": args.target_retention,
        "retention_seed": args.retention_seed,
        "crop": args.target_crop,
        "exclude": tuple(args.target_exclude),
    }
    degraded = args.target_retention < 1 or args.target_crop > 0 or bool(args.target_exclude)
    if (args.target != "mcns" or degraded) and set(args.stages) != {"label_free"}:
        p.error("--target banc and a degraded target run only the label_free stage")
    for min_size in args.min_sizes:
        if "within" in args.stages:
            for split, attr in SPLITS.items():
                for seed in getattr(args, attr):
                    run_within(min_size, split, seed, args.epochs)
        if "label_free" in args.stages:
            run_label_free(min_size, args.target, args.select, degrade)
    if "report" in args.stages:
        report()


if __name__ == "__main__":
    main()
