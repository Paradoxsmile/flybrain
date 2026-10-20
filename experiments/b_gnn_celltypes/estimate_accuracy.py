"""Can the accuracy of a label-free labeling be predicted without target labels?

Run level. Every candidate labeling of label_free.py (candidates.csv) has label-free columns
(mean pairwise agreement between the seeds' labelings, Jensen-Shannon divergence of the
predicted type sizes from the reference, type consistency on the scoring graph, share of the
largest predicted type) and its true accuracy. A few-parameter linear predictor of accuracy from
the label-free columns is fitted per label (cell type, super class) on the candidates of some
target connectomes, every run weighted equally, and evaluated on the others:
  development -> held-out  fit on the FAFB and MCNS targets (FAFB <-> MCNS and the development
                           sandbox: thinned, type-excluded and cropped / partial MCNS copies),
                           predict BANC and the hemibrain;
  leave one connectome out every rotation over the four target connectomes.
Reported: the absolute error of the predicted accuracy of the candidate each run chose (its
selected.json), and over all candidates of a run. Predictors:
  mean        mean accuracy of the training candidates (no label-free information)
  agree0      accuracy := pairwise agreement (no fitting)
  agree       a + b * agreement
  agree_js    a + b * agreement + c * size_js (the pre-specified main predictor)
  all         a + b * agreement + c * size_js + d * consistency + e * top_share / ref_top_share

Type and neuron level (cell types; the five transfers of stratify.py). The per-seed labelings of
the chosen candidate are not saved, so the saved GNNs (3 seeds) are applied once more with the
chosen alignment, normalization and rho (no training; cached in neurons_<run>.parquet and checked
against stages.parquet / selected.json). For the final (+ NTAC) labeling:
  type level   per predicted type, the mean share of seeds that agree with the final label over
               the neurons given that type (label-free), against the type's F1 on the labeled
               target neurons (Spearman, types with >= 10 neurons);
  neuron level the labeled neurons ranked by confidence (votes: seeds agreeing with the final
               label, ties broken by the ensemble probability of that label; prob: the ensemble
               probability alone); the share of all errors among the 10 / 20 / 30 % least
               confident, against random review (= the share reviewed) and an oracle.

Writes results/accuracy_estimation/{candidates.csv, run_level.csv, types.csv, flagging.csv,
tables.md}.

Usage: python experiments/b_gnn_celltypes/estimate_accuracy.py [--skip-neurons]
"""

from __future__ import annotations

import argparse
import json

import label_free
import numpy as np
import pandas as pd
import stratify
from scipy.stats import spearmanr

from flybrain.coverage import coverage_weights
from flybrain.selection import choose, scoring_column
from flybrain.utils import REPO_ROOT

RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "accuracy_estimation"
LABEL = "cell_type"
# label_free.py runs with the agreement columns, by target connectome. Left out as duplicates:
# label_free, label_free_rev, label_free_fafb_banc, label_free_mcns_banc (the same candidates as
# the _r1 / _cols runs, written before the agreement columns) and
# label_free_fafb_mcns_r1_central_brain_crop0.16_prior-coverage (a subset of
# label_free_partial_covnorm).
CANDIDATE_RUNS = {
    "MCNS": [
        "label_free_fafb_mcns_r1",
        "label_free_fafb_mcns_r0.5",
        "label_free_fafb_mcns_r0.2",
        "label_free_fafb_mcns_r0.5_x",
        "label_free_fafb_mcns_r0.5_x_w10",
        "label_free_fafb_mcns_r1_central_brain_crop0.16",
        "label_free_fafb_mcns_r1_central_brain_crop0.16_prior-em",
        "label_free_partial_covnorm",
        "label_free_partial_covnorm_nocrop",
    ],
    "FAFB": ["label_free_mcns_fafb_r1", "label_free_mcns_fafb_r0.35"],
    "BANC": [
        "label_free_fafb_banc_cols",
        "label_free_mcns_banc_cols",
        "label_free_mcns_banc_cols_super",
        "label_free_fafb_banc_aug",
        "label_free_fafb_banc_r1_s0.4",
    ],
    "hemibrain": ["label_free_hemibrain", "label_free_hemibrain_source_prior"],
}
DEVELOPMENT = ("MCNS", "FAFB")
FEATURES = {
    "mean": [],
    "agree": ["pairwise_agreement"],
    "agree_js": ["pairwise_agreement", "size_js"],
    "all": ["pairwise_agreement", "size_js", "consistency", "top_ratio"],
}
MODELS = ["mean", "agree0", "agree", "agree_js", "all"]
REVIEW = (0.1, 0.2, 0.3)
MIN_TYPE = 10


# ---------------------------------------------------------------- run level


def candidates() -> pd.DataFrame:
    rows = []
    for group, runs in CANDIDATE_RUNS.items():
        for run in runs:
            for path in sorted((RESULTS / run).glob("*/candidates.csv")):
                c = pd.read_csv(path)
                sel = json.loads((path.parent / "selected.json").read_text())
                ch = sel["chosen"]
                chosen = (c.variant == ch["variant"]) & (c.norm == ch["norm"])
                chosen &= np.isclose(c.rho, ch["rho"])
                assert chosen.sum() == 1 and np.isclose(c.acc[chosen].iloc[0], sel["selected_acc"])
                rows.append(
                    c.assign(
                        consistency=c[scoring_column(c)],
                        top_ratio=c.top_share / c.ref_top_share,
                        chosen=chosen,
                        run=run,
                        label=path.parent.name,
                        group=group,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def fit(train: pd.DataFrame, model: str) -> np.ndarray:
    """Weighted least squares coefficients (intercept first), every run weighted equally."""
    cols = FEATURES[model]
    w = 1 / train.groupby("run").acc.transform("size").to_numpy()
    X = np.column_stack([np.ones(len(train))] + [train[k] for k in cols])
    return np.linalg.lstsq(X * np.sqrt(w)[:, None], train.acc * np.sqrt(w), rcond=None)[0]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    if model == "agree0":
        return test.pairwise_agreement.to_numpy()
    Xt = np.column_stack([np.ones(len(test))] + [test[k] for k in FEATURES[model]])
    return np.clip(Xt @ fit(train, model), 0, 1)


def run_level(c: pd.DataFrame) -> pd.DataFrame:
    splits = [("development -> held-out", list(DEVELOPMENT), ["BANC", "hemibrain"])]
    groups = list(CANDIDATE_RUNS)
    splits += [("leave one out", [g for g in groups if g != h], [h]) for h in groups]
    rows = []
    for split, train_groups, test_groups in splits:
        for label, cl in c.groupby("label"):
            train = cl[cl.group.isin(train_groups)]
            test = cl[cl.group.isin(test_groups)]
            for model in MODELS:
                pred = fit_predict(train, test, model)
                for run, t in test.assign(pred=pred).groupby("run", sort=False):
                    ch = t[t.chosen].iloc[0]
                    sa = t.loc[choose(t, "size_agreement", key="consistency")]
                    rows.append(
                        {
                            "split": split,
                            "group": ch.group,
                            "run": run,
                            "label": label,
                            "model": model,
                            "acc": ch.acc,
                            "pred": ch.pred,
                            "err": ch.pred - ch.acc,
                            "mae_all": (t.pred - t.acc).abs().mean(),
                            "best": t.acc.max(),
                            "pred_choice_acc": t.acc[t.pred.idxmax()],
                            "acc_sa": sa.acc,
                            "err_sa": sa.pred - sa.acc,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- type and neuron level


def seed_labelings(directory: str, source: str, target: str) -> pd.DataFrame:
    """Per target neuron: label, the final (+ NTAC) and selected labelings, each seed's labeling
    and the ensemble probability of the final and selected labels, for the chosen candidate."""
    cache = OUT / f"neurons_{directory}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = RESULTS / directory / LABEL
    sel = json.loads((d / "selected.json").read_text())
    args = stratify.run_args(sel, source, target)
    ch = sel["chosen"]
    cases, _, _ = label_free.setups(LABEL, args)
    _, paths, src, tgt = next(c for c in cases if c[0] == ch["variant"])
    covered = {}
    if args.coverage:
        covered[id(src)] = coverage_weights(src.node_ids.numpy(), args.coverage)
    print(f"[estimate_accuracy] {directory}: applying the saved GNNs", flush=True)
    probs, weights, _ = label_free.seed_outputs(paths, src, tgt, ch["norm"], args, covered)
    total = np.zeros_like(probs[0])
    seeds = []
    for p, v in zip(probs, weights, strict=True):
        q = p * (v ** ch["rho"]).astype(np.float32)
        q /= q.sum(axis=1, keepdims=True)
        total += q
        seeds.append(q.argmax(axis=1))
    total /= len(probs)
    del probs
    names = np.array(tgt.label_names, dtype=object)
    code = {n: i for i, n in enumerate(names)}
    out = pd.DataFrame({"root_id": tgt.node_ids.numpy()})
    pred = pd.read_parquet(d / "predictions.parquet")[["root_id", "label", "pred"]]
    out = out.merge(pred, on="root_id", how="left", validate="1:1")
    refined = out.pred.map(code).to_numpy().astype(int)
    selected = total.argmax(axis=1)
    rows = np.arange(len(out))
    out = out.rename(columns={"pred": "pred_refined"})
    out["pred_selected"] = names[selected]
    out["p_refined"] = total[rows, refined]
    out["p_selected"] = total[rows, selected]
    for i, s in enumerate(seeds):
        out[f"seed{i}"] = names[s]
    stages = d / "stages.parquet"
    if stages.exists():
        st = pd.read_parquet(stages).set_index("root_id").pred_selected
        same = (st.loc[out.root_id].to_numpy() == out.pred_selected).mean()
        assert same > 0.9999, f"{directory}: {same:.6f} of the selected labels reproduced"
    lab = out.label.notna()
    acc = (out.pred_selected[lab] == out.label[lab]).mean()
    assert abs(acc - sel["selected_acc"]) < 1e-6, (acc, sel["selected_acc"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def votes(n: pd.DataFrame, stage: str) -> np.ndarray:
    seeds = [k for k in n.columns if k.startswith("seed")]
    return np.mean([n[k].to_numpy() == n[f"pred_{stage}"].to_numpy() for k in seeds], axis=0)


def type_level(n: pd.DataFrame, run: str, stage: str = "refined") -> pd.DataFrame:
    p = n[f"pred_{stage}"]
    agree = pd.Series(votes(n, stage)).groupby(p.to_numpy()).agg(["mean", "size"])
    lab = n[n.label.notna()]
    hit = lab[f"pred_{stage}"] == lab.label
    t = pd.DataFrame(
        {
            "n_true": lab.label.value_counts(),
            "n_pred": lab[f"pred_{stage}"].value_counts(),
            "correct": lab.label[hit].value_counts(),
        }
    ).fillna(0)
    t["f1"] = 2 * t.correct / (t.n_true + t.n_pred)
    t = t.join(agree.rename(columns={"mean": "agreement", "size": "n_all_pred"}), how="inner")
    return t.rename_axis("cell_type").reset_index().assign(run=run, stage=stage)


def flagging(n: pd.DataFrame, run: str, stage: str) -> list:
    lab = n[n.label.notna()].reset_index(drop=True)
    err = (lab[f"pred_{stage}"] != lab.label).to_numpy()
    v = votes(lab, stage)
    p = lab[f"p_{stage}"].to_numpy()
    rng = np.random.default_rng(0)
    scores = {"votes": v + 1e-3 * p, "prob": p, "random": rng.random(len(lab))}
    rows = []
    for name, s in scores.items():
        order = np.argsort(s, kind="stable")
        for f in REVIEW:
            k = int(round(f * len(lab)))
            rows.append(
                {
                    "run": run,
                    "stage": stage,
                    "score": name,
                    "review": f,
                    "n": len(lab),
                    "error_rate": err.mean(),
                    "caught": err[order[:k]].sum() / err.sum(),
                    "oracle": min(1.0, k / err.sum()),
                    "precision": err[order[:k]].mean(),
                }
            )
    # label-free estimate of the accuracy: share of neurons on which all seeds agree
    rows.append(
        {
            "run": run,
            "stage": stage,
            "score": "unanimous_share",
            "n": len(lab),
            "error_rate": err.mean(),
            "caught": float((v == 1).mean()),
            "precision": float((~err[v == 1]).mean()),
        }
    )
    return rows


# ---------------------------------------------------------------- tables


def pts(x: float) -> str:
    return f"{100 * x:+.1f}" if x == x else "–"


def tables(
    c: pd.DataFrame, rl: pd.DataFrame, types: pd.DataFrame | None, flags: pd.DataFrame | None
) -> list:
    lines = ["# Label-free accuracy estimation: tables", ""]
    lines += ["## Run level: absolute error of the predicted accuracy of the chosen candidate", ""]
    lines += ["Points; mean / worst over the test runs; in brackets: runs within 5 points.", ""]
    head = ["split", "test", "label", "runs"] + MODELS
    lines += ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for (split, group, label), g in rl.groupby(["split", "group", "label"], sort=False):
        cells = [split, group, label, str(g.run.nunique())]
        for m in MODELS:
            e = 100 * g[g.model == m].err.abs()
            cells.append(f"{e.mean():.1f} / {e.max():.1f} ({(e <= 5).sum()})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Pooled over the held-out runs of each split (mean / worst |error|, points):", ""]
    lines += ["| split | " + " | ".join(MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for split, g in rl.groupby("split", sort=False):
        cells = []
        for m in MODELS:
            e = 100 * g[g.model == m].err.abs()
            cells.append(f"{e.mean():.1f} / {e.max():.1f}")
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The same at the candidate the size_agreement rule chooses (the rule chosen on the "
        "sandbox; mean / worst |error|, points):",
        "",
    ]
    lines += ["| split | " + " | ".join(MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for split, g in rl.groupby("split", sort=False):
        cells = []
        for m in MODELS:
            e = 100 * g[g.model == m].err_sa.abs()
            cells.append(f"{e.mean():.1f} / {e.max():.1f}")
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines += ["", "Mean |error| over all candidates of the held-out runs (points):", ""]
    lines += ["| split | " + " | ".join(MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for split, g in rl.groupby("split", sort=False):
        cells = [f"{100 * g[g.model == m].mae_all.mean():.1f}" for m in MODELS]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines += ["", "Coefficients fitted on the development targets (MCNS, FAFB):", ""]
    dev = c[c.group.isin(DEVELOPMENT)]
    for label, g in dev.groupby("label"):
        for m in ("agree", "agree_js", "all"):
            beta = fit(g, m)
            terms = [f"{beta[0]:+.3f}"] + [
                f"{b:+.3f} {k}" for b, k in zip(beta[1:], FEATURES[m], strict=True)
            ]
            lines.append(f"- {label}, {m}: acc = {' '.join(terms)}")
    lines += ["", "### Per run (agree_js and all)", ""]
    head = ["split", "run", "label", "true acc", "agree_js", "error", "all", "error"]
    head += ["MAE all cand. (agree_js / all)", "best"]
    lines += ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    a = rl[rl.model == "agree_js"].reset_index(drop=True)
    b = rl[rl.model == "all"].reset_index(drop=True)
    for r, s in zip(a.itertuples(), b.itertuples(), strict=True):
        cells = [r.split, r.run, r.label, f"{100 * r.acc:.1f}", f"{100 * r.pred:.1f}"]
        cells += [pts(r.err), f"{100 * s.pred:.1f}", pts(s.err)]
        cells += [f"{100 * r.mae_all:.1f} / {100 * s.mae_all:.1f}", f"{100 * r.best:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "### Choosing the candidate with the highest predicted accuracy", ""]
    lines += [
        "Gap to the best candidate (points, mean / worst) of the run's own choice and of the "
        "argmax of each predictor, development -> held-out:",
        "",
    ]
    d = rl[rl.split == "development -> held-out"]
    own = 100 * (d.best - d.acc)[d.model == "mean"]
    lines += [
        "| label | run's rule | size_agreement | " + " | ".join(MODELS[1:]) + " |",
        "|---" * (len(MODELS) + 2) + "|",
    ]
    for label, g in d.groupby("label"):
        o = own[g[g.model == "mean"].index]
        sa = 100 * (g.best - g.acc_sa)[g.model == "mean"]
        cells = [f"{o.mean():.1f} / {o.max():.1f}", f"{sa.mean():.1f} / {sa.max():.1f}"]
        for m in MODELS[1:]:
            x = 100 * (g.best - g.pred_choice_acc)[g.model == m]
            cells.append(f"{x.mean():.1f} / {x.max():.1f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if types is not None:
        lines += ["", "## Type level: per predicted type, seed agreement vs F1 (+ NTAC)", ""]
        lines += [
            "| run | types (>= 10 neurons) | Spearman rho | F1 of the least / most agreed "
            "quarter |",
            "|---|---|---|---|",
        ]
        for run, t in types.groupby("run", sort=False):
            t = t[(t.n_all_pred >= MIN_TYPE) & (t.n_true + t.n_pred > 0)]
            r = spearmanr(t.agreement, t.f1).statistic
            q = pd.qcut(t.agreement.rank(method="first"), 4, labels=False)
            lo, hi = t.f1[q == 0].mean(), t.f1[q == 3].mean()
            lines.append(f"| {run} | {len(t)} | {r:.2f} | {lo:.2f} / {hi:.2f} |")
    if flags is not None:
        lines += [
            "",
            "## Neuron level: share of errors caught by reviewing the least confident",
            "",
        ]
        for stage in ("refined", "selected"):
            f = flags[flags.stage == stage]
            lines += [
                f"Labeling: {stage} ({'+ NTAC' if stage == 'refined' else 'prior matched'})."
                " Caught % (precision %: errors among the reviewed).",
                "",
            ]
            head = ["run", "labeled", "error %"]
            head += [f"{s} {int(100 * r)} %" for s in ("votes", "prob") for r in REVIEW]
            head += [f"oracle {int(100 * r)} %" for r in REVIEW]
            lines += ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
            for run, g in f.groupby("run", sort=False):
                g = g[g.score != "unanimous_share"].set_index(["score", "review"])
                cells = [run, f"{int(g.n.iloc[0]):,}", f"{100 * g.error_rate.iloc[0]:.1f}"]
                for s in ("votes", "prob"):
                    for r in REVIEW:
                        x = g.loc[(s, r)]
                        cells.append(f"{100 * x.caught:.0f} ({100 * x.precision:.0f})")
                cells += [f"{100 * g.loc[('votes', r)].oracle:.0f}" for r in REVIEW]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        u = flags[flags.score == "unanimous_share"]
        lines += [
            "Label-free per-neuron estimate: share of neurons where all seeds agree with the "
            "label, and the accuracy on them.",
            "",
        ]
        lines += [
            "| run | stage | true acc % | unanimous % | acc on unanimous % |",
            "|---|---|---|---|---|",
        ]
        for r in u.itertuples():
            lines.append(
                f"| {r.run} | {r.stage} | {100 * (1 - r.error_rate):.1f} | "
                f"{100 * r.caught:.1f} | {100 * r.precision:.1f} |"
            )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-neurons", action="store_true", help="run level only")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = candidates()
    c.to_csv(OUT / "candidates.csv", index=False)
    rl = run_level(c)
    rl.to_csv(OUT / "run_level.csv", index=False)
    types = flags = None
    if not args.skip_neurons:
        all_types, all_flags = [], []
        for run, (directory, source, target) in stratify.RUNS.items():
            n = seed_labelings(directory, source, target)
            all_types.append(type_level(n, run))
            for stage in ("refined", "selected"):
                all_flags += flagging(n, run, stage)
        types = pd.concat(all_types, ignore_index=True)
        flags = pd.DataFrame(all_flags)
        types.to_csv(OUT / "types.csv", index=False)
        flags.to_csv(OUT / "flagging.csv", index=False)
    text = "\n".join(tables(c, rl, types, flags)) + "\n"
    (OUT / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
