"""Stage-1 candidate selection for the two-stage rare-type pipeline (plan and rules:
docs/candidate_selection_plan.md).

The two-stage pipeline (graph_matching_mcns.py) sends to graph matching the target neurons whose
stage-1 type (label-free GNN at minimum type size 2, after NTAC) is a 2-4-neuron FAFB type. This
script builds other candidate sets from the same stage-1 run, label-free, and scores them against
the target labels (development targets only: MCNS and its degraded copies).

--stage mass: the rare-type probability mass of every target neuron, m_i = sum over 2-4-neuron
types of the stage-1 ensemble (mean over seeds of the prior-matched probabilities at the chosen
variant, normalization and rho, before NTAC; its argmax is predictions.parquet's pred_selected).
The GNNs are run once on the chosen target graph (lean_label_free.LazyProbs); the prior-matching
weights come from the run's prior_weights.npz or, for the FAFB -> MCNS run that predates it, from
graph_matching_mcns.py's soft_gnn.npz, so the Sinkhorn is not repeated. Writes rare_mass.parquet.

--stage select: candidate sets, per hemisphere where they use a count (neurons without a side
cannot be matched and are never added):
- gnn: stage-1 type after NTAC is 2-4 (the current pipeline);
- selected: prior-matched type before NTAC is 2-4; union: gnn or selected;
- mass: the K_side neurons of highest mass, K_side = target neurons on that side x (FAFB rare-type
  neurons / FAFB labeled neurons on that side), the expected number of rare-type neurons;
- mass_fill: gnn plus the highest-mass other neurons of the side up to K_side;
- mass_at_gnn: as mass with K_side = gnn's count on that side (equal-count comparison);
- band9: stage-1 type after NTAC is 2-9 (diagnostic; matching would need the 5-9 FAFB rows).
Writes candidates_<set>.parquet (root ids, read by graph_matching_mcns.py --candidates
file:<set>), selection.csv (counts, K, recall, precision, composition) and selection_curve.csv
(recall of the mass ranking against the count) next to the stage-1 predictions.

--stage twotier: idea 2 on saved matches. Every neuron the matching did not assign a FAFB neuron
takes the coarse labeling's type where its stage-1 type is in the coarse label space
(graph_matching_mcns.two_tier); writes results/candidate_selection/<stage1>/twotier_<matches>.csv.

Usage: python experiments/b_gnn_celltypes/candidate_selection.py --stage mass [--stage1 label_free]
       python experiments/b_gnn_celltypes/candidate_selection.py --stage select
       python experiments/b_gnn_celltypes/candidate_selection.py --stage twotier
           --matches graph_matching_mcns/matches_gnn_gm_bary.parquet
           --fallback size_threshold/m10/label_free label_free
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import graph_matching_mcns as gmm
import label_free
import lean_label_free
import numpy as np
import pandas as pd
import stratify

from flybrain.splits import make_split
from flybrain.training import load_model
from flybrain.utils import REPO_ROOT

DEVELOPMENT = {"mcns"}  # BANC is spent for rare types (docs/banc_rare_protocol.md)
CACHED_WEIGHTS = {"label_free": "graph_matching_mcns"}  # runs before prior_weights.npz
SIDES = gmm.SIDES
SETS = ("gnn", "selected", "union", "mass", "mass_fill", "mass_at_gnn", "band9")
WRITTEN = ("union", "mass", "mass_fill", "mass_at_gnn")  # sets graph matching can read


def stage1(run: str) -> tuple:
    d = gmm.STAGE1_ROOT / run / "cell_type"
    sel = json.loads((d / "selected.json").read_text())
    if sel["target"] not in DEVELOPMENT:
        raise SystemExit(f"{run}: target {sel['target']} is not a development target")
    return d, sel, pd.read_parquet(d / "predictions.parquet")


def prior_weights(d, sel: dict, run: str) -> list:
    """The chosen candidate's prior-matching weights, one array per seed (model_paths order)."""
    c = sel["chosen"]
    f = d / "prior_weights.npz"
    if f.exists():
        return list(np.load(f)[label_free.weights_key(c["variant"], c["norm"])])
    z = np.load(REPO_ROOT / "results" / CACHED_WEIGHTS[run] / "soft_gnn.npz")
    return list(z["weights"])


def size_cols(names: np.ndarray, sizes: pd.Series, lo: int, hi: int) -> np.ndarray:
    s = pd.Series(names).map(sizes).fillna(0).to_numpy()
    return np.nonzero((s >= lo) & (s <= hi))[0]


def mass(args) -> None:
    d, sel, pred = stage1(args.stage1)
    chosen = sel["chosen"]
    label_free.LABELS["cell_type"] = gmm.MIN_SIZE
    fafb, tgt = gmm.graphs(sel["target"], chosen["variant"], gmm.degrade_args(sel))
    assert (pred.root_id.to_numpy() == tgt.node_ids.numpy()).all()
    weights = prior_weights(d, sel, args.stage1)
    paths = label_free.model_paths(gmm.MODELS, "cell_type")
    assert len(paths) == len(weights)
    sizes = stratify.fafb_types()["size"]
    names = np.array(tgt.label_names, dtype=object)
    rare, mid = size_cols(names, sizes, 2, 4), size_cols(names, sizes, 5, 9)
    t = time.time()
    probs = []
    for path in paths:
        model, cfg, nm = load_model(path)
        assert nm == fafb.label_names == tgt.label_names
        train = make_split(fafb, "random", seed=cfg["seed"])[0]
        x = label_free.target_features(tgt, fafb, train, chosen["norm"], None)
        probs.append(lean_label_free.LazyProbs(model, cfg, tgt, x))
    print(f"GNN features: {time.time() - t:.0f} s", flush=True)
    factors = [(v ** chosen["rho"]).astype(np.float32) for v in weights]
    n = len(pred)
    out = {k: np.empty(n, dtype=np.float32) for k in ("mass_2_4", "mass_5_9", "max_prob")}
    out.update({f"mass_2_4_s{i}": np.empty(n, dtype=np.float32) for i in range(len(probs))})
    ens = np.empty(n, dtype=np.int64)
    top_rare = np.empty(n, dtype=np.int64)
    chunk = probs[0].chunk
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        total = None
        for i, (p, f) in enumerate(zip(probs, factors, strict=True)):
            q = p.rows(start, stop) * f  # label_free.adjusted, as lean_label_free.adjusted
            q /= q.sum(axis=1, keepdims=True)
            out[f"mass_2_4_s{i}"][start:stop] = q[:, rare].sum(axis=1)
            total = q if total is None else total + q
        ens[start:stop] = total.argmax(axis=1)
        total /= len(probs)
        out["mass_2_4"][start:stop] = total[:, rare].sum(axis=1)
        out["mass_5_9"][start:stop] = total[:, mid].sum(axis=1)
        out["max_prob"][start:stop] = total.max(axis=1)
        top_rare[start:stop] = rare[total[:, rare].argmax(axis=1)]
    agree = float((names[ens] == pred.pred_selected.to_numpy()).mean())
    print(f"mass: {time.time() - t:.0f} s, agreement with pred_selected {agree:.6f}", flush=True)
    assert agree > 0.999, "recomputed ensemble differs from stage 1"
    df = pred[["root_id", "side", "label", "pred", "pred_selected"]].assign(
        **out, top_rare=names[top_rare]
    )
    df.to_parquet(d / "rare_mass.parquet", index=False)
    print(df.describe().T.to_string(), flush=True)


def expected_share() -> dict:
    """Per side: FAFB rare-type (2-4) neurons / FAFB neurons labeled in the label space (>= 2)."""
    t = stratify.fafb_types()["size"]
    n = pd.read_parquet(REPO_ROOT / "data" / "processed" / "neurons.parquet")
    s = n.cell_type.map(t).fillna(0)
    n = n[s >= gmm.MIN_SIZE]
    s = s[s >= gmm.MIN_SIZE]
    rare = (s <= gmm.MAX_SIZE).to_numpy()
    return {side: float(rare[n.side == side].sum() / (n.side == side).sum()) for side in SIDES}


def top_by_side(score: np.ndarray, side: np.ndarray, k: dict, base: np.ndarray) -> np.ndarray:
    """base plus, per side, the highest-score neurons of that side not in base, until the side
    has k[side] selected neurons (base unchanged where it already has that many)."""
    sel = base.copy()
    for s in SIDES:
        free = np.nonzero((side == s) & ~base)[0]
        need = k[s] - int((base & (side == s)).sum())
        if need > 0:
            sel[free[np.argsort(-score[free], kind="stable")[:need]]] = True
    return sel


def candidate_sets(m: pd.DataFrame, sizes: pd.Series, share: dict) -> tuple[dict, dict]:
    side = m.side.to_numpy(dtype=object)
    mass_ = m.mass_2_4.to_numpy()
    gnn = gmm.rare(m.pred, sizes)
    selected = gmm.rare(m.pred_selected, sizes)
    k = {s: round((side == s).sum() * share[s]) for s in SIDES}
    k_gnn = {s: int((gnn & (side == s)).sum()) for s in SIDES}
    none = np.zeros(len(m), bool)
    ps = m.pred.map(sizes).fillna(0).to_numpy()
    sets = {
        "gnn": gnn,
        "selected": selected,
        "union": gnn | selected,
        "mass": top_by_side(mass_, side, k, none),
        "mass_fill": top_by_side(mass_, side, k, gnn),
        "mass_at_gnn": top_by_side(mass_, side, k_gnn, none),
        "band9": (ps >= 2) & (ps <= 9),
    }
    return sets, k


def summarize(name: str, c: np.ndarray, m: pd.DataFrame, sizes: pd.Series, k: dict) -> dict:
    label = m.label.to_numpy(dtype=object)
    lab = pd.notna(label)
    rare = lab & gmm.rare(label, sizes)
    gnn = gmm.rare(m.pred, sizes)
    ps = m.pred.map(sizes).fillna(0).to_numpy()
    missed59 = rare & ~gnn & (ps >= 5) & (ps <= 9)
    side = m.side.to_numpy(dtype=object)
    row = {"set": name, "n": int(c.sum())}
    row.update({s: int((c & (side == s)).sum()) for s in SIDES})
    row["no_side"] = int((c & ~np.isin(side, SIDES)).sum())
    row.update({f"K_{s}": k[s] for s in SIDES})
    row["recall"] = float((c & rare).sum() / rare.sum())
    row["precision"] = float((c & rare).sum() / c.sum())
    row["other_type"] = float((c & lab & ~rare).sum() / c.sum())
    row["unlabeled"] = float((c & ~lab).sum() / c.sum())
    row["gained_rare"] = int((c & rare & ~gnn).sum())
    row["lost_rare"] = int((~c & rare & gnn).sum())
    row["recovered_missed_5_9"] = float((c & missed59).sum() / max(missed59.sum(), 1))
    return row


def select(args) -> None:
    d, sel, _ = stage1(args.stage1)
    m = pd.read_parquet(d / "rare_mass.parquet")
    sizes = stratify.fafb_types()["size"]
    share = expected_share()
    print({"expected rare share": share}, flush=True)
    sets, k = candidate_sets(m, sizes, share)
    rows = [summarize(name, sets[name], m, sizes, k) for name in SETS]
    table = pd.DataFrame(rows)
    table.to_csv(d / "selection.csv", index=False)
    print(table.to_string(index=False), flush=True)
    for name in WRITTEN:
        m.loc[sets[name], ["root_id"]].to_parquet(d / f"candidates_{name}.parquet", index=False)
    # recall of the mass ranking against the per-side count, as a multiple of K_side
    side = m.side.to_numpy(dtype=object)
    none = np.zeros(len(m), bool)
    curve = []
    for f in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0):
        c = top_by_side(m.mass_2_4.to_numpy(), side, {s: round(f * k[s]) for s in SIDES}, none)
        curve.append({"factor": f, **summarize(f"mass x{f:g}", c, m, sizes, k)})
    pd.DataFrame(curve).to_csv(d / "selection_curve.csv", index=False)
    print(pd.DataFrame(curve)[["factor", "n", "recall", "precision"]].to_string(index=False))


def twotier(args) -> None:
    d, sel, pred = stage1(args.stage1)
    sizes = stratify.fafb_types()["size"]
    types = stratify.fafb_types()
    matches = pd.read_parquet(REPO_ROOT / "results" / args.matches)
    idx = pd.Series(np.arange(len(pred)), index=pred.root_id).loc[matches.root_id].to_numpy()
    truth = pred.label.to_numpy(dtype=object)
    stage1_pred = pred.pred.to_numpy(dtype=object)
    two = stage1_pred.copy()
    two[idx] = matches.pred.to_numpy(dtype=object)
    assigned = np.zeros(len(pred), bool)  # older matches: every candidate counts as assigned
    if "fafb_root_id" in matches:
        assigned[idx] = matches.fafb_root_id.to_numpy() >= 0
    else:
        assigned[idx] = True
    matched = np.zeros(len(pred), bool)
    matched[idx] = True
    references = [(gmm.REFERENCE[sel["target"]], truth)]
    region = pd.Series(truth).map(types.region).to_numpy(dtype=object)
    labelings = [("stage 1", stage1_pred, np.zeros(len(pred), bool)), ("two-stage", two, assigned)]
    root_ids = pred.root_id.to_numpy()
    rows = []
    for name, p, a in labelings:
        variants = [(name, p)]
        for run in args.fallback:
            coarse, gate = gmm.load_fallback(run, root_ids, sizes)
            variants.append(
                (f"{name} + two-tier m{gate}", gmm.two_tier(p, a, stage1_pred, coarse, gate, sizes))
            )
        for method, q in variants:
            for row in gmm.score_rows(q, references, [], sizes, types, matched):
                rows.append({"method": method, **row})
            for r in sorted(pd.unique(region[pd.notna(region)])):
                k = region == r
                rows.append(
                    {
                        "method": method,
                        "reference": references[0][0],
                        "subset": f"all labeled, {r}",
                        **gmm.score(truth, q, k, types),
                    }
                )
    out = REPO_ROOT / "results" / "candidate_selection" / args.stage1
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    stem = args.matches.replace("\\", "/").split("/")[-1].removesuffix(".parquet")
    table.to_csv(out / f"twotier_{stem}.csv", index=False)
    wide = table.assign(
        cell=lambda t: (100 * t.acc).round(1).astype(str) + " / " + t.macro_f1.round(3).astype(str)
    )
    print(wide.pivot(index="subset", columns="method", values="cell").to_string())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", choices=["mass", "select", "twotier"], required=True)
    ap.add_argument("--stage1", default="label_free", help="run under results/size_threshold/m2")
    ap.add_argument("--matches", help="twotier: matches_*.parquet under results/")
    ap.add_argument("--fallback", nargs="*", default=[], help="twotier: coarse runs under results/")
    args = ap.parse_args()
    if args.stage == "twotier" and not args.matches:
        ap.error("--stage twotier needs --matches")
    {"mass": mass, "select": select, "twotier": twotier}[args.stage](args)


if __name__ == "__main__":
    main()
