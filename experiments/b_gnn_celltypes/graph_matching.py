"""Graph-matching baseline for the rarest cell types (2-4 FAFB neurons per type).

At a minimum type size of 2, the GNN trained on the left FAFB hemisphere types right-hemisphere
neurons of 2-4-neuron types at about 76 % (size_threshold.py, left -> right split). Most of these
types are bilateral pairs, defined by matching neurons across hemispheres (Schlegel et al., 2024),
so the task is close to pairing homologs, which graph matching does without labels (bisected
graph matching; Pedigo, Winding, Priebe & Vogelstein, 2023). This script asks how far graph
matching gets on the same neurons.

Matching problem. Left: every left neuron of a 2-4-neuron type (the candidates, with their
labels); right: every right neuron of such a type. Knowing which right neurons belong to rare
types is an oracle restriction the GNN does not get (it chooses among all 8,582 types), and the
matching sees only the edges among these neurons (synapse counts, log1p), not the rest of the
brain. A doubly stochastic P (left x right, padded to square) maximizes, by Frank-Wolfe (FAQ;
Vogelstein et al., 2015),
    ipsilateral    <A_LL, P A_RR P^T>
    contralateral  <A_LR, P A_RL P>   (bisected; left -> right edges mirror right -> left)
then is projected to a permutation. Each right neuron gets the type of its matched left neuron.

Methods, all scored on the right-hemisphere neurons of 2-4-neuron types that the GNN's
left -> right test set contains (seeds 0-2 of results/size_threshold/m2/within_left_right):
- gm_ipsi:    graph matching, ipsilateral edges only, barycenter start;
- gm_bisected: ipsilateral + contralateral edges, barycenter start;
- gm_gnn_init: bisected, started from the GNN's predictions (P0 ~ number of seeds that predict a
  left neuron's type for a right neuron, mixed with the barycenter, Sinkhorn-normalized);
- gnn:        the GNN's own predictions (mean over 3 seeds);
- gnn_hungarian: the GNN votes restricted to the rare candidates and made one-to-one by a linear
  assignment (same oracle restriction as graph matching, no connectivity).

Writes results/graph_matching/{metrics.csv, per_region.csv, matches.parquet, info.json}.

Usage: python experiments/b_gnn_celltypes/graph_matching.py [--iters 30] [--max-size 4]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import stratify
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

from flybrain.loading import load_graph
from flybrain.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "graph_matching"
GNN_RUNS = REPO_ROOT / "results" / "size_threshold" / "m2" / "within_left_right"
SEEDS = (0, 1, 2)


def subgraph(data, left: np.ndarray, right: np.ndarray, n: int) -> dict[str, sp.csr_matrix]:
    """Sparse log1p(synapse) adjacencies among the selected neurons, padded to n x n:
    LL, RR (ipsilateral), LR (left -> right), RL (right -> left)."""
    src, dst = data.edge_index.numpy()
    w = np.log1p(data.edge_weight.numpy().astype(np.float64))
    pos_l = np.full(data.num_nodes, -1)
    pos_r = np.full(data.num_nodes, -1)
    pos_l[left] = np.arange(len(left))
    pos_r[right] = np.arange(len(right))

    def block(pa, pb):
        a, b = pa[src], pb[dst]
        keep = (a >= 0) & (b >= 0)
        return sp.csr_matrix((w[keep], (a[keep], b[keep])), shape=(n, n))

    return {
        "LL": block(pos_l, pos_l),
        "RR": block(pos_r, pos_r),
        "LR": block(pos_l, pos_r),
        "RL": block(pos_r, pos_l),
    }


def quad(adj: dict, d1: np.ndarray, d2: np.ndarray, bisected: bool) -> float:
    """Bilinear form of f: <LL, d1 RR d2^T> (+ <LR, d1 RL d2>), with sparse x dense products only
    (<X, M d2^T> = <X d2, M> and <X, M d2> = <X d2^T, M>)."""
    LL, RR, LR, RL = adj["LL"], adj["RR"], adj["LR"], adj["RL"]
    val = float(((LL @ d2) * (RR.T @ d1.T).T).sum())  # (d1 RR) .* (LL d2)
    if bisected:
        val += float(((LR @ d2.T) * (RL.T @ d1.T).T).sum())  # (d1 RL) .* (LR d2^T)
    return val


def gradient(adj: dict, p: np.ndarray, bisected: bool) -> np.ndarray:
    """Gradient of f(P) = <LL, P RR P^T> + <LR, P RL P> (contralateral term if bisected)."""
    LL, RR, LR, RL = adj["LL"], adj["RR"], adj["LR"], adj["RL"]
    g = LL @ (RR @ p.T).T + LL.T @ (RR.T @ p.T).T  # LL P RR^T + LL^T P RR
    if bisected:  # LR P^T RL^T + RL^T P^T LR
        g += LR @ (RL @ p).T + RL.T @ (LR.T @ p).T
    return g


def sinkhorn(m: np.ndarray, iters: int = 50) -> np.ndarray:
    m = m.copy()
    for _ in range(iters):
        m /= m.sum(axis=1, keepdims=True)
        m /= m.sum(axis=0, keepdims=True)
    return m


def faq(adj: dict, p0: np.ndarray, bisected: bool, iters: int, tol: float = 1e-4) -> np.ndarray:
    """Frank-Wolfe on the relaxed (doubly stochastic) problem, then projection to a permutation.
    Returns perm with perm[r] = matched left index of right index r."""
    p = p0.copy()
    for it in range(iters):
        g = gradient(adj, p, bisected)
        rows, cols = linear_sum_assignment(g, maximize=True)
        q = np.zeros_like(p)
        q[rows, cols] = 1.0
        d = q - p
        # f(P + a D) = f(P) + a <G, D> + a^2 f(D); maximize on [0, 1]
        c1 = float((g * d).sum())
        c2 = quad(adj, d, d, bisected)
        if c2 < 0:
            a = min(1.0, max(0.0, -c1 / (2 * c2)))
        else:
            a = 1.0 if c1 + c2 > 0 else 0.0
        p += a * d
        print(f"    iter {it}: step {a:.3f}, <G, D> {c1:.1f}", flush=True)
        if a * np.abs(d).max() < tol:
            break
    rows, cols = linear_sum_assignment(p, maximize=True)
    perm = np.empty(p.shape[1], dtype=int)
    perm[cols] = rows
    return perm


def gnn_votes(test_ids: np.ndarray) -> pd.DataFrame:
    """Per right test neuron and seed, the GNN's predicted type (results of size_threshold.py)."""
    preds = []
    for s in SEEDS:
        p = pd.read_parquet(GNN_RUNS / f"s{s}" / "predictions.parquet")
        preds.append(p.set_index("root_id").pred.rename(f"s{s}"))
    v = pd.concat(preds, axis=1)
    return v.loc[test_ids]


def score(truth: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "acc": float((truth == pred).mean()),
        "macro_f1": float(f1_score(truth, pred, labels=np.unique(truth), average="macro")),
    }


def two_stage(data, label: np.ndarray, side: np.ndarray, types: pd.DataFrame, args) -> None:
    """Label-free candidates: the GNN types every right neuron (majority of 3 seeds); the right
    neurons it assigns to a 2-4-neuron type are matched by bisected graph matching to the left
    neurons of such types, all other neurons keep the GNN's type. Scored on all right test
    neurons and on the rare ones of the oracle run."""
    ids = data.node_ids.numpy()
    preds = [
        pd.read_parquet(GNN_RUNS / f"s{s}" / "predictions.parquet").set_index("root_id")
        for s in SEEDS
    ]
    common = preds[0].index.intersection(preds[1].index).intersection(preds[2].index)
    v = pd.concat(
        [p.loc[common, "pred"].rename(f"s{s}") for s, p in zip(SEEDS, preds, strict=True)],
        axis=1,
    )
    gnn = v.mode(axis=1)[0].where(v.nunique(axis=1) < 3, v["s0"]).to_numpy()  # majority, else s0
    truth = preds[0].loc[common, "label"].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    test_idx = pos.loc[common].to_numpy()

    pred_size = pd.Series(gnn).map(types["size"]).fillna(0).to_numpy()
    true_size = pd.Series(truth).map(types["size"]).fillna(0).to_numpy()
    cand = (pred_size >= 2) & (pred_size <= args.max_size)
    rare_true = (true_size >= 2) & (true_size <= args.max_size)
    size = pd.Series(label).map(types["size"]).fillna(0).to_numpy()
    left = np.nonzero((size >= 2) & (size <= args.max_size) & (side == "left"))[0]
    right = test_idx[cand]
    n = max(len(left), len(right))
    print(
        f"two-stage: {cand.sum()} candidates, recall "
        f"{(cand & rare_true).sum() / rare_true.sum():.3f}, "
        f"precision {(cand & rare_true).sum() / cand.sum():.3f}",
        flush=True,
    )
    adj = subgraph(data, left, right, n)
    left_label = np.concatenate([label[left], np.full(n - len(left), "<pad>")])
    t = time.time()
    perm = faq(adj, np.full((n, n), 1.0 / n), True, args.iters)
    final = gnn.copy()
    final[cand] = left_label[perm[: len(right)]]
    rows = []
    for subset, k in [("all right test neurons", np.ones(len(truth), bool)), ("2-4", rare_true)]:
        for meth, p in [("gnn (majority)", gnn), ("gnn + graph matching", final)]:
            row = {"subset": subset, "n": int(k.sum()), "method": meth}
            rows.append({**row, **score(truth[k], p[k])})
    m = pd.DataFrame(rows)
    m["seconds"] = time.time() - t
    m["candidates"] = int(cand.sum())
    m["candidate_recall"] = (cand & rare_true).sum() / rare_true.sum()
    m["candidate_precision"] = (cand & rare_true).sum() / cand.sum()
    m.to_csv(OUT / "two_stage.csv", index=False)
    print(m.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--max-size", type=int, default=4, help="largest FAFB type size included")
    ap.add_argument(
        "--candidates",
        choices=["oracle", "gnn"],
        default="oracle",
        help="oracle: the right neurons of rare types (all methods); gnn: the right neurons the "
        "GNN assigns to a rare type (two-stage, writes two_stage.csv)",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_graph("cell_type", min_class_size=2)
    if args.candidates == "gnn":
        names = np.array(data.label_names)
        y = data.y.numpy()
        label = np.where(y >= 0, names[np.maximum(y, 0)], "")
        two_stage(data, label, np.asarray(data.side), stratify.fafb_types(), args)
        return
    names = np.array(data.label_names)
    y = data.y.numpy()
    side = np.asarray(data.side)
    types = stratify.fafb_types()
    label = np.where(y >= 0, names[np.maximum(y, 0)], "")
    size = pd.Series(label).map(types["size"]).fillna(0).to_numpy()
    rare = (y >= 0) & (size >= 2) & (size <= args.max_size)
    left = np.nonzero(rare & (side == "left"))[0]
    right = np.nonzero(rare & (side == "right"))[0]
    n = max(len(left), len(right))
    print(f"{len(left)} left and {len(right)} right neurons of 2-{args.max_size}-neuron types")

    # scored: right rare neurons in the GNN's left -> right test set (all three seeds)
    ids = data.node_ids.numpy()
    test_sets = [
        set(pd.read_parquet(GNN_RUNS / f"s{s}" / "predictions.parquet").root_id) for s in SEEDS
    ]
    in_test = np.array([all(i in t for t in test_sets) for i in ids[right]])
    scored = np.nonzero(in_test)[0]  # indices into `right`
    truth = label[right][scored]
    print(f"scored: {len(scored)} right neurons, {len(np.unique(truth))} types")

    adj = subgraph(data, left, right, n)
    print({k: v.nnz for k, v in adj.items()})
    votes = gnn_votes(ids[right][scored])
    left_label = np.concatenate([label[left], np.full(n - len(left), "<pad>")])

    rows, matches = [], {}
    gnn_acc = [score(truth, votes[f"s{s}"].to_numpy()) for s in SEEDS]
    rows.append(
        {
            "method": "gnn",
            "acc": np.mean([g["acc"] for g in gnn_acc]),
            "acc_std": np.std([g["acc"] for g in gnn_acc]),
            "macro_f1": np.mean([g["macro_f1"] for g in gnn_acc]),
            "seconds": 0.0,
        }
    )

    # vote matrix: right (scored rows, padded) x left; number of seeds predicting the left type
    vote = np.zeros((n, n))
    left_pos = pd.Series(np.arange(len(left))).groupby(label[left]).apply(list).to_dict()
    for j, r in enumerate(scored):
        for s in SEEDS:
            for li in left_pos.get(votes[f"s{s}"].iloc[j], []):
                vote[li, r] += 1.0

    t = time.time()
    rr, cc = linear_sum_assignment(vote, maximize=True)
    perm = np.empty(n, dtype=int)
    perm[cc] = rr
    pred = left_label[perm[scored]]
    rows.append({"method": "gnn_hungarian", **score(truth, pred), "seconds": time.time() - t})
    matches["gnn_hungarian"] = pred

    bary = np.full((n, n), 1.0 / n)
    init = sinkhorn(0.5 * vote / max(vote.sum(axis=0).max(), 1) + 0.5 * bary)
    for name, bisected, p0 in [
        ("gm_ipsi", False, bary),
        ("gm_bisected", True, bary),
        ("gm_gnn_init", True, init),
    ]:
        print(f"{name}:", flush=True)
        t = time.time()
        perm = faq(adj, p0, bisected, args.iters)
        pred = left_label[perm[scored]]
        rows.append({"method": name, **score(truth, pred), "seconds": time.time() - t})
        matches[name] = pred
        print(f"  {rows[-1]}", flush=True)

    m = pd.DataFrame(rows)
    m.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame({"root_id": ids[right][scored], "truth": truth, **matches}).to_parquet(
        OUT / "matches.parquet"
    )
    region = pd.Series(truth).map(types["region"]).to_numpy()
    per = []
    for reg in np.unique(region):
        k = region == reg
        g = [score(truth[k], votes[f"s{s}"].to_numpy()[k]) for s in SEEDS]
        per.append(
            {
                "method": "gnn",
                "region": reg,
                "n": int(k.sum()),
                "acc": np.mean([x["acc"] for x in g]),
                "macro_f1": np.mean([x["macro_f1"] for x in g]),
            }
        )
        for meth, p in matches.items():
            per.append({"method": meth, "region": reg, "n": int(k.sum()), **score(truth[k], p[k])})
    pd.DataFrame(per).to_csv(OUT / "per_region.csv", index=False)
    info = {
        "left": int(len(left)),
        "right": int(len(right)),
        "scored": int(len(scored)),
        "types": int(len(np.unique(truth))),
        "edges": {k: int(v.nnz) for k, v in adj.items()},
        "iters": args.iters,
    }
    (OUT / "info.json").write_text(json.dumps(info, indent=2))
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
