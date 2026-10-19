"""Two-stage typing of the rarest cell types across connectomes: FAFB -> MCNS, 2-4 FAFB neurons.

Within FAFB, the GNN picks the right-hemisphere neurons of 2-4-neuron types and graph matching
against the left hemisphere types them (graph_matching.py --candidates gnn: 78.9 -> 85.8 %).
This script runs the same two stages from FAFB to MCNS, where the label-free GNN pipeline types
the MCNS neurons of 2-4-neuron FAFB types at 54.2 %.

Stage 1: the label-free FAFB -> MCNS labeling at minimum type size 2 (size_threshold.py with
lean_label_free.py, results/size_threshold/m2/label_free; prediction after NTAC). The MCNS neurons
it assigns to a 2-4-neuron FAFB type are the candidates (labeled or not); all other MCNS neurons
keep the stage-1 type. With --candidates oracle, the candidates are instead the labeled MCNS
neurons whose flywireType is such a type (a diagnostic of the matching itself).

Stage 2: the FAFB neurons of 2-4-neuron types are matched one-to-one to the candidates. There is
no left/right bisection across animals, so the problem is the whole rare subgraph of each brain
with the matching restricted to the same hemisphere: P = diag(P_L, P_R) (FAFB x MCNS, each block
padded to n x n) maximizes, by Frank-Wolfe (FAQ),
    <F_LL, P_L M_LL P_L^T> + <F_RR, P_R M_RR P_R^T>          (ipsilateral)
  + <F_LR, P_L M_LR P_R^T> + <F_RL, P_R M_RL P_L^T>          (contralateral)
with F, M the log1p(synapse) adjacencies among the FAFB rare neurons and the MCNS candidates
(graph_matching.subgraph; MCNS in the preprocessing variant the label-free run chose, "raw").
Each candidate matched to a FAFB neuron gets that neuron's type; candidates matched to padding
(dummies) keep the stage-1 type. Neurons without a left/right side (FAFB 49 rare, MCNS a few
hundred candidates) are not matched and keep the stage-1 type.

Starts (--start):
- bary: the barycenter. Its gradient is a sum of a few outer products of degree vectors, on which
  scipy's assignment solver takes 5-10 min per block (also in float64, after rounding, or from a
  randomized start), so the first direction pairs the leading singular vectors by rank instead;
- soft: the prior-matched GNN probabilities (stage-1 ensemble before NTAC, the chosen variant,
  normalization and rho): P0[f, m] = q_m(type f) / (FAFB neurons of type f in the hemisphere),
  mixed with the barycenter and Sinkhorn-normalized as graph_matching.py does with GNN votes.
The soft start's projection to a one-to-one matching (no connectivity) is reported too. With
--lam > 0, the soft probabilities S also stay in the objective as a linear term lam' <S, P>, lam'
chosen so that lam = 1 makes it equal to the graph term at the soft projection.

Scored on the labeled MCNS neurons (accuracy, macro-F1 = mean per-type F1 as stratify.per_type):
those whose flywireType is a 2-4-neuron FAFB type (13,742) and all labeled ones (133,356).

--stage weights recomputes and caches the stage-1 prior-matching weights and both candidate sets'
soft probabilities (one dense N x K float32 matrix, ~5 GB, at a time). --stage match runs one
matching and appends to metrics.csv.

Writes results/graph_matching_mcns/{soft_<candidates>.npz, metrics.csv, info_*.json,
matches_*.parquet}.

Usage: python experiments/b_gnn_celltypes/graph_matching_mcns.py --stage weights
       python experiments/b_gnn_celltypes/graph_matching_mcns.py --stage match [--start soft]
           [--lam 1] [--candidates gnn|oracle] [--iters 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from types import SimpleNamespace

import label_free
import lean_label_free
import numpy as np
import pandas as pd
import stratify
import torch
from graph_matching import sinkhorn, subgraph
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds

from flybrain.loading import load_graph
from flybrain.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "graph_matching_mcns"
STAGE1 = REPO_ROOT / "results" / "size_threshold" / "m2" / "label_free" / "cell_type"
MODELS = "size_threshold/m2/models"
MIN_SIZE, MAX_SIZE = 2, 4
SIDES = ("left", "right")
# <F[key], P_a M[key] P_c^T>: (key, a, c), blocks 0 = left, 1 = right
TERMS = (("LL", 0, 0), ("RR", 1, 1), ("LR", 0, 1), ("RL", 1, 0))


def graphs(variant: str):
    fafb = load_graph("cell_type", min_class_size=MIN_SIZE)
    mcns = load_graph("cell_type", min_class_size=MIN_SIZE, dataset="mcns", mcns_variant=variant)
    return fafb, mcns


def labels(data) -> np.ndarray:
    names = np.array(data.label_names, dtype=object)
    y = data.y.numpy()
    return np.where(y >= 0, names[np.maximum(y, 0)], None)


def rare(type_names, sizes: pd.Series) -> np.ndarray:
    s = pd.Series(type_names).map(sizes).fillna(0).to_numpy()
    return (s >= MIN_SIZE) & (s <= MAX_SIZE)


def candidates(stage1: pd.DataFrame, sizes: pd.Series, kind: str) -> np.ndarray:
    """Indices into MCNS: predicted (gnn) or labeled (oracle) as a 2-4-neuron FAFB type."""
    if kind == "gnn":
        return np.nonzero(rare(stage1.pred, sizes))[0]
    return np.nonzero(stage1.label.notna().to_numpy() & rare(stage1.label, sizes))[0]


def setup(args):
    """Graphs, stage-1 labeling and the candidates (indices into MCNS)."""
    sel = json.loads((STAGE1 / "selected.json").read_text())
    fafb, mcns = graphs(sel["chosen"]["variant"])
    stage1 = pd.read_parquet(STAGE1 / "predictions.parquet")
    assert (stage1.root_id.to_numpy() == mcns.node_ids.numpy()).all()
    sizes = stratify.fafb_types()["size"]
    cand = candidates(stage1, sizes, args.candidates)
    return SimpleNamespace(sel=sel, fafb=fafb, mcns=mcns, stage1=stage1, sizes=sizes, cand=cand)


def soft_probabilities(s) -> None:
    """Cache the stage-1 prior-matched ensemble probabilities of both candidate sets on the rare
    types (mean over seeds of the adjusted probabilities; their argmax over all types is
    predictions.parquet's pred_selected)."""
    label_free.LABELS["cell_type"] = MIN_SIZE
    chosen = s.sel["chosen"]
    ns = argparse.Namespace(prior=s.sel["prior"], max_weight=s.sel["max_weight"])
    paths = label_free.model_paths(MODELS, "cell_type")
    t = time.time()
    probs, weights, _ = lean_label_free.seed_outputs(paths, s.fafb, s.mcns, chosen["norm"], ns, {})
    print(f"prior matching: {time.time() - t:.0f} s", flush=True)
    ens, _ = lean_label_free.adjusted(probs, weights, chosen["rho"])
    names = np.array(s.mcns.label_names, dtype=object)
    agree = (names[ens] == s.stage1.pred_selected.to_numpy()).mean()
    print(f"agreement with pred_selected: {agree:.6f}", flush=True)
    rare_cols = np.nonzero(rare(names, s.sizes))[0]
    for kind in ("gnn", "oracle"):
        cand = candidates(s.stage1, s.sizes, kind)
        out = np.zeros((len(cand), len(rare_cols)), dtype=np.float32)
        idx = torch.from_numpy(cand)
        for p, v in zip(probs, weights, strict=True):
            with torch.no_grad():
                q = torch.softmax(p.z[idx] @ p.w.T + p.b, dim=1).numpy()
            q *= (v ** chosen["rho"]).astype(np.float32)
            q /= q.sum(axis=1, keepdims=True)
            out += q[:, rare_cols] / len(probs)
        np.savez(
            OUT / f"soft_{kind}.npz",
            cand=s.mcns.node_ids.numpy()[cand],
            types=names[rare_cols].astype(str),
            probs=out,
            weights=np.stack(weights),
            agreement=agree,
        )


def as32(adj: dict) -> dict:
    return {k: v.astype(np.float32).tocsr() for k, v in adj.items()}


def gradient(F: dict, M: dict, P: list) -> list:
    """d/dP_a of sum over TERMS of <F[k], P_a M[k] P_c^T>: F[k] P_c M[k]^T and F[k]^T P_a M[k]."""
    G = [np.zeros_like(p) for p in P]
    for k, a, c in TERMS:
        G[a] += F[k] @ (M[k] @ P[c].T).T
        G[c] += F[k].T @ (M[k].T @ P[a].T).T
    return G


def bilinear(F: dict, M: dict, D: list, E: list) -> float:
    """sum over TERMS of <F[k], D_a M[k] E_c^T> = <F[k] E_c, D_a M[k]>."""
    return float(sum(((F[k] @ E[c]) * (M[k].T @ D[a].T).T).sum() for k, a, c in TERMS))


def rank1_assignment(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assignment maximizing <u v^T, Q> for the leading singular pair of g: pair u and v by
    rank (rearrangement inequality). At the barycenter the gradient is a sum of a few outer
    products of degree vectors, a near-degenerate matrix on which scipy's solver takes 5-10 min
    for n ~ 9k (and as long after rounding or a randomized start); this approximate first
    direction takes seconds."""
    u, sv, vt = svds(g.astype(np.float64), k=1)
    u, v = u[:, 0], vt[0]
    if (g @ v) @ u < 0:
        v = -v
    rows, cols = np.argsort(-u), np.argsort(-v)
    return rows, cols


def faq_blocks(
    F: dict, M: dict, P0: list, iters: int, S: list | None = None, tol: float = 1e-4
) -> list:
    """graph_matching.faq for a block-diagonal P (one block per hemisphere), maximizing
    f(P) + sum_a <S_a, P_a> if a linear term S is given. From the barycenter, the first
    direction is rank1_assignment's. Returns per block match[m] = FAFB index of MCNS index m."""
    P = [p.astype(np.float32) for p in P0]
    flat = all(np.ptp(p) == 0 for p in P)
    for it in range(iters):
        t = time.time()
        G = gradient(F, M, P)
        if S is not None:
            for g, s in zip(G, S, strict=True):
                g += s
        D = []
        for g, p in zip(G, P, strict=True):
            lmo = rank1_assignment if it == 0 and flat else _lap
            rows, cols = lmo(g)
            d = -p
            d[rows, cols] += 1.0
            D.append(d)
        c1 = float(sum((g * d).sum() for g, d in zip(G, D, strict=True)))
        del G
        c2 = bilinear(F, M, D, D)
        if c2 < 0:
            a = min(1.0, max(0.0, -c1 / (2 * c2)))
        else:
            a = 1.0 if c1 + c2 > 0 else 0.0
        for p, d in zip(P, D, strict=True):
            p += a * d
        step = a * max(np.abs(d).max() for d in D)
        del D
        print(f"    iter {it}: step {a:.3f}, <G, D> {c1:.1f}, {time.time() - t:.0f} s", flush=True)
        if step < tol:
            break
    return [project(p) for p in P]


def _lap(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return linear_sum_assignment(g, maximize=True)


def project(p: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(p, maximize=True)
    match = np.empty(p.shape[1], dtype=int)
    match[cols] = rows
    return match


def soft_start(s, args, f_idx: np.ndarray, m_idx: np.ndarray, flabel, n: int) -> np.ndarray:
    """P0 (FAFB x MCNS, n x n): q_m(type f) / FAFB neurons of type f in the block."""
    z = np.load(OUT / f"soft_{args.candidates}.npz")
    assert (z["cand"] == s.mcns.node_ids.numpy()[s.cand]).all()
    row = pd.Series(np.arange(len(s.cand)), index=s.cand).loc[m_idx].to_numpy()
    col = pd.Series(np.arange(len(z["types"])), index=z["types"]).loc[flabel[f_idx]].to_numpy()
    count = pd.Series(flabel[f_idx]).map(pd.Series(flabel[f_idx]).value_counts()).to_numpy()
    soft = np.zeros((n, n), dtype=np.float32)
    soft[: len(f_idx), : len(m_idx)] = (z["probs"][row][:, col] / count).T
    return soft


def score(truth: np.ndarray, pred: np.ndarray, subset: np.ndarray, types) -> dict:
    """Accuracy on the labeled neurons of `subset` and macro-F1 over their true types, F1 per
    type over all labeled neurons (stratify.per_type)."""
    lab = pd.notna(truth)
    df = pd.DataFrame({"label": truth[lab], "pred_x": pred[lab]})
    t = stratify.per_type(df, "x", types).set_index("cell_type")
    k = lab & subset
    t = t.loc[pd.Series(truth[k]).unique()]
    return {
        "n": int(k.sum()),
        "types": len(t),
        "acc": float((pred[k] == truth[k]).mean()),
        "macro_f1": float(t.f1.mean()),
    }


def match(s, args) -> None:
    flabel = labels(s.fafb)
    fside = np.asarray(s.fafb.side)
    mside = np.asarray(s.mcns.side)
    frare = rare(flabel, s.sizes)
    f_blocks = [np.nonzero(frare & (fside == side))[0] for side in SIDES]
    m_blocks = [s.cand[mside[s.cand] == side] for side in SIDES]
    n = max(len(b) for b in f_blocks + m_blocks)
    print(
        f"FAFB rare {[len(b) for b in f_blocks]}, MCNS candidates {[len(b) for b in m_blocks]}"
        f" (of {len(s.cand)}), n = {n}",
        flush=True,
    )
    F = as32(subgraph(s.fafb, *f_blocks, n))
    M = as32(subgraph(s.mcns, *m_blocks, n))
    print({"F": {k: v.nnz for k, v in F.items()}, "M": {k: v.nnz for k, v in M.items()}})

    truth = s.stage1.label.to_numpy(dtype=object)
    stage1 = s.stage1.pred.to_numpy(dtype=object)
    types = stratify.fafb_types()
    rows = []
    in_rare = pd.notna(truth) & rare(truth, s.sizes)

    def evaluate(method: str, matches: list | None, seconds: float) -> None:
        pred = stage1.copy()
        info = {"to_dummy": 0, "fafb_unmatched": 0}
        for fb, mb, mt in zip(f_blocks, m_blocks, matches or [], strict=False):
            fm = mt[: len(mb)]
            real = fm < len(fb)
            pred[mb[real]] = flabel[fb[fm[real]]]
            info["to_dummy"] += int((~real).sum())
            info["fafb_unmatched"] += int(len(fb) - real.sum())
        rare_cand = np.isin(np.arange(len(truth)), np.concatenate(m_blocks)) & in_rare
        for subset, k in [
            ("2-4 (MCNS label)", in_rare),
            ("2-4, matched candidates", rare_cand),
            ("all labeled", np.ones(len(truth), bool)),
        ]:
            rows.append(
                {
                    "candidates": args.candidates,
                    "method": method,
                    "subset": subset,
                    **score(truth, pred, k, types),
                    **info,
                    "seconds": round(seconds),
                }
            )
            print(rows[-1], flush=True)
        if method.startswith("gm"):
            pd.DataFrame(
                {
                    "root_id": s.mcns.node_ids.numpy(),
                    "label": truth,
                    "stage1": stage1,
                    "pred": pred,
                }
            ).iloc[s.cand].to_parquet(OUT / f"matches_{args.candidates}_{method}.parquet")

    evaluate("stage 1 (label-free GNN)", None, 0.0)
    bary = np.full((n, n), 1.0 / n, dtype=np.float32)
    if args.start == "soft":
        softs = [
            soft_start(s, args, fb, mb, flabel, n)
            for fb, mb in zip(f_blocks, m_blocks, strict=True)
        ]
        t = time.time()
        q0 = [project(p) for p in softs]
        evaluate("soft probabilities, one-to-one", q0, time.time() - t)
        linear = None
        if args.lam > 0:  # lam = 1: both terms equal at the soft probabilities' matching
            perms = [np.zeros((n, n), dtype=np.float32) for _ in q0]
            for perm, q in zip(perms, q0, strict=True):
                perm[q, np.arange(n)] = 1.0
            quad_value = bilinear(F, M, perms, perms)
            lin_value = sum(float((sf * pm).sum()) for sf, pm in zip(softs, perms, strict=True))
            del perms
            scale = args.lam * quad_value / lin_value
            print(f"linear term: f = {quad_value:.0f}, <S, Q> = {lin_value:.0f}, x {scale:.1f}")
            linear = [sf * np.float32(scale) for sf in softs]
        p0 = []
        for soft in softs:
            soft = 0.5 * soft / max(float(soft.sum(axis=0).max()), 1e-12) + 0.5 * bary
            p0.append(sinkhorn(soft).astype(np.float32))
        del softs
    else:
        p0, linear = [bary, bary.copy()], None
    del bary
    t = time.time()
    matches = faq_blocks(F, M, p0, args.iters, linear)
    method = f"gm_{args.start}" + (f"_lam{args.lam:g}" if linear is not None else "")
    evaluate(method, matches, time.time() - t)

    m = pd.DataFrame(rows)
    f = OUT / "metrics.csv"
    if f.exists():
        old = pd.read_csv(f)
        old = old[~((old.candidates == args.candidates) & old.method.isin(m.method))]
        m = pd.concat([old, m], ignore_index=True)
    m.to_csv(f, index=False)
    info = {
        "candidates": args.candidates,
        "method": method,
        "iters": args.iters,
        "variant": s.sel["chosen"]["variant"],
        "fafb_rare": [len(b) for b in f_blocks],
        "mcns_candidates": [len(b) for b in m_blocks],
        "candidates_without_side": int(len(s.cand) - sum(len(b) for b in m_blocks)),
        "fafb_rare_without_side": int(frare.sum() - sum(len(b) for b in f_blocks)),
        "n": n,
        "edges": {
            "fafb": {k: v.nnz for k, v in F.items()},
            "mcns": {k: v.nnz for k, v in M.items()},
        },
    }
    (OUT / f"info_{args.candidates}_{method}.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", choices=["weights", "match"], required=True)
    ap.add_argument("--candidates", choices=["gnn", "oracle"], default="gnn")
    ap.add_argument("--start", choices=["bary", "soft"], default="bary")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--lam",
        type=float,
        default=0.0,
        help="with --start soft: also keep the soft probabilities as a linear term, weighted so "
        "that lam = 1 makes it equal to the graph term at their one-to-one matching",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = setup(args)
    if args.stage == "weights":
        soft_probabilities(s)
    else:
        match(s, args)


if __name__ == "__main__":
    main()
