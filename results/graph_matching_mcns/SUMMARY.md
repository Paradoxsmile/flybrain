# Two-stage typing of the rarest cell types across connectomes (FAFB → MCNS)

Script: `experiments/b_gnn_celltypes/graph_matching_mcns.py` (no training). Logs: `weights.log`,
`run_gnn_soft.log`, `run_gnn_bary.log`, `run_oracle_bary.log`, `run_gnn_soft_lam1.log`. Results:
`metrics.csv`, `matches_*.parquet` (candidates with label, stage-1 type and final type), `info_*.json`.

## Setup

- **Task:** type the MCNS neurons of cell types with 2–4 FAFB neurons. No MCNS label is used. Scored
  on the 13,742 labeled MCNS neurons whose flywireType is such a type (5,370 types), which is the
  stratum behind the 54.2 % in `results/size_threshold/SUMMARY.md`, and on all 133,356 labeled MCNS
  neurons. Macro-F1 is the mean per-type F1 over the stratum's true types, with predictions counted
  over all labeled neurons (as in `stratify.per_type`).
- **Stage 1:** the label-free FAFB → MCNS labeling at minimum type size 2
  (`results/size_threshold/m2/label_free`, after NTAC; chosen MCNS variant "raw", source
  normalization, ρ = 0.75). Its candidates are the 18,479 MCNS neurons it assigns to a 2–4-neuron
  type. They include 83.8 % of the labeled rare-type neurons (recall). Of the candidates, 62.3 % are
  labeled rare-type, 14.0 % are labeled with another type and 23.6 % are unlabeled. All other MCNS
  neurons keep their stage-1 type.
- **Stage 2:** graph matching (FAQ / Frank–Wolfe, 30 iterations, as in `graph_matching.py`) between
  the 16,365 FAFB neurons of 2–4-neuron types and the candidates. Left and right do not
  correspond one-to-one across animals as they do within one, so the matcher works on the whole rare
  subgraph of each brain. The matching P = diag(P_L, P_R) is block-diagonal by hemisphere and
  maximizes ipsilateral plus contralateral agreement:
  ⟨F_LL, P_L M_LL P_Lᵀ⟩ + ⟨F_RR, P_R M_RR P_Rᵀ⟩ + ⟨F_LR, P_L M_LR P_Rᵀ⟩ + ⟨F_RL, P_R M_RL P_Lᵀ⟩.
  Edge weights are log1p(synapses) among the selected neurons only.
  - The blocks are 8,183 FAFB × 8,872 MCNS (left) and 8,182 × 9,342 (right), each padded to
    9,342 × 9,342.
  - Every FAFB rare neuron is matched to a real candidate. The surplus of 1,849 candidates
    (689 left, 1,160 right) lands on dummies and keeps its stage-1 type.
  - Neurons without a left/right side are not matched and keep their stage-1 type: 49 FAFB rare
    neurons and 265 candidates.
  - Each run takes about 27 min, with 4.5 GB peak RAM.
- **Starts:**
  - *Barycenter.* No GNN information enters the matching.
  - *Soft start (P0).* The stage-1 prior-matched GNN probabilities, before NTAC: P0[f, m] = q_m(type
    of f) divided by the number of FAFB neurons of that type in the hemisphere. P0 is mixed with the
    barycenter and Sinkhorn-normalized. The recomputed ensemble agrees with `pred_selected` on 100 %
    of neurons.
  - *Soft start + linear term.* The same P0, with the probabilities also kept in the objective as
    λ⟨S, P⟩. At λ = 1 this term equals the graph term at the soft probabilities' own matching.
- **Oracle candidates (diagnostic):** the matching runs against exactly the labeled MCNS rare-type
  neurons: 6,837 left and 6,851 right. 2,677 FAFB neurons go unmatched.

## Results (accuracy % / macro-F1)

| Method | 2–4-neuron types (13,742) | Rare-type candidates (11,475) | All labeled (133,356) |
|---|---|---|---|
| Stage 1: label-free GNN | 54.2 / 0.49 | 64.8 | 84.2 / 0.51 |
| Stage-1 soft probabilities, made one-to-one (no wiring) | 59.9 / 0.62 | 71.6 | 84.8 / 0.61 |
| **Two-stage: GNN candidates + graph matching, barycenter start** | **74.5 / 0.78** | **89.1** | **86.3 / 0.73** |
| Two-stage, soft start (P0 = GNN probabilities) | 74.6 / 0.78 | 89.2 | 86.3 / 0.74 |
| Two-stage, soft start + GNN probabilities as linear term (λ = 1) | 73.4 / 0.77 | 87.8 | 86.2 / 0.73 |
| Oracle candidates + graph matching, barycenter start | 85.3 / 0.87 | – | 87.4 / 0.81 |

"Rare-type candidates" are the labeled rare-type neurons that stage 1 sends to matching and that have
a side. Only these can change. The rest of the 2–4 stratum was not selected as candidates, so its
stage-1 type is wrong by construction.

By region (FAFB super class of the true type), 2–4-neuron types:

| Region | Neurons | Stage 1 | Two-stage (barycenter) | Oracle candidates |
|---|---|---|---|---|
| Central brain | 11,094 | 58.0 | 75.2 | 85.7 |
| Other (ascending, descending, …) | 2,045 | 25.2 | 66.1 | 79.6 |
| Visual projection | 423 | 83.7 | 91.0 | 97.4 |
| Optic-lobe intrinsic | 106 | 97.2 | 98.1 | 97.2 |
| Sensory | 74 | 59.5 | 75.7 | 95.9 |

## Findings

1. **The two-stage approach works across connectomes, by a wide margin.** Without labels in MCNS,
   accuracy on 2–4-neuron types rises from 54.2 to 74.5 %, and macro-F1 from 0.49 to 0.78. Over all
   labeled MCNS neurons, accuracy rises from 84.2 to 86.3 % and macro-F1 from 0.51 to 0.73. The gain
   (+20 points) is larger than within FAFB (78.9 → 85.8 %, `results/graph_matching`). So it is a
   cross-dataset contribution, not only a within-connectome observation.
2. **The wiring does the work.** Forcing the GNN's own probabilities to be one-to-one gives 59.9 %.
   Graph matching gives 74.5 % from the barycenter, with no GNN input, and 74.6 % from the GNN's
   probabilities. The two runs agree on 92.6 % of candidates. Keeping the GNN's probabilities in the
   objective lowers accuracy slightly (73.4 %). As within FAFB, the GNN's role is to select the
   candidates, not to guide the matching.
3. **Graph matching between two animals is nearly as accurate as between two hemispheres.**
   - With oracle candidates it types 85.3 % of the MCNS rare-type neurons, against 92.2 % within
     FAFB. The two brains differ in reconstruction and individual, the rare sets differ in size
     (16,365 FAFB vs 13,742 labeled MCNS neurons), and each type's neuron count differs between
     them.
   - On the rare-type neurons that the GNN does select, the two-stage matching reaches 89.1 %.
   - The 2,592 common-type neurons sent to matching were already mistyped by stage 1, so they
     cost nothing.
   - On rare-type candidates, matching fixes 3,288 stage-1 errors and breaks 491 correct types
     (soft start).
4. **The remaining gap is candidate selection.** Stage 1 misses 16 % of the rare-type neurons. That
   loss (85.3 % with oracle candidates vs 74.5 %) is larger than the loss from matching across animals
   (92.2 % vs 85.3 %). The largest gains are in the regions where the GNN is weakest:
   - "other" (mostly ascending and descending neurons): 25.2 → 66.1 %;
   - central brain: 58.0 → 75.2 %.

## Caveats

- One run per variant. FAQ finds a local optimum; the barycenter start is deterministic.
- From the barycenter, the first Frank–Wolfe direction is approximate. There the gradient is a sum
  of a few outer products of degree vectors, and scipy's assignment solver needed 5–10 min per
  block on it: 612 s in float32, 429 s in float64, 267 s after rounding, and similar from a
  randomized start. The first step therefore pairs the leading singular vectors by rank instead.
  The later iterations use the exact solver. The soft start, which never uses the approximation,
  converges to nearly the same matching.
- The hemisphere blocks assume that FAFB `side` and MCNS soma side mean the same thing. The high
  accuracy suggests they do. Neurons without a side are not matched.
- The matching is one-to-one, while MCNS can have more or fewer neurons of a type than FAFB. Many-to-one
  matching (replicated FAFB rows, or dummies per type) might recover some of the remaining errors.
- 4,366 unlabeled MCNS candidates also receive matched types. They are not scored.
- Stage 1 is the single label-free run of `size_threshold.py` at minimum type size 2, with no seeds
  varied.
