# Graph-matching baseline for the rarest cell types

Script: `experiments/b_gnn_celltypes/graph_matching.py` (no training; logs `run.log`,
`run_two_stage.log`). Compared with the left → right GNN at a minimum type size of 2
(`results/size_threshold/m2/within_left_right`, seeds 0–2).

## Setup

- **Task:** type the right-hemisphere FAFB neurons of types with 2–4 FAFB neurons, given the labels
  of the left hemisphere. Scored on the 8,142 such neurons in all three GNN test sets (6,438 types).
- **Graph matching** (FAQ / Frank–Wolfe, Vogelstein et al. 2015; bisected variant after Pedigo et al.
  2023): a left × right doubly stochastic matrix maximizing ipsilateral agreement
  ⟨A_LL, P A_RR Pᵀ⟩ (+ contralateral ⟨A_LR, P A_RL P⟩), 30 iterations, projected to a one-to-one
  matching. Edge weights are log1p(synapses). Each right neuron gets the type of its matched left
  neuron. The implementation recovers a hidden permutation exactly on synthetic graphs, from
  ipsilateral or from contralateral edges alone.
- **Oracle candidates:** the matching runs between the 8,183 left and 8,182 right neurons of
  2–4-neuron types. It sees only the edges among them, not the rest of the brain. Knowing which
  right neurons are rare-type comes from the labels, and the GNN does not get this: it chooses among
  all 8,582 types.
- **Label-free two-stage:** the GNN (majority of 3 seeds) types all 68,593 right test neurons. The
  9,041 it assigns to a 2–4-neuron type are matched against the left rare neurons, and all others
  keep the GNN's type. These candidates cover 93.1 % of the true rare-type neurons (recall), and
  83.8 % of them are rare-type (precision).

## Results (accuracy % / macro-F1)

Oracle candidates, 8,142 rare-type right neurons:

| Method | Accuracy | Macro-F1 | Time |
|---|---|---|---|
| GNN (mean of 3 seeds) | 75.9 ± 0.7 | 0.74 | – |
| GNN votes + one-to-one assignment (same oracle restriction) | 79.8 | 0.81 | 1 s |
| Graph matching, ipsilateral | 91.2 | 0.92 | 16 min |
| **Graph matching, bisected** | **92.2** | **0.93** | 13 min |
| Graph matching, bisected, started from the GNN votes | 92.2 | 0.93 | 13 min |

By region (bisected matching vs. GNN): central brain 91.7 vs. 76.8 % (6,632 neurons); other
(ascending, descending and others) 93.5 vs. 67.2 % (1,185); visual projection 98.7 vs. 90.1 % (230).
Sensory and optic-lobe intrinsic have under 60 neurons each and favor graph matching too.

Label-free two-stage (GNN → graph matching for predicted rare types):

| Neurons scored | GNN (majority of 3) | GNN + graph matching |
|---|---|---|
| 2–4-neuron types (8,142) | 78.9 / 0.78 | **85.8 / 0.88** |
| All right test neurons (68,593) | 86.6 / 0.75 | **87.4 / 0.84** |

## Findings

1. **Classic graph matching beats the GNN on the rarest types within a connectome, by a wide
   margin.** With the same oracle candidate set, bisected matching types 92.2 % of the 2–4-neuron
   types' neurons. The GNN, restricted to the same candidates and forced one-to-one, reaches
   79.8 %. Contralateral edges add 1 point, as in Pedigo et al. Starting from the GNN's votes
   changes nothing: the matching converges to the same solution.
2. **The GNN's 75.9 % "one-shot" result is therefore not evidence that a trained model is needed
   for rare types.** These types were defined by matching neurons across hemispheres, so pairing
   homologs by wiring is the natural method. The matching wins even though it sees only the
   edges among about 8,000 neurons per side.
3. **The two components combine well without labels.** The GNN picks the rare-type candidates
   (93 % recall), and graph matching types them. On rare types this raises accuracy from 78.9 to
   85.8 % and macro-F1 from 0.78 to 0.88. Over all 68,593 right test neurons, macro-F1 rises from
   0.75 to 0.84, and accuracy rises by 0.8 points. The candidates the GNN misses (7 %) and the
   common-type neurons it wrongly sends to matching (16 % of candidates) cost about 6 points
   against the oracle version.
4. **Open, and what matters for the paper:** this is pairing within one animal. Across
   connectomes, where the GNN types 2–4-neuron types at 54 % (FAFB → MCNS, label-free), there is
   no one-to-one correspondence, the two graphs differ in reconstruction, and graph matching has
   not been tested. A FAFB → MCNS run of the same two-stage pipeline (GNN candidates, matching
   between FAFB rare neurons and MCNS candidates, possibly with the prior-matched GNN labels as
   soft seeds) decides whether this is a contribution or only a within-connectome observation.

## Caveats

- One matching run per variant. The barycenter start is deterministic, so no seeds vary. FAQ
  finds a local optimum.
- The matching uses only edges inside the candidate set. Adding seeds from common types or the
  full graph might raise it further, but needs a scalable matcher (the dense 8k × 8k problem
  already takes about 13 minutes).
- GNN numbers are the existing left → right runs. The GNN was not retrained or tuned for this
  comparison.
