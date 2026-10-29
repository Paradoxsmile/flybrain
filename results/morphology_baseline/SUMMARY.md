# Morphology baseline for the rare-type tests (exploratory)

Plan: `docs/finalize_plan.md` (A1, A2), committed in `9ecc769` before any run here. BANC's labels were known from the pre-registered tests, so everything below is **exploratory**. Script: `experiments/b_gnn_celltypes/morphology_baseline.py`; numbers in `metrics.csv` here and `gm_banc*/metrics.csv`.

**Morphology baseline.** Each BANC neuron takes the type of its NBLAST top match: `fafb_nblast_match` in the brain, `manc_nblast_match` in the cord. Both come from BANC's metadata, before review. The baseline uses no BANC label and no connectivity. The script reproduces the reported two-stage numbers exactly (BANC brain 41.3 %, BANC cord 61.0 %), which checks the scoring.

## Result: A4's framing is "wiring adds to morphology"

On the rare stratum (2–4-neuron source types, reference A), the best combination beats NBLAST alone by **+6.0 points** on the BANC brain (95 % CI 5.1–7.1, resampling 5,416 types). On the BANC cord the gap is **+34.4 points** (32.7–36.0). Under the rule fixed in the plan (Δ ≥ 3), the headline is "wiring adds to morphology".

*Accuracy % / macro-F1 on the 2–4 stratum. Reference A: BANC's typing (brain) or the matched MANC type (cord). Reference B: fafb_cell_type (brain, central brain; NBLAST with review, so it favors NBLAST) or BANC's curated type (cord).*

| Labeling | Brain, ref A (14,512) | Brain, ref B | Cord, ref A (5,989) | Cord, ref B |
|---|---|---|---|---|
| Stage 1 (label-free GNN) | 22.2 / 0.19 | 21.3 | 39.5 / 0.36 | 40.7 |
| Two-stage, `gnn` candidates (pre-registered) | 41.3 / 0.47 | 40.6 | – | – |
| Two-stage, `union` candidates | 51.2 / 0.57 | 50.6 | 61.0 / 0.61 (pre-registered) | 61.0 |
| NBLAST alone | 66.5 / 0.65 | 66.8 | 30.5 / 0.29 | 30.2 |
| NBLAST, else two-stage (`union`) | 70.7 / 0.68 | 70.3 | 31.3 / 0.30 | 31.0 |
| Two-stage, candidates `union` + NBLAST rare-type (A2-ii) | 62.6 / 0.67 | 62.2 | 64.7 / 0.63 | 64.7 |
| **Two-stage matched, else NBLAST (`union`)** | **72.5 / 0.68** | **72.0** | **64.9 / 0.59** | 64.6 |
| Two-stage matched, else NBLAST (A2-ii candidates) | 70.6 / **0.71** | 70.0 | 65.1 / 0.61 | 64.5 |

- **Brain.** Morphology alone (66.5 %) beats wiring alone (51.2 % with `union` candidates). The two are complementary, though. Where the matching assigned a FlyWire neuron, its type is better than NBLAST's; elsewhere NBLAST is better. The combination gains 6 points over NBLAST, and the conclusion is the same against reference B, which favors NBLAST.
- **Cord.** The raw NBLAST column is weak. Its match has the reviewed match's type in only 31 % of cases, and only 29 % of its wrong matches lie in the reviewed match's hemilineage. Errors range from sibling types (IN09A056 vs IN09A057) to other classes (sensory neurons matched to interneurons). So the cord gap measures wiring against *unreviewed automatic* morphology, not against morphology done well.
- **Where they disagree** (ref A, 2–4, neurons with an NBLAST match; `agreement.csv`): where NBLAST and the pre-registered two-stage agree, both are right 94.7 % (brain) and 92.2 % (cord) of the time. Where they disagree, NBLAST is right in 60.7 % of cases on the brain and two-stage in 15.5 %; on the cord, two-stage 51.1 % and NBLAST 11.8 %. Agreement is therefore a cheap confidence flag.
- **NBLAST as candidate selection (A2-ii).** Adding NBLAST's rare-type neurons raises candidate recall from 69.7 to 94.6 % (brain) and from 80.4 to 89.9 % (cord). The matching then gains 11.4 and 3.7 points over `union`, and macro-F1 is best with these candidates. The cost falls on types of 5–19 neurons (cord 52.8 → 49.9 %), as with `union`.
- **`union` on the BANC brain.** This is the first run of the improved candidate rule on the held-out brain: 41.3 → 51.2 %. It is exploratory, since the rule was fixed after the pre-registered BANC test, and it agrees with the development result (Table 7).

## Without the GNN: NBLAST-only candidates (exploratory, added 2026-10-29)

Run after the pre-specified seeded-matching analysis (`results/seeded_matching/`), on labels already seen. Candidates are only the neurons whose NBLAST type is a 2-4-neuron source type (`candidates_nblast_only.parquet`, written by `--candidates-out`); the matching is unchanged. "No GNN" takes the matched type where the matching assigned a source neuron, NBLAST's type elsewhere, and no type where neither, so no step uses the GNN. Neurons without a type count as wrong; they are exactly the neurons without an NBLAST hit (brain 1,024 of 14,512, 7.1 %; cord 72 of 5,989, 1.2 %), which NBLAST alone also gets wrong. The A2-ii combination types 60.3 % (brain) and 65.3 % (cord) of them by stage 1, which accounts for 4.3 of its 11.7 points over the GNN-free version on the brain and 0.8 of 19.5 in the cord. Accuracy %, ref A, 2-4 stratum:

| | Brain | Cord |
|---|---|---|
| Candidate recall: NBLAST only / A2-ii (`union` + NBLAST) | 85.2 / 94.6 | 63.0 / 89.9 |
| NBLAST alone | 66.5 | 30.5 |
| Two-stage, NBLAST-only candidates (stage 1 outside the matching) | 59.9 | 57.5 |
| **Matched, else NBLAST, no GNN** | **58.9** | **45.6** |
| Matched, else NBLAST, A2-ii candidates | 70.6 | 65.1 |
| No GNN − NBLAST alone [95 % CI] | −7.6 [−8.7, −6.5] | +15.0 [13.5, 16.5] |
| A2-ii combination − no GNN [95 % CI, 10,000 resamples] | +11.7 [11.0, 12.4] | +19.5 [18.2, 20.8] |

Without the GNN's candidates, matching combined with NBLAST is worse than NBLAST alone on the brain. Stage 1 is therefore still needed when morphology matches exist. The last interval was computed with `morphology_baseline.bootstrap_delta` on the saved matchings (not written to a file by the script); the others are in `agreement.csv`.

## Caveats

- Exploratory: BANC's labels were known, and the "best combination" is the best of six. Every combination built on the `union` or A2-ii matching beats NBLAST on the brain by 4.1 points or more.
- NBLAST here is the raw top hit. A reviewed morphology match would be stronger, but reviewing is the step a label-free method is meant to save.
- Reference A on the brain rests partly on connectivity, and reference B on NBLAST. The conclusion holds under both.
- For the pre-registered `gnn` matching, the saved file lacks the assigned FlyWire neuron, so "assigned" is approximated there as "the matching changed the type". The `union` and A2-ii rows are exact.
- Over all labeled neurons the picture differs: "NBLAST, else two-stage" is best on the brain (62.8 %), and the rare-type combination costs common types. A practical pipeline would apply the matching result only to the rare candidates.

## Commands

```bash
uv run python experiments/b_gnn_celltypes/morphology_baseline.py --candidates-out
uv run python experiments/b_gnn_celltypes/graph_matching_mcns.py --source manc --target banc_cord --stage match --candidates file:nblast_union --passing-block --out morphology_baseline/gm_banc_cord
uv run python experiments/b_gnn_celltypes/graph_matching_mcns.py --target banc --stage match --candidates union --out morphology_baseline/gm_banc
uv run python experiments/b_gnn_celltypes/graph_matching_mcns.py --target banc --stage match --candidates file:nblast_union --out morphology_baseline/gm_banc
uv run python experiments/b_gnn_celltypes/graph_matching_mcns.py --source manc --target banc_cord --stage match --candidates file:nblast_only --passing-block --out morphology_baseline/gm_banc_cord
uv run python experiments/b_gnn_celltypes/graph_matching_mcns.py --target banc --stage match --candidates file:nblast_only --out morphology_baseline/gm_banc
uv run python experiments/b_gnn_celltypes/morphology_baseline.py
```

Runtime on a CPU: cord matching a few minutes; brain `union` 29 min; brain A2-ii 59 min; brain NBLAST-only 24 min.
