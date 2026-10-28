# Pre-registration: seeded graph matching, "wiring adds to morphology"

Written and committed on 2026-10-28, together with the analysis code (`experiments/b_gnn_celltypes/seeded_matching.py`, `tests/test_seeded_matching.py`), **before** the seeded matching was run or any of its results were computed. Only the label-free seed counts below were computed beforehand; they read no labels. Results will go to `results/seeded_matching/`.

## Why this test

The exploratory morphology baseline (`results/morphology_baseline/SUMMARY.md`, plan [finalize_plan.md](finalize_plan.md) A1-A4) found on BANC's rare types (2-4 FAFB neurons per type, reference A) that "graph-matched type where assigned, NBLAST elsewhere" reaches 72.5 %, NBLAST alone 66.5 % (+6.0, 95 % CI 5.1-7.1). That number is the best of six combinations, chosen after seeing BANC's labels. A review called it the paper's weakest point. This test fixes one new procedure, one contrast and one decision rule in advance.

## Feasibility (checked before writing this)

1. **MCNS has no morphology match to FlyWire.** The MCNS body annotations (`body-annotations-male-cns-v1.0-minconf-0.5.feather`) have only curated type columns (`flywireType`, `hemibrainType`, `mancType`) and a reviewed per-neuron MANC match (`mancBodyid`). There is no automatic NBLAST top hit, so the MCNS combination test (option a) is not possible without computing NBLAST ourselves (skeletons and a cross-template transform for ~150k neurons). We therefore run option (b), seeded graph matching on BANC.
2. **BANC's reference labels are partly circular with NBLAST.** BANC's preprint (Bates et al., bioRxiv 2025.07.31.667571) describes the matching to FlyWire and MANC as NBLAST-proposed candidates, then manual review and correction, also using connectivity. In the data, on the reference-A rare stratum of the brain (14,512 neurons):
   - the reference type equals the type of BANC's reviewed match (`fafb_match`) for 99.8 % of the neurons that have one (98.6 % do);
   - the reviewed match is the NBLAST top hit (`fafb_nblast_match`) for 67.5 % of the neurons that have both;
   - NBLAST's type is "correct" for 99.9 % of the neurons where review kept its top hit and for 12.0 % where review replaced it.

   In the cord (5,989 neurons) the reviewed `manc_match` is the NBLAST top hit for only 25.9 %, with the same pattern (100 % vs 6.6 %). NBLAST's accuracy against reference A is therefore largely the share of top hits that reviewers kept. This favors NBLAST in every comparison below, so the test is conservative for the hypothesis. (Review also used connectivity, which may favor wiring where reviewers overrode NBLAST; the descriptive split below shows both parts.)

## Hypothesis

**H1.** On BANC brain rare types, a pre-specified label-free combination of wiring and morphology (seeded graph matching; its type where it assigned a FAFB neuron, NBLAST elsewhere) types at least 3 points more neurons correctly than NBLAST alone.

## Data and split

- Source FAFB (FlyWire 783), target BANC (v888), both as already used for the pre-registered BANC test ([banc_rare_protocol.md](banc_rare_protocol.md)). Stage 1: `results/size_threshold/m2/label_free_banc_size_agreement` (unchanged).
- Candidates: A2-ii `nblast_union` (stage-1 `union` plus the neurons whose NBLAST type is a 2-4-neuron FAFB type), 24,325 neurons, the same set as the unseeded matching in `results/morphology_baseline/gm_banc/matches_nblast_union_gm_bary.parquet`.
- **No split.** No step uses a BANC label: stage 1 is label-free, NBLAST hits are BANC's automatic, pre-review matches, and seeds are chosen from them by a label-free rule. The whole reference-A 2-4 stratum is the test set: 14,512 neurons, 5,416 types.
- **Seeds** (per hemisphere block): a candidate whose NBLAST top hit is a FAFB neuron of a 2-4-neuron type in the same block, where that FAFB neuron is the top hit of no other candidate in the block. Counts (computed label-free): 3,088 left + 3,050 right = **6,138 seeds** in the brain; 728 + 725 + 691 (left, right, passing) = 2,144 in the cord.

## Method

`seeded_matching.py --stage match`: FAQ (Frank-Wolfe) as in `graph_matching_mcns.py`, blocks by hemisphere (cord: plus the passing block, `--passing-block`), 30 iterations, start = seeds at 1 and the barycenter on the rest. Seed pairs stay fixed; each linear step and the final projection are exact assignments (`linear_sum_assignment`) on the non-seed rows and columns. The candidates matched to a FAFB neuron take its type; the others keep stage 1.

## Primary analysis

- **Metric:** accuracy on the reference-A 2-4 stratum of the BANC brain.
- **Comparison:** Δ = accuracy("two-stage matched, else NBLAST [seeded]") − accuracy("NBLAST"), on the same 14,512 neurons. "NBLAST" counts neurons without a hit as wrong (as in Table 8).
- **Interval:** 95 % percentile bootstrap over true types (resample the 5,416 types with replacement, Δ = Σ correct differences / Σ neurons), 10,000 resamples, `numpy.random.default_rng(0)` (`morphology_baseline.bootstrap_delta`). It is conditional on the matching; the matching is not re-run.
- **Decision rule** (A4 of [finalize_plan.md](finalize_plan.md)):
  - **confirmed**: Δ ≥ 3 points and the CI's lower bound > 0;
  - **not confirmed, wiring adds less than 3 points**: CI lower bound > 0 but Δ < 3;
  - **inconclusive**: the CI includes 0;
  - **failed, NBLAST alone is better**: the CI's upper bound < 0.

## Secondary analyses (reported, do not change the decision)

1. Seeded vs unseeded: "matched, else NBLAST [seeded]" − "matched, else NBLAST [unseeded]" (same candidates); and two-stage seeded − unseeded.
2. BANC nerve cord (source MANC, `manc_nblast_match`, passing block), the same primary contrast and rule. Expected to pass easily, since NBLAST's cord top hits are weak (Table 8: 30.5 %).
3. Reference B (brain: `fafb_cell_type`, central brain, built from NBLAST with review; cord: BANC's curated type).
4. Descriptive splits of the primary Δ: non-seed neurons; neurons whose NBLAST top hit review kept vs replaced.

## What a pass does and does not show

BANC's labels were seen before this test, and the unseeded combinations were explored on them. A pass shows that one procedure fixed in advance, not the best of six, beats NBLAST by the A4 margin; it removes the selection concern. It is not a replication on unseen data. The result is reported as-is, whichever way it goes, in `results/seeded_matching/SUMMARY.md`.

## Commands

```bash
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage seeds
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage match --targets banc_cord
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage match --targets banc
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage score
```

The unseeded comparison needs the A2-ii matchings of `results/morphology_baseline/SUMMARY.md` ("Commands").
