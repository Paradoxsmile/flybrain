# Seeded graph matching: the confirmatory test of "wiring adds to morphology"

Pre-registration: [docs/preregistration_confirmatory.md](../../docs/preregistration_confirmatory.md), committed in `3949b10` with the analysis code, before the matching was run. Script: `experiments/b_gnn_celltypes/seeded_matching.py`. Numbers: `metrics.csv` (accuracies), `delta.csv` (contrasts with 95 % CIs), `decision.json` (primary).

## Result: the pre-registered rule was not met

On BANC brain rare types (reference A, 2–4 FAFB neurons per type, 14,512 neurons, 5,416 types), the pre-specified combination "seeded matching's type where it assigned a FAFB neuron, NBLAST elsewhere" reaches **68.7 %**, NBLAST alone **66.5 %**:

**Δ = +2.3 points, 95 % CI 1.3–3.2** (10,000 resamples of types).

The interval excludes 0 but Δ is below the 3-point bar, so the outcome is **"not confirmed: wiring adds less than 3 points"**. Under A4 of [finalize_plan.md](../../docs/finalize_plan.md) this is the middle case: wiring is a cross-check and a fallback where morphology is missing or wrong, not a headline gain. The exploratory +6.0 (best of six combinations) does not survive as a confirmed effect of that size.

| Labeling (accuracy %, rare stratum, ref A) | Brain | Cord |
|---|---|---|
| NBLAST alone | 66.5 | 30.5 |
| Two-stage, unseeded (A2-ii candidates) | 62.6 | 64.7 |
| Two-stage, seeded | 61.6 | 55.4 |
| Matched, else NBLAST, unseeded | 70.6 | 65.1 |
| **Matched, else NBLAST, seeded (pre-registered)** | **68.7** | 53.7 |
| Δ seeded combination − NBLAST [95 % CI] | **+2.3 [1.3, 3.2]** | +23.2 [21.8, 24.5] |
| Δ seeded − unseeded combination [95 % CI] | −1.8 [−2.4, −1.3] | −11.4 [−12.5, −10.2] |

## Secondary analyses

- **Seeding hurt.** Against the unseeded matching on the same candidates, seeds lower accuracy by 1.8 points (brain) and 11.4 (cord). The seeds are NBLAST top hits that are unique within their hemisphere, and they are fixed: 15.5 % of the brain seeds in the stratum carry a wrong type (NBLAST correct on 84.5 % of 5,362), and 32.4 % in the cord (67.6 % of 1,529). Fixing them removes the matching's chance to correct them and pulls their neighbours toward wrong partners. Unseeded matching, which uses morphology only to choose candidates, was the better design here; this was not tested as a pre-registered contrast.
- **Cord.** The seeded combination beats NBLAST by 23.2 points (the rule's "confirmed" case), because BANC's automatic cord NBLAST hits are weak (30.5 %). It is still 11.4 points below unseeded matching.
- **Reference B.** Brain: Δ = +1.4 [0.4, 2.4], also "not confirmed"; cord: +23.1 [21.7, 24.5].
- **Over all labeled neurons** (brain): 61.0 % seeded combination, 56.4 % NBLAST.

## Circularity of the reference (feasibility finding)

BANC's types were assigned by NBLAST proposals, manual review and correction (Bates et al., bioRxiv 2025.07.31.667571). On the brain rare stratum, reference A equals the type of BANC's reviewed match (`fafb_match`) for 99.8 % of neurons, and the reviewed match is the NBLAST top hit for 67.5 %. NBLAST alone is therefore "right" on 99.9 % of the neurons where review kept its top hit and on 12.0 % where review replaced it. Split this way (descriptive, `delta.csv`):

| Brain, rare stratum, ref A | n | NBLAST | Seeded combination | Δ [95 % CI] |
|---|---|---|---|---|
| Review kept the NBLAST hit | 9,002 | 99.9 | 84.5 | −15.4 [−16.3, −14.4] |
| Review replaced the NBLAST hit | 4,331 | 12.0 | 39.1 | +27.1 [25.5, 28.6] |
| Not a seed | 9,150 | 55.9 | 59.5 | +3.6 [2.0, 5.1] |

Where review kept NBLAST's hit the reference is NBLAST's answer by construction, so every comparison against NBLAST on BANC is tilted toward NBLAST. Where reviewers overrode NBLAST, wiring recovers the reviewed type in 39 % of cases against 12 % for NBLAST. (Review also used connectivity, which may favor wiring in that split.) The cord shows the same pattern (−6.3 and +32.9 points).

## What this does and does not show

- A pass would have removed the "best of six" selection concern; the test instead shows that one reasonable pre-specified combination gains 2.3 points, not 3 or more. The exploratory combinations (+4.1 to +6.0) remain exploratory.
- BANC's labels were seen before this test; it is pre-registered in procedure, not on unseen data.
- The CI is conditional on the matching (types resampled, matching not re-run).
- MCNS has no automatic morphology match to FlyWire, so the MCNS combination test could not be run.

## Deviations from the pre-registration

- The `outcome` column in `delta.csv` was first filled for every non-descriptive row with the NBLAST wording of the rule; it is now filled only for contrasts against NBLAST alone, where the rule is defined. Numbers are unchanged.
- The cord was scored once before the brain matching finished, to check the scoring code; the scoring was not changed afterwards beyond the label above.

## Commands

```bash
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage seeds
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage match --targets banc_cord
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage match --targets banc
uv run python experiments/b_gnn_celltypes/seeded_matching.py --stage score
```

Needs the stage-1 runs and the unseeded A2-ii matchings of `results/morphology_baseline/` ("Commands" there). Runtime on a CPU: cord 8 min, brain 35 min.
