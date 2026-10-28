# Finalize plan: morphology baseline and v1.0

Goal: answer the reviewer question "why wiring alone, when morphology is free?" and release the project. This plan is committed before any morphology baseline is run, so the framing decision in A4 is fixed in advance. BANC labels are already known, so every result in part A is **exploratory**, not a pre-registered test.

## A. Morphology baseline (about 3-5 days)

Data: `data/raw/banc/banc_888_meta.feather` already has each neuron's NBLAST top match: `fafb_nblast_match` (127,298 neurons) and `manc_nblast_match` (26,985). These are the BANC project's morphology matches, before review. MCNS has `somaLocation` (141,781 neurons). Nothing new needs to be downloaded.

**A1. Morphology alone.** Give each BANC neuron the type of its NBLAST match: FAFB for the brain, MANC for the cord.
- Evaluate on the rare stratum and on all labeled neurons, against **reference A** only. Reference B was built from NBLAST plus review, so it would be circular here. Report it only as a footnote.
- Report accuracy on matched neurons and on all neurons. For the all-neuron number, unmatched neurons fall back to stage 1.

**A2. Complementarity.**
- Build a 2×2 table: NBLAST type and two-stage type, agree or disagree. Give the accuracy in each cell.
- Test two label-free combinations:
  - (i) use the NBLAST type where there is a match, otherwise the two-stage type;
  - (ii) **NBLAST as candidate selection**: the candidates are the `union` candidates plus the neurons whose NBLAST match is a rare type, then matching runs unchanged.

  Candidate selection is the bottleneck: oracle candidates are 5-25 points better. Option (ii) is therefore the most promising use of morphology.

**A3. Position baseline (optional, within MCNS).**
- Type right-hemisphere rare-type neurons by their nearest left neuron in soma position, after mirroring x around the median x. That mirror is label-free.
- Compare with the GNN and with graph matching on the same stratum.

**A4. Framing decision.** On the BANC rare stratum, against reference A, with Δ = (best combination) − (NBLAST alone):
- **Δ ≥ 3 points**: headline becomes "wiring adds to morphology". Rare types are where it helps.
- **0 ≤ Δ < 3**: wiring is a cross-check and a fallback where morphology is missing or ambiguous. Report how many neurons lack an NBLAST match.
- **NBLAST alone is best by a wide margin**: say so in the abstract. The contribution narrows to label-free candidate selection, the pre-registered evaluation and the sandbox.

Outputs: `experiments/b_gnn_celltypes/morphology_baseline.py`, `results/morphology_baseline/*.csv`, one table in Section 4.4 and one paragraph in the Discussion.

## B. Paper (about 1 week)

1. **Focus.** Rebuild the abstract around rare types, the pre-registered tests and the A4 outcome, with at most 8 numbers. Keep 3 contributions: the two-stage pipeline, pre-registration plus sandbox, and the failure modes. Shorten Sections 4.1-4.2 to support material.
2. **Uncertainty.** Bootstrap 95 % confidence intervals over types for every pre-registered number in Tables 6 and 8. These come from saved predictions; no reruns.
3. **Seeded matching.** Do not run it. Name it in the Limitations as the natural next step: confident common-type neurons as anchors.
4. **Main figure.** A pipeline diagram plus one panel of stage 1 / two-stage / oracle (and NBLAST) for all five pairs.
5. **References.** Close the open checks in `docs/literature.md` section 5. Verify Berg et al. 2026 (volume, pages, DOI) and the Bates et al. 2026 author list. Add Shiu et al. 2024 only if it is cited.
6. **Placeholders.** Remove every `[bracket]` item: repository URL, Zenodo DOI.
7. **Length.** Target about 7,000 words in the main text; move the details to the supplement.

## C. Release (1 day)

1. Pin `docs/reproduce.md` to the final result files, and make sure `uv run pytest` and CI are green.
2. Tag `v1.0`, archive it on Zenodo, and add the DOI to `CITATION.cff` and the paper.
3. Post a preprint on bioRxiv (Neuroscience), then submit to *Network Neuroscience* (first choice) or *PLOS Computational Biology*.

## Done when

- [ ] A1-A2 results are committed and A4's framing is applied to the abstract.
- [ ] No `[bracket]` items remain in `paper/draft.md`; references are complete.
- [ ] CIs are in Tables 6 and 8.
- [ ] `v1.0` is tagged, the Zenodo DOI is in place and the preprint is posted.
