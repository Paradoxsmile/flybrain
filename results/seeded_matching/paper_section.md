# Draft for the paper: confirmatory test of "wiring adds to morphology"

*Draft text for Section 4.4 (after the morphology baseline) and the Limitations. Not yet in `paper/main.tex`. Numbers from `results/seeded_matching/` and `results/morphology_baseline/`.*

---

**A pre-registered test of the combination.** The morphology comparison above is exploratory: BANC's labels were known, and the best of six combinations was reported. We therefore fixed one further procedure, its contrast and the decision rule before running it (`docs/preregistration_confirmatory.md`). A confident NBLAST match, meaning a candidate whose NBLAST top hit is a rare-type FlyWire neuron in the same hemisphere and is the top hit of no other candidate, is fixed as a seed (6,138 seeds in the brain), and FAQ matches the remaining candidates around the seeds. Each neuron takes the matched type where the matching assigned one, and NBLAST's type elsewhere. The rule was to call the gain confirmed if it reached at least 3 points over NBLAST alone with a 95 % interval excluding zero. On the 14,512 rare-type neurons of the BANC brain, the combination reached 68.7 % against 66.5 % for NBLAST, a gain of 2.3 points (95 % CI 1.3–3.2, bootstrap over 5,416 types). Wiring therefore adds to morphology, but by less than the pre-registered margin, and we do not claim a confirmed gain of the exploratory size (+6.0). Seeding itself was harmful. With unseeded matching on the same candidates, the combination reached 70.6 % (brain) and 65.1 % (cord) instead of 68.7 % and 53.7 %. About 16 % of the brain seeds and 32 % of the cord seeds carry a wrong type, and fixing them prevents the matching from correcting them. In the nerve cord, where BANC's automatic NBLAST hits are weak, the seeded combination still beat NBLAST by 23.2 points (21.8–24.5).

**The reference favors NBLAST.** BANC's types were assigned by reviewing and correcting NBLAST proposals (Bates et al. 2025). On the brain rare stratum, the reference type equals the type of BANC's reviewed match for 99.8 % of neurons, and that match is the NBLAST top hit for 67.5 %. NBLAST's accuracy against BANC's types is therefore largely the share of top hits that reviewers kept: 99.9 % correct where the hit was kept, 12.0 % where it was replaced. Where reviewers replaced the NBLAST hit, the combination recovers the reviewed type for 39.1 % of the neurons (+27.1 points). Where they kept it, the combination loses 15.4 points against a reference that is NBLAST's own answer. Comparisons of wiring against morphology on BANC are thus tilted toward morphology. A test on a dataset whose types were assigned independently of NBLAST would be cleaner, but MCNS, the other target, provides no automatic morphology match to FlyWire.

*Table X. Confirmatory test, rare types (2–4 source neurons), reference A, accuracy %. Δ: 95 % CI from resampling types.*

| Labeling | BANC brain (14,512) | BANC cord (5,989) |
|---|---|---|
| NBLAST alone | 66.5 | 30.5 |
| Two-stage, unseeded | 62.6 | 64.7 |
| Two-stage, seeded | 61.6 | 55.4 |
| Matched, else NBLAST, unseeded (exploratory) | 70.6 | 65.1 |
| Matched, else NBLAST, seeded (pre-registered) | 68.7 | 53.7 |
| Δ pre-registered − NBLAST | +2.3 (1.3–3.2) | +23.2 (21.8–24.5) |
| Pre-registered rule (Δ ≥ 3, CI > 0) | not met | met |

---

*Suggested consequences for the rest of the paper.* The abstract should drop "wiring adds to morphology" as a headline and use the A4 middle framing instead: wiring is a label-free cross-check that corrects NBLAST where it is wrong, with a pre-registered gain of 2.3 points on BANC brain rare types and larger gains where automatic morphology is weak (the cord). The Limitations should state the circularity of BANC's reference with NBLAST, and that seeding with hard NBLAST anchors hurt; soft seeds (NBLAST as a prior term rather than fixed pairs) are the natural next step.
