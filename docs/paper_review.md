# Review of the paper draft (v0.4, `paper/main.tex`, 21 pages)

Reviewed on 2026-10-28, after PR #12 (restructure around rare types) and PR #13 (pre-registered seeded-matching test) were merged. Page numbers refer to the built `paper/main.pdf`; line numbers to `paper/main.tex`.

## Overall rating

**Ready for bioRxiv?** Almost. One fix is required before posting: the "wiring adds to morphology" claim (abstract, contribution 3, Section 4.5, Discussion, Conclusion, Figure 1, README) is contradicted by the paper's own pre-registered test and must be replaced. A second claim should be weakened before posting: that the BANC morphology reference shows the rare-type gain "is not an artifact of connectivity-based labels" (see A2). With these two fixes the paper is honest and postable. Everything else below improves it but does not block it.

**Level.** Clearly above a typical bachelor thesis. The pre-registered protocols with commit hashes, the reported failures and the per-table reproduction map are rarer than they should be, even in published work. As a research paper it is at the level of a **strong master's thesis or a good workshop paper** (e.g. a NeurIPS / ICML workshop on graph learning or a computational neuroscience workshop). For a journal (e.g. *PLOS Comput Biol*, *eLife*, *Bioinformatics*) it would need: a reference that is independent of both NBLAST and connectivity, comparison with an existing cross-dataset typing procedure on rare types, and a shorter, more focused text.

**Main scientific strengths**
- A clear, useful idea: a classifier cannot learn a type from 1–2 examples, but graph matching pairs neurons without learning; the transferred GNN only has to decide *which* neurons to match.
- Large, consistent gains in three pre-registered tests on held-out data (BANC brain 22→41 %, BANC cord 40→61 %, MCNS cord 72→82 %), with bootstrap intervals.
- Honest bookkeeping: four (now five) pre-registered tests, failures reported (type-consistency rule, coarse fallback, hemibrain, MCNS threshold, now seeded matching).
- The oracle-candidate diagnostic cleanly separates "candidate selection" from "matching" as error sources.
- Full reproducibility map (`docs/reproduce.md`).

**Main scientific weaknesses**
- **References are not independent.** For BANC rare types, the "alignment" and "morphology" references agree on 98.7–99.5 % of neurons; both come from the same reviewed `fafb_match`, which was proposed by NBLAST and reviewed partly with connectivity. MCNS's rare-type labels also come from connectivity-aware matching. No rare-type result is measured against labels made without both NBLAST and connectivity.
- **Weak baselines for rare types.** The comparisons are stage 1 (the paper's own GNN) and raw NBLAST. Missing: how a simple label-free alternative does (e.g. graph matching on candidates chosen by degree / super-class only, or NTAC's unsupervised mode), and how far the result is from what the connectome teams' own matching achieves.
- **One development pair** (FAFB↔MCNS) and few held-out targets, which the Limitations already say.
- **The morphology story is now a mixed result** (exploratory +6.0, pre-registered +2.3 below its 3-point bar, cord +23); it must be told precisely, not as a headline.

## Flaws by category

Severity: **major** = changes what a reader concludes or blocks posting; **minor** = clarity or polish.

### A. Argument and claims

| # | Location | Severity | Problem | Proposed fix |
|---|---|---|---|---|
| A1 | Abstract l.41; contribution 3 l.57; Sec. 4.5 l.318; Discussion l.336; Conclusion l.366; Fig. 1; README l.26 | **major** | "Wiring complements/adds to morphology … 67→73 %" is the best of six exploratory combinations. The pre-registered test of one fixed combination gave +2.3 (95 % CI 1.3–3.2), below its 3-point rule. | Replace with the confirmed result: +2.3 points on the BANC brain (rule not met), +23 in the cord, where automatic NBLAST is weak. Framing: wiring is an independent signal that adds a little where morphology is strong and a lot where it is weak. Add the confirmatory test as a short subsection and table (draft in `results/seeded_matching/paper_section.md`), and the pre-registered test as a fifth row of Table 10. |
| A2 | Sec. 4.2 l.235 and l.252; Discussion "Circularity" l.342; Contribution 2 l.55; abstract "References built from morphology … gave the same gains" | **major** | The "morphology reference" is called "independent of connectivity", and its agreement with the alignment reference is used to argue the gain "is not an artifact of connectivity-based labels". But on the rare stratum the two references agree on 98.7–99.5 % (`docs/banc_rare_protocol.md` l.38): both are the type of BANC's reviewed `fafb_match`, which was NBLAST-proposed and reviewed partly with connectivity. Same gain on two near-identical label sets is not independent confirmation. | Say plainly that for rare types the two BANC references nearly coincide and both come from reviewed NBLAST matches, so they are not an independent check. Keep the BANC **cord curated reference** as the better argument, if its provenance differs (check). Add the NBLAST circularity: comparisons against NBLAST on BANC favour NBLAST (99.9 % "correct" where review kept the top hit, 12.0 % where it replaced it). |
| A3 | Table 1, BANC "types assigned by connectivity alignment to FAFB" | **major** (factual) | Contradicts the circularity finding: BANC's types come from NBLAST proposals plus manual review that also used connectivity. | "types from reviewed NBLAST matches to FAFB (review also used connectivity)". |
| A4 | Contribution 1 l.53; Discussion l.334 "matching wins by 12–16 points" | minor | "92 % against 76 %" compares matching *given oracle candidates* with the unrestricted GNN. The fair comparison is 79.8 vs 92.2 (same candidates) or 78.9→85.8 (label-free). | Quote the fair pair: "same candidates: 92 vs 80 %; fully label-free: 79 → 86 %". |
| A5 | Fig. 1 vs Sec. 4.1 | minor | Stage 1 on FlyWire left→right is 78.9 in Fig. 1 (majority vote) but 75.9 in the text (mean of seeds); MCNS two-stage is 81.6 in the figure (broad) but 74.5 in Table 4 (strict). | Use one definition throughout, and mark in the figure which value is which (the caption tries but is hard to parse). |
| A6 | Abstract | minor | Mixes strict (BANC brain 41 %) and broad candidates (cords 61, 82 %) without saying so. | One clause: "(cords: with an improved candidate rule developed in between)". |
| A7 | Sec. 4.4 l.295 | minor | The MCNS-cord expectation failed (+10.2 < 15); the post-hoc explanation "should have set it relative to the gap" is fair but reads like an excuse right after the miss. | Keep, but move to the Discussion's pre-registration paragraph where it already appears. |
| A8 | Limitations l.362 | **major** (stale) | Says seeded graph matching "is the natural next step and was not tried". It was tried and hurt. | Replace: seeding with hard NBLAST anchors lowered accuracy (−1.8 brain, −11.4 cord); soft seeds are the next step. |
| A9 | Throughout | minor | No statement that BANC labels were seen before the confirmatory test (pre-registered in procedure, not on unseen data). | One sentence in the new subsection. |

### B. Structure (the PR #12 restructure)

What works:
- One main story (rare types), with stage 1 on common types reduced to a summary section plus Supplement. Right decision.
- Figure 1 (pipeline + all results in one dot plot) is a good "paper in one picture".
- Pre-registration table in the Discussion.

What does not work:

| # | Location | Severity | Problem | Proposed fix |
|---|---|---|---|---|
| B1 | Sec. 4.2–4.4 | **major** | Results read as a lab notebook in time order: test → "because BANC was then spent" → fix on sandbox → next test. The reader has to track which candidate rule, fallback and reference held when. | Organize results by question, not by time: (1) does matching beat classifying for rare types? (development); (2) does it hold on held-out data? (BANC brain and both cords together, one table, one figure); (3) where is the error? (candidates vs matching, incl. the candidate-rule improvement); (4) how does it relate to morphology? (exploratory + confirmatory). Chronology goes into Table 10 only. |
| B2 | Sec. 3, Table 2 (p. 5) and Sec. 3.1 | **major** | The methods spend most space on stage 1's preprocessing selection (30 candidate preprocessings, two selection rules, prior-matching strength ρ) and a 20-row names table, before the reader knows why it matters. For the rare-type story, stage 1 is "a GNN that transfers without labels". | Shorten Sec. 3.1 to one paragraph (model, prior matching, label-free choice in one sentence); move the selection-rule details to Supplement S3. Shrink Table 2 to ~8 rows (see C3). |
| B3 | Sec. 4.6 "Stage 1 on its own" (p. 10) and contribution 4 | minor | A long dense paragraph block about common types, the hemibrain and hand labels; it is a second paper in miniature at the end of Results. | Either cut to ~1/3 (three sentences: transfers on complete targets, fails on partial volumes, worth ~1 % hand labels) or move before 4.1 as "Stage 1 transfers without labels" (it is a prerequisite). Drop contribution 4 or merge it into 1. |
| B4 | Intro | minor | Four contributions, of which 2 (pre-registration) is a method of evaluation, not a finding. | Three contributions: the method + its held-out gains; where it fails (candidates, partial volumes); how it relates to morphology. Pre-registration as a sentence. |
| B5 | Discussion | minor | Six bold paragraphs, partly repeating Results numbers. | Four: what the result means; morphology (new framing); circularity and limits; pre-registration lessons. |
| B6 | Page layout | minor | Tables 1 and 2 float to pp. 4–5 ahead of their text; pp. 8–9 are mostly tables. | Fewer tables (C1) fixes most of this; otherwise `[htbp]`. |

### C. Tables

| # | Location | Severity | Problem | Proposed fix |
|---|---|---|---|---|
| C1 | Main text | **major** | 9 tables in the main text (10 after the new test) plus 6 in the Supplement; each result has its own table with "accuracy % / macro-F1" pairs. Too many for a 12-page paper. | Merge Tables 5 and 7 (all held-out tests: one row per target, columns stage 1 / two-stage / oracle, with CI); merge Tables 3 and 4 (development); fold Table 8 + the confirmatory table into one morphology table. Target: 5–6 main tables. Move Table 1 to the Supplement or reduce to 4 columns. |
| C2 | Table 1 (p. 4) | minor | 7 columns in `\scriptsize`, cells with long prose ("Particularities"), row "Connections / synapses per neuron (≥ 5 synapses)". | Transpose (one row per connectome) or cut to neurons, density, labels, role; prose into a footnote. |
| C3 | Table 2 (p. 5) | **major** | 20 rows, includes code options and obsolete variants (type-consistency rule, coarse fallback, NBLAST-first combination, "reference A/B"). Code options belong in `docs/reproduce.md`, which already lists them. | Keep only names the reader meets in the main text; drop the code column; add "seeded matching". |
| C4 | Tables 3–8 | minor | "accuracy % / macro-F1" in every cell doubles the numbers; macro-F1 is rarely discussed. | Accuracy in the main tables; macro-F1 in the Supplement, or as a second column only where discussed. |
| C5 | Table 7 | minor | Column headers "BANC cord, alignment ref. (5,989)"; dashes for missing macro-F1 without explanation. | Short headers + a caption note for the dashes. |
| C6 | Table 10 | minor | Needs a fifth row (the seeded-matching test, `3949b10`, "1: not met (+2.3 < 3)"). | Add; renumber in `docs/reproduce.md`. |

### D. Figures

| # | Location | Severity | Problem | Proposed fix |
|---|---|---|---|---|
| D1 | Fig. 1b | **major** | Shows the exploratory "matching-first combination" (72.5) as the green headline marker for BANC, which is now superseded; six marker types, two of them only on two rows. | Replace the green triangle with the pre-registered seeded combination (68.7 brain, 53.7 cord) or drop morphology from Fig. 1 and give it its own small panel. Reduce the legend. |
| D2 | Fig. 1a | minor | Pipeline panel takes ~40 % of the figure height for five boxes; lots of white space; arrow labels small and grey. | Make it one compact row; put "each matched candidate takes its partner's type" inside the stage-2 box. |
| D3 | Fig. 1b | minor | Filled vs open blue circles (broad vs strict) are hard to tell apart; the reader must read the caption to know that the BANC brain filled dot is exploratory. | Mark exploratory points with a lighter colour or hatched marker and say "exploratory" in the legend. |
| D4 | Fig. 1 caption | minor | One very long sentence with nested parentheses. | Split into: what panel a shows; what each marker is; which rows are pre-registered. |
| D5 | Fig. S2 (transfer) | minor | Markers overlap on FAFB→MCNS and hemibrain rows; orange and green mean different things than in Fig. 1 (NBLAST / combination there, NTAC / rule here). | Offset markers vertically; use a consistent colour code across figures. |
| D6 | Fig. S1 | minor | Imported as a PNG from `docs/`, different style and font from the other figures. | Regenerate in `make_figures.py` as PDF. |
| D7 | All | minor | No figure shows the morphology result or the circularity split, which is now an important finding. | Optional: small bar chart "review kept vs replaced the NBLAST hit" (NBLAST 99.9 / 12.0, combination 84.5 / 39.1). |

### E. Language

| # | Location | Severity | Problem | Proposed fix |
|---|---|---|---|---|
| E1 | Throughout Results | **major** | Very high number density: many sentences carry 4–8 numbers (e.g. l.252, l.295, l.318). The point gets lost. | One claim per sentence, 1–2 numbers each; the rest stays in tables. |
| E2 | Throughout | minor | Internal or lab jargon: "rare stratum", "2–4 stratum", "BANC was then spent", "sandbox", "candidate preprocessings", "the ≥ 10 model", "reference A/B" (in Table 2), "R1-6", "matching-first", "A2-ii". | Define once or replace: "rare-type neurons", "used up for testing", "degraded development copies", "the model trained on types with ≥ 10 neurons". |
| E3 | Title | minor | 27 words with a colon. | E.g. "Typing rare neurons of an unlabeled connectome by graph matching, guided by a transferred GNN". |
| E4 | Throughout | minor | "Morphology reference", "alignment reference", "curated reference" vs "reference A/B" in result files; "FAFB" vs "FlyWire" used interchangeably. | Use FlyWire for the dataset and FAFB only for the volume name, or define once. |
| E5 | Abstract | minor | 8 sentences, ~190 words, 9 numbers; the last sentence (hemibrain) feels tacked on. | Rewrite with the new framing, ≤ 180 words. |
| E6 | Front matter | minor | "Draft v0.4" in the date; no affiliation, no e-mail, no ORCID. | Remove the draft label for posting; add affiliation and contact. |

### F. References and reproducibility

| # | Location | Severity | Problem | Proposed fix |
|---|---|---|---|---|
| F1 | `references.bib` | **major** (for posting) | Author fields are surnames only ("Bates and Phelps and Kim and Yang and others"), so the bibliography prints without initials; several entries "X and others". | Full author lists (or first author + "and others" with initials) from the DOIs. |
| F2 | `references.bib` | minor | Missing citations: seeded graph matching (Fishkind et al. 2019, *Pattern Recognition*), Sinkhorn scaling (Cuturi 2013), the software used (PyTorch Geometric, graspologic/SciPy `quadratic_assignment`), the BANC preprint for the review procedure if it differs from the *Nature* version. | Add. |
| F3 | Methods | minor | Missing: how accuracy and macro-F1 are computed (over which types; how unmatched candidates count); bootstrap details (types resampled, matching not re-run); hardware and run times beyond stage 2. | One "Evaluation" paragraph in Sec. 3. |
| F4 | Code and data | minor | `[Zenodo DOI]` placeholder (expected; leave until release). Protocol list misses the confirmatory pre-registration (`docs/preregistration_confirmatory.md`, `3949b10`). | Add the fifth protocol. |
| F5 | Limitations l.362 | minor | All-neuron numbers without the fallback "come from runs reported in that test but not from a fresh one". | Fine as is; keep. |
| F6 | `docs/reproduce.md`, README | minor | Table numbers will change; README still headlines "wiring adds to morphology" (+6.0). | Update with the paper. |

## Recommended order of work

1. Required: framing update (A1, A2, A3, A8, A9, D1, C6, F4, F6).
2. High value, moderate effort: merge tables (C1, C3, C4), shorten numbers per sentence (E1), fair comparison in contribution 1 (A4), bibliography (F1).
3. Larger restructure (B1–B3), only if the user wants it: results by question instead of by time, stage 1 compressed.
4. Polish: figures D2–D7, title, abstract, jargon.

## Applied (v0.5, branch `claude/paper-review`)

All four steps were applied at the user's request ("Everything").

- **Framing (A1–A3, A8, A9).** Abstract, contributions, Section 4.4, Discussion, Conclusion and README now report the pre-registered seeded-matching test (+2.3 points, 95 % CI 1.3–3.2, rule not met; cord +23.2) and the NBLAST circularity of BANC's references. The claim that BANC's second references are independent was removed; Table 1's description of BANC's typing was corrected. Limitations no longer say seeded matching was not tried.
- **Structure (B1–B5).** Results are ordered by question: 4.1 matching vs. classifying (development), 4.2 held-out tests, 4.3 candidate selection, 4.4 wiring and morphology, 4.5 stage 1 on common types (shortened). Stage-1 details (alignment variants, preprocessings, selection rules) moved to Supplement S1. Three contributions instead of four. The Discussion has four paragraphs; the Limitations are merged into "Circularity and limitations". New Evaluation paragraph (F3).
- **Tables (C1–C6).** Main-text tables went from 9 to 7: development (old 3 + 4), held-out tests with intervals (old 5 + 7), morphology with the confirmatory test (old 8 + new). Table 1 has one row per connectome; Table 2 has 12 rows and no code column. Main tables show accuracy only; macro-F1 is in the new Table S7. Table 7 has a fifth row for the seeded-matching test.
- **Figures (D1–D7).** Figure 1 no longer shows the exploratory morphology combination and marks the exploratory BANC-brain point. The new Figure 2 shows the pre-registered combination and the review-kept / review-replaced split. Figure S2 has offset markers and its own colours. Figure S1 is redrawn in the paper's style as a PDF.
- **Language (E1–E6).** Fewer numbers per sentence, no internal names in the main text ("sandbox", "rare stratum", "≥ 10 model", "reference A/B"), shorter title, no "Draft" label. Affiliation and contact are still missing: they need the author's input.
- **References (F1–F4).** Full author names or first author with initials, NeuNet's full title, and new citations for seeded graph matching, Sinkhorn, PyTorch Geometric and SciPy. The fifth protocol is listed in Code and data availability.
- One correction found on the way: old Table 8 gave macro-F1 0.30 for NBLAST in the cord; the result file gives 0.29.

## Second round (external comments, 2026-10-29)

- "Independent signal" → "complementary" (abstract, contribution 3, Discussion, Conclusion); the Discussion says the two err on different neurons, not that they are statistically independent.
- The fifth protocol is called a pre-specified analysis (labels seen before) in the abstract, Methods, Section 4.4, Table 6, Table 7, Figure 2, Discussion, Conclusion, README and the reproduction map.
- Why seeding: all six unseeded combinations, including the second best (70.6 %), had known results, so registering one of them would not have removed the selection concern; seeding tests whether confident morphology matches can anchor the matching.
- "+23 points over unreviewed NBLAST" in the abstract, Discussion and Figure 2.
- Table 4 and 6 captions and the Limitations state that the intervals resample types only and leave out the run-to-run variation of stage 1 (one run per target, three GNN seeds combined) and of the matching.
- All commit hashes in Table 7 and in the reproduction map are on the public `main`. `docs/reproduce.md` covers Table 6 and Figure 2 (`results/seeded_matching/metrics.csv`). The Zenodo DOI needs a GitHub release archived by Zenodo, which only the author can create.
- Reviewer ask answered (exploratory, run after the pre-specified analysis): matching with NBLAST-only candidates and no GNN anywhere reaches 58.9 % on the BANC brain, 7.6 points below NBLAST alone, and 45.6 % in the cord. Adding the GNN's candidates raises the same combination to 70.6 and 65.1 % (+11.7 [11.0, 12.4]; +19.5 [18.2, 20.8]). Added to Table 6, Section 4.4, contribution 3, the Discussion, README and `results/morphology_baseline/SUMMARY.md`. NBLAST as a soft prior was not run: it needs a weighting chosen on BANC labels that are already spent.
