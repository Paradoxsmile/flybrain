# Hemibrain test protocol

Fixed and committed before any prediction on the hemibrain was compared with a hemibrain label. After this commit, only data-loading fixes may change, and every deviation is reported with the results. Analyses not listed here are labeled exploratory.

## Purpose

The pre-registered BANC test ([docs/banc_protocol.md](banc_protocol.md)) showed that prior matching carries over to a new connectome but the selection by type consistency does not. A development sandbox of degraded copies of FAFB and MCNS with known labels (README, "Development sandbox and a revised selection rule") then led to three changes: a new selection rule, a type-size prior restricted to the volume a target covers, and a normalization against that part of the source. The hemibrain (Scheffer et al., 2020) has not been used in developing any of them. It tests the revised pipeline on a connectome that differs from FAFB in a way the sandbox only approximated: a partial volume (the central brain, mostly of the right hemisphere, without the optic lobes) with neurons cut by the volume boundary.

## What was looked at before this protocol

Metadata columns and counts, label overlap with the 434 FAFB types, the share of cropped neurons, graph statistics, and the super-class composition (as a check of the coverage-projected expectation: 67 / 14 / 11 % central / visual projection / sensory expected, 73 / 14 / 11 % in the metadata). The harmonized edgelist was found to repeat rows (931,329 exact copies); they are dropped (`flybrain/hemibrain.py`). No model was run on the hemibrain.

## Data

- Hemibrain v1.2.1, harmonized by the BANC project, from `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/hemibrain_121/` (CC BY 4.0): `hemibrain_121_meta.feather` (MD5, base64 `X5fZkw3EHfmigoMGwvq0Bg==`) and `hemibrain_121_simple_edgelist.feather` (`0yUATeLjt4ZAnrGWAhSiMA==`).
- Neurons: all 25,348 traced neurons; 4,039 (16 %) are flagged as cropped by the volume boundary.
- Connections between them, copies dropped. At FAFB's threshold of 5 synapses: 26.7 connections and 387 synapses per neuron (FAFB: 26.8 and 364).
- Alignment variants by the rules used for BANC: `raw` (threshold 5) and `scaled` (threshold 5, counts × 0.94 so that synapses per neuron match FAFB's). The density rule gives threshold 5, the same graph as `raw`.

## Labels (used only for evaluation)

- **Super classes:** hemibrain `super_class` mapped to the FAFB classes (`SUPER_CLASS_MAP`); 25,342 neurons.
- **Cell types:** the hemibrain type where it is one of the 434 FAFB types: 7,639 neurons, 202 types; 2,322 of them cropped, 2,772 visual projection neurons with their somata in the optic lobe. FAFB adopted most central-brain type names from the hemibrain, by matching neurons across the two datasets with morphology and connectivity (Schlegel et al., 2024), and hemibrain types were themselves defined with morphology and connectivity (Scheffer et al., 2020); agreement is therefore partly circular, as for MCNS and BANC.
- 77 % of FAFB's labeled neurons belong to types that do not occur in the hemibrain.

## Method (fixed by the development sandbox)

- GNNs: the FAFB GNNs of `results/transfer2` (3 seeds per label; SHA-256 in [docs/banc_protocol.md](banc_protocol.md)). Nothing is trained on or for the hemibrain.
- Candidates: variant (`raw`, `scaled`) × normalization (target, source, coverage) × rho (0, 0.25, 0.5, 0.75, 1).
- Coverage `central_brain`: FAFB neurons with at least 10 % of their synapses outside the optic-lobe neuropils (ME, AME, LO, LOP, LA) and the ocellar ganglion count as inside the volume (`flybrain/coverage.py`). The type-size prior is the type distribution the GNN predicts for them (`--prior coverage`), and the coverage normalization gives the target features their mean and spread.
- Selection: `size_agreement` (`flybrain/selection.py`), the best mean rank of predicted type sizes close to the prior and agreement between the three seeds. NTAC refinement seeded where all three GNNs agree, as before.

## Runs

1. **Primary:** `python experiments/b_gnn_celltypes/label_free.py --target hemibrain --coverage central_brain --prior coverage --select size_agreement --out label_free_hemibrain`
2. **Secondary, to isolate the coverage prior:** the same with `--prior source --out label_free_hemibrain_source_prior`.

## Reported, whatever the outcome

For each run and label: accuracy / macro-F1 without prior matching (chosen variant and normalization, rho 0), after prior matching (the selected candidate), after NTAC, and of the best candidate (post hoc); the chosen candidate and the choice of every rule in `flybrain/selection.py` (including the old consistency rule); Spearman correlations of type consistency, seed agreement and type-size divergence with accuracy; accuracy for cropped and complete neurons and by region (central brain, optic lobe).

Context, not targets: FAFB → MCNS 92.6 % on cell types; in the sandbox, MCNS restricted to the central brain reached 46.9 % (best candidate, no cropping) and 36.3 % (16 % cropped), against 69.9 % for the same neurons in the full volume.

## Expectations stated in advance

1. The selected candidate is within 5 points of the best candidate, for cell types and for super classes.
2. With the coverage prior the best cell-type candidate is better than with the source prior (sandbox: 36.3 vs. 30.5 %).
3. Complete neurons are typed more accurately than cropped ones.
4. Cell-type accuracy stays below 60 %: a partial volume removes the partners outside it, which the sandbox showed to cost about 23 points.
