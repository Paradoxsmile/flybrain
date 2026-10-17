# BANC test protocol

Fixed and committed before any prediction on BANC was compared with a BANC label. After this commit, only data-loading fixes may change, and every deviation from this protocol is reported together with the results. Analyses not listed here are labeled exploratory.

## Purpose

The label-free transfer pipeline (`experiments/b_gnn_celltypes/label_free.py`: type-consistency selection, type-size prior matching, GNN-seeded NTAC) was developed on FAFB → MCNS while looking at its transfer errors, and checked in the reverse direction on the same pair. BANC (Bates et al., 2026) played no part in its development: an independently reconstructed female brain with its own segmentation and synapse detection. The test applies the pipeline unchanged, with the GNNs already trained on FAFB and on MCNS; nothing is trained on BANC.

## What was looked at before this protocol

Column contents and label coverage of the BANC metadata (per region and side), where the labels come from, and graph statistics (numbers of neurons, connections and synapses). No model was run on BANC.

## Data

- BANC v888 from the public bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/` (CC BY 4.0): `banc_888_meta.feather` (MD5, base64 `jCuTpgjHFj7J2U1o4HVv9A==`) and `banc_888_edgelist_simple_v2.feather` (`OUQG+Km98JPIlfla/09sSQ==`), the v2 synapse set used throughout the BANC paper.
- Neurons (`flybrain/banc.py`): proofread or roughly proofread; not glia, trachea or non-neuronal; in the central brain or optic lobe, plus ascending and descending neurons. 133,549 neurons (FAFB: 139,255).
- Connections between these neurons. The BANC brain graph is sparse: at FAFB's threshold of 5 synapses, 8.6 connections and 100 synapses per neuron (FAFB: 26.8 and 364), with a similar number of synapses per connection (11.6 vs. 13.6).
- Alignment variants, each matching one statistic with FAFB by a fixed rule: `raw` (threshold 5, nothing aligned), `density` (the threshold whose connections per neuron are closest to FAFB's: >= 2, 29.7 per neuron), `scaled` (threshold 5, counts × 3.63 so that synapses per neuron match FAFB's).

## Labels (used only for evaluation)

- **Super classes:** BANC `super_class` mapped to the 10 FAFB classes (`SUPER_CLASS_MAP`); 122,634 neurons labeled.
- **Cell types, reference A (BANC typing):** `cell_type` where it names one of the 434 FAFB types; 63,332 neurons, 409 types. The optic lobe is typed less completely in the left hemisphere (12,897 vs. 32,106 labeled neurons). These labels are the output of BANC's alignment to FAFB, which uses morphology and connectivity: `cell_type` equals `fafb_alignment_cell_type` for 100 % of the labeled optic-lobe and 98 % of the labeled central-brain neurons. Agreement with reference A therefore measures agreement with another connectivity-informed transfer from FAFB, not with an independent ground truth.
- **Cell types, reference B (morphology match, the cleanest test):** central brain only, `fafb_cell_type` (the type of the FAFB neuron matched by NBLAST, with review) where it names one of the 434 types; 11,656 neurons, 159 types. In the optic lobe this column holds automatic placeholders and is not used.

## Known violations of the pipeline's assumptions (not corrected)

- BANC contains no R1-6 photoreceptors (8,456 neurons, 6 % of FAFB); R7 and R8 are present. Prior matching assumes similar type sizes and will try to fill R1-6.
- Both antennal nerves were damaged: Johnston's organ neurons and nearby central-brain neurons are under-represented.
- Proofreading is incomplete; many optic-lobe types are only partly typed (e.g. Tm5f 9 %, Tm25 7 % of their FAFB size).

## Runs

1. **Primary: FAFB → BANC.** GNNs from `results/transfer2` (3 seeds per label):
   `python experiments/b_gnn_celltypes/label_free.py --target banc`
2. **Secondary: MCNS → BANC.** GNNs from `results/transfer_rev_{density,raw,janelia}`:
   `python experiments/b_gnn_celltypes/label_free.py --source mcns --target banc`

Candidates: every source variant × BANC variant × normalization (target, source) × rho (0, 0.25, 0.5, 0.75, 1); selection by type consistency on the BANC `raw` graph; NTAC refinement seeded where all three GNNs agree. Same settings as in the FAFB ↔ MCNS runs.

## Reported, whatever the outcome

For each run and label:
- accuracy / macro-F1 without prior matching (chosen variant and normalization, rho 0), after prior matching (the selected candidate), after NTAC, and of the best candidate (post hoc);
- the chosen candidate, and the Spearman correlation between type consistency and accuracy over all candidates;
- for cell types: everything against reference A, split into central brain and optic lobe, and against reference B.

Context, not targets: within FAFB 95.2 % (cell types) / 97.8 % (super classes); FAFB → MCNS 92.6 % / 96.2 %; MCNS → FAFB 83.4 % / 95.5 %.

## Expectations stated in advance

1. The selected candidate is within 2 points of the best candidate (FAFB → MCNS: 0.2, MCNS → FAFB: 1.6).
2. The pipeline (+ NTAC) beats every candidate without prior matching on cell types (reference A). This may fail because of the missing R1-6 photoreceptors.
3. Cell-type accuracy is below FAFB → MCNS, because the BANC brain graph is about three times sparser.

## Saved models (SHA-256, first 16 hex digits)

```
transfer2/cell_type/gnn/s0 c66a0ab3659f4f3f   s1 cdd64585dd642d1d   s2 b2c04fd417de5346
transfer2/super_class/gnn/s0 fab86e4fdb64779d   s1 78c6ef52ffb85c70   s2 d35187514dc753e6
transfer_rev_density/cell_type/gnn/s0 2db42a0680a4a5fe   s1 d14d459d4526f911   s2 c36e0c095bab8abb
transfer_rev_density/super_class/gnn/s0 39d655e1b2b52359   s1 8b805d9a73883566   s2 86e84efd348ecac2
transfer_rev_janelia/cell_type/gnn/s0 2d464e7285ac9549   s1 6027ce99f5756672   s2 31ea2687e707a760
transfer_rev_janelia/super_class/gnn/s0 45be8296cbd3637b   s1 778e099647271e23   s2 839dadb277b842a6
transfer_rev_raw/cell_type/gnn/s0 6986b3d429e53dd7   s1 c732b5c044185f35   s2 8bcc0e7305697659
transfer_rev_raw/super_class/gnn/s0 1a49cd08aa312069   s1 62d5ed576a39dea8   s2 880ea3a26b953284
```
