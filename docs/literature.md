# Literature review: cell-type prediction from connectome wiring

Status: October 2026. Searched with alphaXiv (arXiv full text), bioRxiv, Semantic-Scholar-indexed pages and web search. Items marked *(unverified)* were seen only in secondary sources; check them before citing.

## 1. Bottom line

| Claim we could make | Status | Consequence |
|---|---|---|
| "Cell types can be recovered from connectivity alone" | **Taken.** NTAC (Schwartzman et al., 2026) does it on FlyWire with > 90 % (full brain) and > 95 % (visual system) accuracy; the optic-lobe types were *defined* this way (Matsliah et al., 2024). | Not a contribution. Cite as established. |
| "Separate aggregation over inputs and outputs helps" | **Taken.** This is Dir-GNN / Dir-SAGE (Rossi et al., 2023), shown to help on heterophilic graphs. | Our `DirectedSAGE` is Dir-SAGE with synapse-weighted means. Cite it; do not present the architecture as new. |
| "A GNN on anonymous structure, without any partner labels, matches or beats partner-label methods on FlyWire cell types" | **Partly.** On the random split NTAC is better on cell types (97.0 vs. 95.3 %, equal macro-F1), the GNN on super classes (97.9 vs. 94.4 %). NeuNet (Liao et al., 2024) reaches only 59 % on hemibrain from connectivity alone. | Claim "comes close without partner labels", not "beats". |
| "Label-dependent methods fail on an unlabeled hemisphere, the anonymous model does not" | **Tested, holds.** NTAC drops to 8–9 % on cell types (at most 9.4 % at any of 40 steps) and to 78–82 % on super classes (91 % at its best step); the GNN keeps 87–98 %. | Core claim, now against the published method. |
| "Transfer of a cell-type model between connectomes (FAFB → MCNS)" | **Done, as far as we found the first.** Trained on FAFB, the GNN identifies 65–75 % of MCNS brain neurons over 394 types without MCNS labels (86 % with a post-hoc alignment; 96.6 % when trained within MCNS); with the label-free pipeline below 92.6 % (MCNS → FAFB: 83.4 % vs. 95.2 % within FAFB). Partner-label methods cannot be applied. Prior cross-dataset typing uses matching (NBLAST, graph alignment, connectivity similarity). On the sparser, independently reconstructed BANC (pre-registered test), the best candidates reach 58 % (55.7 % chosen label-free, 61.7 % with NTAC). On the partial hemibrain volume (pre-registered) only 19.6 %. | Strongest contribution, but limited by reconstruction completeness of the target. |
| "The quality of a cell-type assignment of an unlabeled connectome can be estimated without labels" | **Not found in connectomics.** Our type-consistency score (chance-corrected self-agreement of partner-type profiles, after the connectomic type definition) ranks transfer labelings like their accuracy (Spearman 0.95; 0.87 in the reverse direction). On BANC it failed in three of four pre-registered runs (0.61–0.86; once it chose a collapsed labeling). Related ideas: clustering validity indices; NTAC's equitable-partition objective; agreement-based accuracy estimation from unlabeled data. | New, but not yet reliable: needs a penalty for collapsed labelings and a test on a further connectome. |
| "Stereotyped type sizes correct a transferred classifier" | **Not found in connectomics.** Prior-shift correction / optimal transport with class marginals is standard in domain adaptation, and reference mapping in single-cell genomics faces the same batch-effect problem. | New in this domain: 87.6 → 91.2 % on MCNS cell types without labels (on top of the label-free choice of preprocessing, which alone lifts 73–75 % to 87.6 %); MCNS → FAFB: 72.0 → 81.3 %; on BANC +13.0 and +17.5 points for the best candidates, despite missing R1-6 photoreceptors. |
| "Errors concentrate where reconstruction is less complete" | **Partly supported by the literature:** outgoing photoreceptor synapses are markedly underdetected in FlyWire (Matsliah et al., 2024). | Explains our R1-6 failures; cite it. The left/right difference in connection counts is our own observation. |

## 2. Annotated bibliography

### A. Direct competitors: neuron typing from connectomes with ML

- **Schwartzman, Jourdan, García-Soriano & Matsliah (2026). NTAC: Neuronal Type Assignment from Connectivity. *Nature Communications* 17, 1284. doi:10.1038/s41467-025-68044-1** (bioRxiv 10.1101/2025.06.11.659184).
  - Method: each neuron is a 2k-dim vector of synapse counts to and from each type (inputs and outputs separate); a non-seed neuron is assigned to the nearest seed by Jaccard distance, then partner labels are replaced by the predictions and the step repeats (about 15 iterations). An unseeded variant searches for an approximately equitable partition.
  - Data: FlyWire female brain (central brain, optic lobes), Janelia male optic lobe, MANC.
  - Results (seeded): > 95 % on visual systems with 2 % of neurons labeled; > 90 % on the full brain; 87 % on the central brain with 80 % labeled. Unseeded: about 70 % (visual systems), 52 % (full brain).
  - Baselines: only NBLAST + k-NN (morphology). No GNN, no label propagation, no cross-hemisphere or cross-dataset test.
  - Relation to us: an iterative version of our partner-profile baseline, and the strongest published method on our data. Available as `pip install ntac`.
- **Liao, Wan & Du (2024). Joint learning neuronal skeleton and brain circuit topology with permutation invariant encoders for neuron classification (NeuNet). AAAI 2024. arXiv:2312.14518.**
  - Hemibrain (21,400 neurons, 191 coarsened classes) and human H01; random 8:1:1 split; skeleton point-cloud encoder + GCNII on the connectome.
  - Full model 91.7 % on hemibrain; connectome encoder alone 58.7 %, skeleton alone 86.1 %.
  - Relation: the only GNN-on-fly-connectome typing result we found, with much weaker connectivity-only performance than ours (different dataset and labels, and the node features of the connectome encoder are not described).
- **Lu, Han, Wang, Mi & Yang (2026). A benchmark analysis of graph and non-graph methods for *C. elegans* neuron classification. arXiv:2603.02241.**
  - 300 neurons, 3 classes (sensory / inter / motor), functional connectome of Randi et al. (2023); GCN, GraphSAGE, GAT, GraphTransformer vs. logistic regression, MLP, LOLCAT, NeuPRINT; 5-fold CV.
  - Best about 62 % accuracy; GNNs beat non-graph baselines moderately.
  - Relation: same question at a tiny scale; shows GNN-vs-baseline framing is accepted.
- **Troidl et al. (2024). Global neuron shape reasoning with point affinity transformers. bioRxiv 10.1101/2024.11.24.625067.** Morphology-based typing (contrastive embedding + k-NN); NTAC cites 81 % on optic-lobe families and 77 % on hemibrain *(numbers unverified)*. Relation: morphology, not wiring.

### B. How the labels were made (circularity)

- **Schlegel et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature*.** Hierarchical annotation (super class, class, cell type, hemilineage) of FlyWire; 8,453 types, of which 3,643 match hemibrain types. Defines a cell type as cells that are each more similar to cells in another brain than to any other cell in the same brain, validated jointly with the hemibrain. Relation: our labels; the cross-brain definition makes hemisphere-level consistency partly built into the labels.
- **Matsliah et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634, 166–180.**
  - Typed about 38,500 intrinsic neurons of the **right** optic lobe (227 intrinsic types); the left lobe was typed afterwards and "the results match the right optic lobe".
  - Type definition: each cell's input and output synapse vectors over types, compared with weighted Jaccard distance, refined iteratively from human-seeded types. This is exactly the information of our partner-profile baseline.
  - "Outgoing photoreceptor synapses are markedly underdetected."
  - Relation: central for the circularity discussion and for explaining our photoreceptor errors.
- **Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*.** The FlyWire connectome itself (v783 used here).
- **Scheffer et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife*.** The hemibrain.

### C. Methods we build on

- **Rossi, Charpentier, Di Giovanni, Frasca, Günnemann & Bronstein (2023). Edge directionality improves learning on heterophilic graphs. LoG 2023, PMLR 231. arXiv:2305.10498.**
  - Dir-GNN: separate aggregation over in- and out-neighbors with separate weights (Dir-GCN, Dir-SAGE, Dir-GAT); as expressive as directed WL.
  - Directed operators raise "effective homophily" on heterophilic graphs; gains of 10–20 points there, none on homophilic graphs; using only one direction leaves nodes with zero in- or out-degree without information.
  - Relation: our model is Dir-SAGE with synapse-weighted means; our finding (large gain on heterophilic cell types, small on homophilic super classes) reproduces theirs on a new domain.
- **Hamilton, Ying & Leskovec (2017). Inductive representation learning on large graphs (GraphSAGE). NeurIPS.** Our plain baseline.
- **Zhou, Bousquet, Lal, Weston & Schölkopf (2004). Learning with local and global consistency. NeurIPS.** Label spreading, our label-propagation baseline.
- **Zhu, Ghahramani & Lafferty (2003). Semi-supervised learning using Gaussian fields and harmonic functions. ICML.** Class mass normalization.
- **Zhu, Yan, Zhao, Heimann, Akoglu & Koutra (2020). Beyond homophily in graph neural networks: current limitations and effective designs (H2GCN). NeurIPS.** Standard reference for heterophily and the homophily measure.
- **Ribeiro, Saverese & Figueiredo (2017). struc2vec: learning node representations from structural identity. KDD. arXiv:1704.03165.** Structural (role) identity vs. proximity; conceptual background for "anonymous wiring identifies a neuron".

### D. Matching across hemispheres and datasets

- **Pedigo, Winding, Priebe & Vogelstein (2023). Bisected graph matching improves automated pairing of bilaterally homologous neurons from connectomes. *Network Neuroscience*.** Graph matching of left/right homologs in the larval connectome, using contralateral edges. Relation: hemisphere correspondence by matching, not by a trained classifier.
- **Priebe et al. (2017). Semiparametric spectral modeling of the *Drosophila* connectome.** Adjacency spectral embedding + Gaussian mixtures on the larval mushroom body recover biologically meaningful groups. Authors: Priebe, Park, Tang, Athreya, Lyzinski, Vogelstein, Qin, Cocanougher, Eichler, Zlatic & Cardona; arXiv:1705.03297 (journal version not confirmed). Relation: unsupervised structure-based typing.
- **BANC brain-and-cord connectome (2026). Distributed control circuits across a brain-and-cord connectome. *Nature*.** Cell types matched to other datasets with NBLAST, connectivity-based graph alignment and a combined annealing method *(author list unverified)*. Relation: current practice for cross-dataset typing is matching.
- **Berg et al. (2026). Sexual dimorphism in the complete connectome of the *Drosophila* male central nervous system. *Cell* 189(18), 5504–5526, doi:10.1016/j.cell.2026.08.015 (bioRxiv 10.1101/2025.10.09.680999).** 166,700 neurons, brain and ventral nerve cord, one male fly; type mapping to FlyWire and MANC published on male-cns.janelia.org, with connection weights scaled by 0.581 to align with FlyWire *(volume, pages and date from secondary sources; check the journal page)*. Relation: our transfer target; the type mapping removes most label-alignment work.

### E. Network statistics and modeling context

- **Lin et al. (2024). Network statistics of the whole-brain connectome of *Drosophila*. *Nature* 634, 153–165.** Rich club (about 30 % of neurons), motifs, region subnetworks. Relation: context for degree features and heterogeneity.
- **Lappalainen et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature*.** Part c of this project.

## 3. What this changes in our project

1. **NTAC baseline: done** (`flybrain/ntac.py`, `baselines.py --method ntac`). The package needs two dense ~40 GB matrices on FlyWire, so it was reimplemented row by row; with one thread both agree exactly at every step. With several threads the package's similarity kernel writes to shared cells and its results vary from run to run, worth noting when citing it.
2. **Rename and cite the architecture.** Describe `DirectedSAGE` as Dir-SAGE (Rossi et al., 2023) with synapse-weighted mean aggregation. Our contribution is the application and analysis, not the layer.
3. **Frame the contribution as label-free identification and transfer:** the GNN needs no labeled partners in the target graph, which is what the hemisphere split and the MCNS transfer test.
4. **Cite the photoreceptor synapse underdetection** (Matsliah et al., 2024) when discussing the right → left gap.
5. **State the circularity precisely:** optic-lobe types were defined from input/output type vectors, i.e. the partner-profile information; types are cross-brain consistent by definition.

## 4. Related work (draft)

> **Cell typing from connectomes.** Whole-brain connectomes of *Drosophila* (Scheffer et al., 2020; Dorkenwald et al., 2024) have been annotated with thousands of cell types (Schlegel et al., 2024; Matsliah et al., 2024). Types were assigned from morphology and connectivity, and in the optic lobe they were defined directly from each neuron's synaptic input and output vectors over types (Matsliah et al., 2024). NTAC (Schwartzman et al., 2026) automates this: starting from a few labeled seeds, it repeatedly assigns each neuron to the seed with the most similar partner-type connectivity and reaches over 90 % accuracy on the FlyWire brain. Learned approaches have mostly used morphology (Troidl et al., 2024) or combined it with the connectome (Liao et al., 2024), where connectivity alone reached 59 % on hemibrain classes; on *C. elegans*, GNNs moderately outperform non-graph baselines for three coarse classes (Lu et al., 2026). All of these methods either read the labels of a neuron's partners or are evaluated within one labeled graph.
>
> **Directed and heterophilic graph learning.** Neural circuits are directed, and neurons mostly connect to neurons of other types, a heterophilic setting in which standard message passing struggles (Zhu et al., 2020). Aggregating separately over in- and out-neighbors (Dir-GNN; Rossi et al., 2023) raises the effective homophily of such graphs and improves node classification; our model is a synapse-weighted instance of Dir-SAGE (Hamilton et al., 2017; Rossi et al., 2023). Structural-identity embeddings (Ribeiro et al., 2017) motivate the question of how much of a node's role is encoded in anonymous structure alone.
>
> **Matching across hemispheres and individuals.** Homologous neurons have been paired across hemispheres by graph matching (Pedigo et al., 2023) and cell types across datasets by morphological and connectivity similarity (Schlegel et al., 2024; BANC, 2026). We instead ask whether a model trained on one labeled hemisphere identifies neurons on an unlabeled one, and, as a next step, on a different animal.

## 5. Open checks before citing

- MCNS paper (Berg et al., 2026): volume, pages and date against the *Cell* page.
- BANC paper: author list.
- Troidl et al.: accuracy numbers.
- Priebe et al. (2017): whether a journal version exists (arXiv:1705.03297 confirmed).
- NTAC: exact per-dataset numbers (in figures and supplement, not in the text we read).
