# flybrain

Machine learning on the *Drosophila* connectome: how much of a neuron's identity is encoded in its wiring alone, and what does connectome-constrained modelling of the fly visual system tell us about the role of wiring?

> Status: scaffold. Results sections are placeholders.

## Motivation

Whole-brain connectomes (FlyWire/FAFB, Male CNS) list every neuron and synapse of an adult fly brain. Cell types are defined largely by morphology and connectivity, which raises two questions: can a model recover them from connectivity alone, and does the wiring diagram by itself constrain function?

## Research Questions

- **Part b.** Can a GNN predict super class and cell type of a neuron from pure connectivity on FlyWire (FAFB)? Does a model trained on FAFB transfer to the Male CNS (MCNS) connectome?
- **Part c.** Can the connectome-constrained visual-system network of Lappalainen et al. (2024, "flyvis") be reproduced, and how do its predictions change under ablations: real vs. degree-preserving shuffled vs. random wiring?

## Data

FlyWire (FAFB) and MCNS data are released under CC-BY. Raw files are **not** in this repository and must be placed manually in `data/raw/`; see [data/README.md](data/README.md) for expected files and licenses. Check them with `python data/download_fafb.py`.

## Methods

- **Graph:** neurons are nodes; directed edges pre → post, weighted by synapse count. Raw tables are cleaned once and cached as Parquet.
- **Baseline:** logistic regression on degree / synapse-count features (`experiments/b_gnn_celltypes/baseline_logreg.py`).
- **GNN:** GraphSAGE, configured via YAML (`experiments/b_gnn_celltypes/train_gnn.py`).
- **Splits:** seeded, stratified train/val/test over labeled neurons. All runs log config and metrics to `results/<run_name>/`.
- **Part c:** plan only, see [experiments/c_constrained_rnn](experiments/c_constrained_rnn/README.md).

## Quickstart

```bash
uv sync
uv run pytest
uv run python experiments/b_gnn_celltypes/baseline_logreg.py --synthetic
uv run python experiments/b_gnn_celltypes/train_gnn.py --synthetic
```

CPU works by default; CUDA is used automatically when available.

## Results

| Model | Label | Accuracy | Macro-F1 |
|---|---|---|---|
| Logistic regression (degree features) | super_class | TBD | TBD |
| GraphSAGE | super_class | TBD | TBD |
| GraphSAGE | cell_type | TBD | TBD |
| GraphSAGE, FAFB → MCNS transfer | super_class | TBD | TBD |

## Limitations

- Degree-only baselines can be strong on coarse labels; super-class results must be read against them.
- Cell-type labels are highly imbalanced and partly derived from connectivity, so there is a risk of label leakage.
- FAFB and MCNS differ in proofreading state, synapse detection and annotation schemes; transfer results depend on how neurons and classes are aligned.
- The synthetic graph is for testing only and says nothing about biology.

## Citation

```bibtex
@misc{vogt2026flybrain,
  author = {Vogt, Maximilian},
  title  = {flybrain: ML on the Drosophila connectome},
  year   = {2026},
  url    = {https://github.com/paradoxsmile/flybrain}
}
```

Please also cite the data sources (FlyWire / FAFB, MCNS) and Lappalainen et al. (2024), *Connectome-constrained networks predict neural activity across the fly visual system*, Nature.

## License

Code: MIT (see [LICENSE](LICENSE)). Data: see the licenses of the respective sources (CC-BY).
