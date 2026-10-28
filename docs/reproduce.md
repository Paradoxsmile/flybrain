# Reproducing the numbers

Every table and figure of the paper ([paper/draft.md](../paper/draft.md), v0.2), with the commands that produce it and the file the numbers are read from.

## Setup

- `uv sync`, then run every command from the repository root as `uv run python ...`.
- Put the raw data in `data/raw/` as described in [data/README.md](../data/README.md); `python data/download_fafb.py` checks the FlyWire files.
- `S` below stands for `experiments/b_gnn_celltypes`.
- **Hardware.** GNN training for the benchmark, the transfer and the type-size sweep ran on a Kaggle T4 GPU; everything else ran on a CPU. CUDA is used automatically when available.
- **What is committed.** The summaries of the reported runs are in the repository: `results/*/summary.md` or `SUMMARY.md`, every `metrics.csv`, `results/vnc_rare/report.json` and `results/selection_rules_*.md`. Models, predictions and matchings (about 1.6 GB) are not committed; the commands below regenerate them.
- **Order.** Later steps read earlier outputs: label-free runs need the saved GNNs, graph matching needs a stage-1 run, `stratify.py` and `label_effort.py` need the label-free runs.
- **Protocols.** The four pre-registered tests list their own commands, in the order they were run: [banc_protocol.md](banc_protocol.md), [hemibrain_protocol.md](hemibrain_protocol.md), [banc_rare_protocol.md](banc_rare_protocol.md) and [vnc_rare_protocol.md](vnc_rare_protocol.md).

## Section 4.1: within a connectome

| Paper | Commands | Numbers in |
|---|---|---|
| Table 2, Figure 1 | `S/benchmark.py`; then `S/plot_benchmark.py` for the figure. Baselines at commit `2000b7b`, NTAC at `bfbd103`, GNN rows at `343125f`. The run can be split with `--labels` / `--methods`. | `results/benchmark/summary.md`; `docs/benchmark_{light,dark}.png` |
| MCNS replication | `S/benchmark.py --dataset mcns` (at `4ec2a8a`) | `results/benchmark_mcns/summary.md` |
| GNN-seeded NTAC | `S/hybrid.py` (needs the benchmark and transfer runs) | `results/hybrid/summary.md` |
| Ablations (README) | `S/ablation.py --label-col cell_type --min-class-size 20` and `--label-col super_class`, each also with `--split left_right` / `right_left` | `results/ablation_*/summary.csv` (not committed) |

## Section 4.2: label-free transfer

| Paper | Commands | Numbers in |
|---|---|---|
| Single transferred GNN (73–75 %, best alignment 85.7 %) | `S/transfer.py` (at `4ec2a8a`) | `results/transfer/summary.md` |
| GNNs saved for the label-free pipeline | `S/transfer.py --methods gnn --out transfer2` | `results/transfer2/summary.md` |
| FAFB → MCNS, 92.6 % | `S/label_free.py` (models `transfer2`, output `label_free`) | `results/label_free/summary.md` |
| Reverse, MCNS → FAFB, 83.4 % | for each `v` in `density raw janelia`: `S/transfer.py --source mcns --source-variant v --methods gnn --out transfer_rev_v`; then `S/label_free.py --source mcns --target fafb` | `results/transfer_rev_*/summary.md`, `results/label_free_rev/summary.md` |
| Figure 2 | `paper/make_figures.py` | `paper/figures/transfer.png` |
| Region and type-size breakdown, Figure 3 | `S/stratify.py` | `results/stratified/summary.md`; `paper/figures/type_f1.png` |
| Table 3 (minimum type size) | `S/size_threshold.py --min-sizes 20 10 5 3 2`. The `within` stage trains the GNNs (GPU; lean training below 5), `label_free` runs the transfer (CPU; lean below 5) and `report` writes the tables. Stages can be run separately with `--stages`. | `results/size_threshold/SUMMARY.md` |

## Section 4.3: rare types, within FlyWire and FAFB → MCNS

| Paper | Commands | Numbers in |
|---|---|---|
| Table 4 | `S/graph_matching.py` (oracle candidates, both edge sets); `S/graph_matching.py --candidates gnn` (label-free two-stage). Needs the `size_threshold` m2 left → right runs. | `results/graph_matching/SUMMARY.md`, `metrics.csv` |
| Table 5 | `S/graph_matching_mcns.py --stage weights`; then `--stage match` (barycenter start, `gnn` candidates), `--stage match --start soft` and `--stage match --candidates oracle`. Needs `size_threshold/m2/label_free`. | `results/graph_matching_mcns/SUMMARY.md`, `metrics.csv` |

## Section 4.4: pre-registered rare-type tests

| Paper | Commands | Numbers in |
|---|---|---|
| Table 6 (BANC brain) | `S/size_threshold.py --min-sizes 2 --stages label_free --target banc --select size_agreement`; `S/graph_matching_mcns.py --target banc --stage match`, the same with `--candidates oracle`, and with `--variant raw` / `--variant scaled` | `results/graph_matching_banc/SUMMARY.md`, `metrics.csv` |
| Table 7 (candidate selection, development) | Run table R1–R14 in [candidate_selection_plan.md](candidate_selection_plan.md). The steps: `S/candidate_selection.py --stage mass`, `--stage select`, `--stage twotier`. The 25 % sandbox stage 1: `S/size_threshold.py --min-sizes 2 --stages label_free --select size_agreement --target-retention 0.25` (and `--min-sizes 10` for the fallback; `--retention-seed 3` for the second realization). The matching: `S/graph_matching_mcns.py --stage match --stage1 label_free_mcns_r0.25_size_agreement --out graph_matching_mcns_r0.25` with `--candidates gnn`, `union`, `file:mass_fill` (written by `--stage select`) and `oracle`; the second realization with `--stage1 label_free_mcns_r0.25_seed3_size_agreement --out graph_matching_mcns_r0.25_seed3`. Full MCNS `union`: `--candidates union --out graph_matching_mcns_cand`. Many-to-one: `--candidates oracle --capacity 2 --out graph_matching_mcns_cap`. | `results/candidate_selection/SUMMARY.md`; `results/graph_matching_mcns_{r0.25,r0.25_seed3,cand,cap}/metrics.csv` |
| Morphology baseline (exploratory, [finalize_plan.md](finalize_plan.md) A1–A2) | The five commands in `results/morphology_baseline/SUMMARY.md` ("Commands"): `S/morphology_baseline.py --candidates-out`, three `S/graph_matching_mcns.py` runs into `morphology_baseline/gm_banc*`, then `S/morphology_baseline.py` | `results/morphology_baseline/SUMMARY.md`, `metrics.csv`, `gm_banc*/metrics.csv` |
| Table 8 (nerve cords) | For `t` in `banc_cord mcns_cord`: `S/size_threshold.py --source manc --min-sizes 2 --stages within` (MANC GNNs; also `--min-sizes 10`), then the seven runs of [vnc_rare_protocol.md](vnc_rare_protocol.md) ("Runs"); `S/vnc_report.py` collects them | `results/vnc_rare/SUMMARY.md`, `report.json`; `results/graph_matching_{banc,mcns}_cord*/metrics.csv` |

## Section 4.5: where label-free transfer fails

| Paper | Commands | Numbers in |
|---|---|---|
| Table 9 (BANC, common types) | `S/label_free.py --target banc`; `S/label_free.py --source mcns --target banc` | `results/label_free_fafb_banc/summary.md`, `results/label_free_mcns_banc/summary.md` |
| Figure 4, density | `S/thinning.py --labels cell_type super_class`; augmented: `S/thinning.py --labels cell_type super_class --retentions 1 --target-retentions 1 0.6 0.4 0.25 --thin-aug 0.25 --out thinning_aug`; figure: `paper/make_figures.py` | `results/thinning/summary.md`, `results/thinning_aug/summary.md`; `paper/figures/thinning.png` |
| On BANC with thinned GNNs (exploratory) | `S/label_free.py --target banc --source-retention 0.4`; `S/label_free.py --target banc --models thinning_aug/models --out label_free_fafb_banc_aug` | `results/label_free_fafb_banc_r1_s0.4/summary.md`, `results/label_free_fafb_banc_aug/summary.md` |
| Table 10, sandbox column | `S/label_free.py --target mcns --out label_free_fafb_mcns_r1`; `--target mcns --target-retention 0.5`; `--target-retention 0.2`; `--target-retention 0.5 --target-exclude R1-6`, the same with `--max-weight 10`; `S/label_free.py --source mcns --target fafb --out label_free_mcns_fafb_r1`; the same with `--target-retention 0.35`. Then `S/selection_rules.py` on these seven runs with `--out selection_rules_sandbox.md` | `results/selection_rules_sandbox.md` |
| Table 10, BANC column | `S/label_free.py --target banc --out label_free_fafb_banc_cols`; `S/label_free.py --source mcns --target banc --labels cell_type --out label_free_mcns_banc_cols`, the same with `--labels super_class --out label_free_mcns_banc_cols_super`; then `S/selection_rules.py label_free_fafb_banc_cols label_free_mcns_banc_cols_super label_free_mcns_banc_cols label_free_fafb_banc_r1_s0.4 --out selection_rules_banc.md` | `results/selection_rules_banc.md` |
| Partial-volume sandbox | `S/label_free.py --target mcns --target-coverage central_brain --target-crop 0.16`, the same with `--prior em`, and with `--coverage central_brain --prior coverage --out label_free_partial_covnorm`; without the crop: `--target-coverage central_brain --coverage central_brain --prior coverage --out label_free_partial_covnorm_nocrop` | `results/label_free_fafb_mcns_r1_central_brain_crop0.16*/summary.md`, `results/label_free_partial_covnorm*/summary.md` |
| Table 11 (hemibrain) | `S/label_free.py --target hemibrain --coverage central_brain --prior coverage --select size_agreement --out label_free_hemibrain`; FlyWire prior: the same with `--prior source --out label_free_hemibrain_source_prior` | `results/label_free_hemibrain/summary.md`, `results/label_free_hemibrain_source_prior/summary.md` |

## Section 4.6: annotation team

| Paper | Commands | Numbers in |
|---|---|---|
| Table 13, Figure 5 | `S/label_effort.py` (needs `label_free` and `label_free_fafb_banc`); the figure is written next to the results and copied to `paper/figures/` | `results/label_effort/SUMMARY.md`, `metrics.csv`; `paper/figures/label_effort.png` |
| Accuracy estimation, error flagging | `S/estimate_accuracy.py` (needs the label-free runs above) | `results/accuracy_estimation/SUMMARY.md` |

## Notes

- Commands for runs before the protocols were reconstructed from the scripts' output names and the session logs; they match the folder names under `results/`. The protocols give the commands of the pre-registered runs verbatim.
- Two rare-type options are kept in the code because their negative results are reported: many-to-one matching (`--capacity`) and the two-tier fallback (`--fallback`). Neither is part of the recommended pipeline (`--candidates union`, barycenter start, no fallback).
