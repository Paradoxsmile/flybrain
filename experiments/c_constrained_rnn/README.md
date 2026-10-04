# Part c: connectome-constrained network (plan only)

Nothing is implemented here yet.

## Goal

Reproduce the connectome-constrained model of the fly visual system from Lappalainen et al. (2024), "flyvis", then test how much the real wiring matters.

## Plan

1. **Environment.** Use a separate environment (own `pyproject.toml` or venv in this folder). `flyvis` pins its own torch and other dependencies, which may conflict with the main `flybrain` environment.
2. **Reproduction.** Install `flyvis`, train or load the published ensemble, and check that it reproduces key results (e.g. direction-selectivity of T4/T5 on moving-edge stimuli).
3. **Ablations.** Same training protocol and task for each wiring:
   - real connectome,
   - degree-preserving shuffled wiring,
   - random wiring (matched density).
4. **Evaluation.** Task performance, tuning-curve similarity to the reference, and variation across ensemble members.
5. **Logging.** Same convention as part b: config and metrics in `results/<run_name>/`.

## Open questions

- Compute budget and whether pretrained flyvis checkpoints are enough for the baseline.
- How to define the degree-preserving shuffle given cell types and the retinotopic structure.
