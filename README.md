# Beyond Average Dice

Code and derived experimental results for:

**Beyond Average Dice: Case-Level Failure and Recoverability under Language Conditioning in Medical Image Segmentation**

## Status

Research repository under active preparation.

This repository contains scripts, prompts, derived metrics, statistical analyses,
and reproducibility information for experiments studying case-level sensitivity
to language conditioning in medical image segmentation.

## Experiments

- MedCLIP-SAMv2 — Breast P20 controlled language variation
- MedCLIP-SAMv2 — Brain P20 controlled language variation
- MedCLIP-SAMv2 — Brain600 parent-compatible reproduction and controlled lexical perturbations
- Text3DSAM — AMOS liver language-sensitivity experiment

Large datasets and model checkpoints are not included.

## Repository structure

- `scripts/` — experiment and analysis scripts
- `results/` — derived experimental results
- `prompts/` — controlled prompt variants
- `docs/` — environment and upstream repository provenance
- `figures/` — publication figures (to be added)

## Reproducibility

Exact upstream repository commits and the frozen experimental Python
environments are recorded under `docs/`.

Detailed setup, data acquisition, execution commands, statistical definitions,
and figure reproduction instructions will be added during manuscript preparation.
