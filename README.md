# Beyond Average Dice

### Case-Level Failure and Recoverability under Language Conditioning in Medical Image Segmentation

This repository contains the code, experimental outputs, prompt configurations, and reproducibility records for our study of **case-level reliability under language conditioning in medical image segmentation**.

## Overview

Language-guided medical image segmentation systems are commonly evaluated using aggregate metrics such as mean Dice. However, aggregate performance can conceal substantial variation at the individual-case level when the language description changes.

This work asks:

> **What does aggregate prompt robustness conceal about individual-case failure and recoverability in language-guided medical segmentation?**

We investigate this question across three complementary experimental regimes.

### 1. Broad language variation — MedCLIP-SAM

We evaluate 20 semantically related descriptions per case in two settings:

- **Breast:** 97 valid test cases
- **Brain:** 400 slices from 174 patients

We analyze:

- case-level segmentation sensitivity;
- within-case performance ranges;
- heavy-tailed and catastrophic failures;
- retrospective recoverability across descriptions;
- language-conditioned localization changes;
- residual failures when localization remains similar.

### 2. Controlled lexical variation at a strong operating point — MedCLIP-SAMv2

We additionally evaluate a separate released Brain cohort of **600 images** using a parent-compatible high-performance configuration.

Three language conditions are compared:

- **H0:** original image-specific description;
- **H1:** observation-verb substitution;
- **H2:** normalized imaging prefix plus verb substitution.

These perturbations preserve the clinically informative lesion description while introducing small lexical changes.

### 3. Cross-system/task boundary condition — Text3DSAM

We evaluate Text3DSAM on **30 AMOS abdominal CT liver cases** using four semantically equivalent liver descriptions.

This experiment provides a boundary condition for the analysis: large language sensitivity is not observed uniformly across all evaluated settings.

---

## Main Findings

### Broad 20-description evaluation

| Setting | Mean pairwise \|ΔDice\| | Mean case range | Best fixed Dice | Retrospective oracle | Oracle advantage |
|---|---:|---:|---:|---:|---:|
| Breast (N=97) | 0.1047 | 0.3406 | 0.4997 | 0.6040 | +0.1043 |
| Brain (N=400) | 0.1117 | 0.3859 | 0.4760 | 0.5641 | +0.0881 |

The broad-description experiments reveal substantial case-level variation that is not apparent from aggregate performance alone.

The **retrospective oracle** selects the highest-Dice description independently for each case using ground truth. It is used only as a diagnostic measure of recoverability and **is not a deployable prompt-selection method**.

### Strong Brain600 operating point

For the original H0 condition:

- **Mean Dice:** 0.7538
- **Median Dice:** 0.8195

Across H0/H1/H2:

- **Mean case-level range:** 0.0346
- **Range > 0.10:** 8.33%
- **Range > 0.20:** 4.33%
- **Range > 0.50:** 1.33%
- **Retrospective oracle advantage:** +0.0133
- **Failure-success switches:** 7/600 (1.17%)

Thus, small information-preserving lexical changes are usually stable in this stronger operating regime, while a small severe failure tail remains.

### Text3DSAM contrast

On AMOS liver segmentation:

- **Mean Dice:** ≈ 0.9144
- **Mean pairwise |ΔDice|:** ≈ 0.00021
- **Mean within-case range:** ≈ 0.00039
- **Oracle advantage:** ≈ +0.00009

The evaluated Text3DSAM setting is therefore nearly invariant to the four tested descriptions.

This is a **cross-system/task contrast**, not a controlled architectural comparison.

---

## Repository Structure

```text
beyond-average-dice/
├── scripts/
│   ├── breast_p20/
│   ├── brain_p20/
│   ├── brain600_highop/
│   └── text3dsam/
│
├── results/
│   ├── breast_p20/
│   ├── brain_p20/
│   ├── brain600_highop/
│   └── text3dsam/
│
├── prompts/
├── docs/
├── figures/
└── README.md
```

### `scripts/`

Experiment and analysis code for the controlled language-conditioning evaluations.

### `results/`

Frozen numerical outputs used in the final analysis, including case-level metrics, bootstrap summaries, failure-mode analyses, localization analyses, and high-operating-point results.

### `prompts/`

Language conditions used in the Brain600 H0/H1/H2 experiments.

### `docs/`

Reproducibility information including:

- upstream repository states;
- frozen Python environments;
- environment information;
- relevant modifications to upstream source code.

---

## Reproducibility

The repository preserves the main information required to trace the experiments:

- experiment and analysis scripts;
- exact language conditions;
- frozen numerical results;
- bootstrap outputs;
- RNG protocol;
- upstream Git commit information;
- relevant upstream source modifications;
- Python environment snapshots.

Large datasets, pretrained model checkpoints, Hugging Face caches, and large generated intermediate artifacts are intentionally not stored in this GitHub repository.

These dependencies should be obtained from their original sources and remain subject to their respective licenses and access conditions.

---

## Experimental Notes

### Breast P20

The controlled Breast evaluation contains **97 valid cases**. One case from the original 98-case test set was excluded because its native image and mask geometries were incompatible.

Twenty official Breast P2 descriptions were evaluated individually.

### Brain P20

The controlled Brain evaluation contains:

- **400 slices**
- **174 patients**
- **20 descriptions per case**

Uncertainty intervals use patient-cluster bootstrap resampling.

### Brain600

The high-operating-point experiment uses a **separate released MedCLIP-SAMv2 Brain cohort containing 600 images**.

Brain400 and Brain600 have zero filename overlap. Patient-level correspondence between these cohorts is unavailable and is not assumed.

Patient identifiers are unavailable for the released Brain600 cohort, so uncertainty intervals use image-level bootstrap resampling.

### Text3DSAM

The Text3DSAM experiment evaluates **30 AMOS liver CT cases** using four semantically equivalent descriptions.

Because this experiment differs from the MedCLIP-SAM evaluations in model, dataset, anatomy, and language conditions, it is used as a supporting boundary condition rather than a controlled architecture comparison.

---

## Upstream Implementations

This work builds on the official implementations of:

- **MedCLIP-SAM**
- **MedCLIP-SAMv2**
- **Text3DSAM**

Exact upstream Git states and relevant local source modifications are recorded under `docs/`.

The full upstream repositories are not duplicated here.

---

## Scope

This work focuses on **case-level failure and recoverability under language conditioning**.

It does **not** propose a new segmentation architecture.

The experiments also do not imply that language-guided medical segmentation is universally unstable. Instead, the results indicate that language sensitivity is strongly dependent on the evaluated operating regime, model, task, and language variation.

---

## Status

**Experimental phase: frozen.**

The repository is currently being prepared alongside the manuscript and course report. No additional substantive model experiments are planned unless manuscript construction exposes a specific evidentiary gap.

---

## Citation

Citation information will be added upon manuscript release.

---

## License

Repository licensing and redistribution terms are being finalized.

Upstream code, datasets, pretrained models, and checkpoints remain subject to their original licenses and terms.
