# Beyond Average Dice

### Case-Level Reliability, Failure Mechanisms, and Ground-Truth-Free Risk Signals in Language-Guided Medical Image Segmentation

This repository contains the code, experimental outputs, prompt configurations, frozen protocols, figures, and reproducibility records for our study of **case-level reliability under language conditioning in medical image segmentation**.

Language-guided segmentation systems are often summarized using aggregate metrics such as mean Dice. However, similar aggregate performance can conceal substantial variation at the individual-case level when the language condition changes.

This work therefore asks:

> **When language changes while the image and intended target remain fixed, how stable is the prediction, where do severe failures emerge, and can that instability itself provide a useful reliability signal?**

The project progresses from reproduction and case-level sensitivity analysis to failure localization, controlled representation intervention, and ground-truth-free risk assessment.

---

## Study Overview

The study evaluates language-conditioned segmentation across several complementary regimes:

1. **Broad language variation — MedCLIP-SAM**  
   Twenty semantically related descriptions are evaluated on Breast97 and Brain400 to quantify case-level sensitivity, severe failure tails, and retrospective recoverability.

2. **Distinct stronger operating regime — MedCLIP-SAMv2 / Brain600**  
   Six language conditions are evaluated on a separate 600-image Brain cohort to determine whether instability persists when overall segmentation performance is substantially stronger.

3. **Failure-mechanism analysis**  
   Catastrophic failure-success transitions are decomposed through the localization stage and downstream SAM segmentation stage.

4. **Ground-truth-free reliability analysis**  
   Localization disagreement is frozen on a development partition and evaluated on a patient-disjoint Brain400 hold-out set, then transferred to Brain600 without retuning.

5. **Controlled representation intervention**  
   The projected text representation is moved along a predefined direction while the image and model are fixed, testing whether representation movement produces ordered localization movement.

6. **Cross-system/task boundary condition — Text3DSAM**  
   Four semantically equivalent descriptions are evaluated on 30 AMOS liver CT volumes to test whether substantial language sensitivity is universal.

---

## Main Findings

### 1. Aggregate performance can conceal substantial case-level instability

Under broad 20-description evaluation:

| Setting | Mean pairwise \|ΔDice\| | Mean case range | Cases with range > 0.50 | Best fixed Dice | Retrospective oracle | Oracle advantage |
|---|---:|---:|---:|---:|---:|---:|
| Breast97 | 0.1047 | 0.3406 | 35.1% | 0.4997 | 0.6040 | +0.1043 |
| Brain400 | 0.1117 | 0.3859 | 40.5% | 0.4760 | 0.5641 | +0.0881 |

The broad-description experiments reveal substantial case-level variation that is not apparent from aggregate performance alone.

The **retrospective oracle** selects the highest-Dice language condition independently for each case using ground truth. It is used only to quantify **recoverability**: whether a substantially better prediction already exists among the evaluated language conditions.

It is **not a deployable prompt-selection method**.

---

### 2. A stronger operating regime can have a stable center but still retain a severe tail

Brain600 is a separate released 600-image Brain cohort evaluated under a stronger MedCLIP-SAMv2 operating regime.

Across the six evaluated language conditions:

- **Median within-case Dice range:** 0.0368
- **Range > 0.20:** 20.3%
- **Range > 0.50:** 8.0%

This produces an important distinction:

> **Low average sensitivity does not imply the absence of severe individual-case failures.**

The central tendency is relatively stable, while a smaller but important tail remains highly language-sensitive.

---

### 3. Catastrophic segmentation transitions are strongly associated with localization changes

For Brain400, catastrophic failure-success transitions were defined operationally as language-condition pairs in which the same image changes from:

- **Dice < 0.10** under one condition to
- **Dice ≥ 0.50** under another.

Across **5,398 catastrophic transitions**:

- **83.7%** had localization-box IoU < 0.10
- **94.3%** had localization-box IoU < 0.50
- only **25 transitions** retained localization-box IoU ≥ 0.90

This indicates that the dominant catastrophic failure pathway in the evaluated 2D pipeline is associated with a change in **where the model localizes the target**, rather than only with downstream mask refinement.

Localization is not the sole possible failure source. Residual high-overlap cases were separately analyzed to distinguish localization-driven failures from downstream segmentation failures.

---

### 4. Localization disagreement can act as a ground-truth-free risk signal

The reliability analysis uses disagreement between localization outputs across language conditions as an inference-time signal.

The scoring procedure and threshold were frozen on a development partition before evaluation on a **patient-disjoint Brain400 hold-out set**.

On held-out Brain400:

- **AUPRC:** 0.373
- **79.1% of catastrophic slices** were identified by the frozen signal

The purpose is not to estimate Dice directly. Instead, the signal identifies cases whose localization is unstable across language conditions and therefore may warrant additional scrutiny.

Ground truth is used to **evaluate** the signal, but is **not required to compute the disagreement score at inference time**.

---

### 5. The frozen risk signal transfers to Brain600 without retuning

The same frozen reliability rule was applied to Brain600 without changing the threshold.

On Brain600:

- **AUROC:** 0.899
- catastrophic prevalence was **6.3% overall**
- **2.1%** among retained cases
- **32.5%** among flagged cases

Thus, the frozen signal transfers to a distinct and substantially stronger operating regime without retuning.

The result supports **risk enrichment**, not definitive failure classification.

---

### 6. Detecting instability is easier than automatically correcting it

A consistency-based automatic correction strategy was also evaluated during development.

The proposed gated-medoid remedy reduced mean Dice from:

- **0.4754 → 0.4467**

and increased the number of Dice < 0.10 failures.

Because the correction strategy failed the predefined development gate, it was **not advanced to held-out evaluation**.

This negative result is retained deliberately:

> **A useful reliability signal does not automatically provide a reliable correction rule.**

The current evidence therefore supports detection and selective risk stratification more strongly than automatic prediction recovery.

---

### 7. Controlled representation movement produces ordered localization movement

To move beyond observational association, a controlled intervention was performed in the projected text-representation space.

The image and model were held fixed while the representation was moved along a predefined direction between language conditions.

Across **1,200 intervention trajectories**:

- all trajectories moved in the predicted primary direction;
- mean Spearman correlation between interpolation dose and projected localization displacement was **ρ = 0.999250**;
- all 12 case-level mean directional responses were positive.

This provides direct intervention evidence that controlled movement in the projected text representation can induce ordered movement in localization.

The result should be interpreted specifically at the **projected representation → localization** stage; it does not establish that every segmentation failure is caused exclusively by language.

---

### 8. Substantial language sensitivity is not universal

On Text3DSAM with 30 AMOS liver CT volumes and four semantically equivalent descriptions:

- **Mean Dice:** ≈ 0.9144
- **Mean pairwise |ΔDice|:** ≈ 0.00021
- **Mean within-case range:** ≈ 0.00039
- **Maximum observed range:** ≈ 0.00122
- **Cases with range > 0.01:** 0 / 30
- **Retrospective oracle advantage:** ≈ +0.00009

The evaluated Text3DSAM setting is therefore nearly invariant to the tested language descriptions.

This experiment is a **cross-system/task boundary condition**, not a controlled architectural comparison, because the model, anatomy, dataset, dimensionality, and language conditions differ from the MedCLIP-SAM experiments.

---

## Scientific Interpretation

Taken together, the experiments support four main conclusions:

- **Language conditioning can create substantial case-level instability even when aggregate metrics appear acceptable.**
- **In the evaluated 2D pipeline, catastrophic segmentation transitions are predominantly associated with localization changes.**
- **Localization disagreement can provide a useful, ground-truth-free reliability signal, including under no-retuning transfer.**
- **The effect is regime-dependent rather than universal: some evaluated systems and tasks are nearly invariant to the tested language variation.**

The work therefore focuses on **reliability**, rather than proposing a new segmentation architecture.

---

## Repository Structure

```text
beyond-average-dice/
├── scripts/
│   ├── breast_p20/                  # Breast97 broad-language experiments
│   ├── brain_p20/                   # Brain400 sensitivity and failure analyses
│   ├── brain600_highop/             # Brain600 localization analyses
│   ├── brain600_language_ladder/    # Brain600 six-condition analysis
│   ├── text3dsam/                   # AMOS30 Text3DSAM experiment
│   └── paper_final/                 # Final figure-generation utilities
│
├── results/
│   ├── breast_p20/                  # Breast97 case-level results
│   ├── brain_p20/                   # Brain400 sensitivity / mechanism outputs
│   ├── brain600_highop/             # Brain600 localization outputs
│   ├── brain600_language_ladder/    # Brain600 six-condition results
│   └── text3dsam/                   # Text3DSAM language-sensitivity results
│
├── remedy/
│   ├── protocol/                    # Frozen development / held-out protocols
│   ├── results/                     # Reliability, partitions, intervention outputs
│   ├── scripts/                     # Risk and mechanism experiments
│   └── figures/                     # Reliability / mechanism figures
│
├── paper_results/
│   ├── frozen/
│   │   └── MASTER_RESULTS.md        # Frozen manuscript-facing numerical record
│   └── FINAL_FIGURES/               # Figures, source data, provenance, hashes
│
├── reproducibility/
│   └── manuscript_qc/               # Independent manuscript reconstruction / QC
│
├── prompts/                          # Frozen Brain600 language conditions
├── docs/                             # Environments, upstream states, source patches
├── LICENSE_NOTICE.md
└── README.md
```

---

## Key Repository Components

### `results/`

Contains the primary frozen numerical outputs for the observational experiments, including:

- case-level Dice metrics;
- pairwise language sensitivity;
- catastrophic transition analyses;
- localization overlap analyses;
- residual downstream-failure analyses;
- Brain600 language-ladder results;
- bootstrap summaries;
- Text3DSAM language-sensitivity results.

### `remedy/`

Contains the later reliability and mechanism experiments, including:

- frozen development and held-out evaluation protocols;
- patient and slice partitions;
- localization-disagreement risk analyses;
- held-out evaluation;
- Brain600 stress/transfer evaluation;
- stochastic and representation-mechanism analyses;
- controlled embedding-trajectory experiments;
- associated manuscript figures.

Large generated arrays and scheduler logs are intentionally excluded from version control where they are not required for the public reproducibility record.

### `paper_results/frozen/`

`MASTER_RESULTS.md` provides the frozen manuscript-facing numerical record together with its SHA-256 checksum.

### `paper_results/FINAL_FIGURES/`

Contains manuscript figures together with figure-level source data, provenance tables, and checksums.

### `reproducibility/manuscript_qc/`

Contains an independent manuscript QC program and its outputs.

The QC reconstruction checks the principal Breast97, Brain400, Brain600, Text3DSAM, localization, and reliability quantities against the frozen manuscript assertions.

All **126 frozen numerical and structural assertions** passed the final QC reconstruction.

### `prompts/`

Contains frozen language-condition definitions used in the controlled Brain600 experiments.

### `docs/`

Contains supporting reproducibility information:

- upstream Git states;
- frozen Python environments;
- HPC environment information;
- source-file inventories;
- relevant modifications to upstream implementations.

---

## Experimental Regimes

### Breast97

The broad Breast evaluation contains **97 valid cases**.

One case from the original 98-case test set was excluded because its native image and mask geometries were incompatible.

Twenty official Breast P2 descriptions were evaluated individually.

### Brain400

The broad Brain evaluation contains:

- **400 slices**
- **174 patients**
- **20 descriptions per image**

Patient identity is retained for uncertainty estimation and held-out reliability evaluation.

Where appropriate, uncertainty intervals use **patient-cluster bootstrap resampling** rather than treating slices from the same patient as independent.

### Brain600

Brain600 is a separate released MedCLIP-SAMv2 cohort containing **600 images**.

The controlled language-ladder experiment evaluates six conditions constructed from the released image-specific descriptions.

Brain400 and Brain600 have zero filename overlap. Patient-level correspondence between the two released cohorts is unavailable and is not assumed.

Because patient identifiers are unavailable for Brain600, uncertainty estimates use image-level resampling where required.

### Text3DSAM / AMOS30

The boundary-condition experiment evaluates **30 AMOS abdominal CT liver volumes** using four prespecified semantically equivalent descriptions.

Because this experiment differs from the MedCLIP-SAM evaluations in model, dataset, anatomy, dimensionality, and language conditions, it is interpreted as a supporting cross-system/task contrast rather than a controlled architecture comparison.

---

## Reliability Evaluation Protocol

The ground-truth-free reliability experiment was designed to separate development from evaluation.

The repository preserves the frozen protocol under:

```text
remedy/protocol/
```

and the corresponding patient/slice partitions under:

```text
remedy/results/
```

The general sequence is:

```text
Multiple language conditions
        ↓
Localization outputs
        ↓
Localization disagreement
        ↓
Frozen risk rule
        ↓
Flag / retain
```

Ground truth is used for **evaluation of the reliability signal**, but the localization-disagreement score itself does not require ground truth at inference time.

The Brain600 transfer experiment reuses the frozen rule **without retuning**.

---

## Representation Intervention

The mechanism experiment directly manipulates the projected text representation while holding the image and model fixed.

Relevant scripts are under:

```text
remedy/scripts/
```

with analysis outputs under:

```text
remedy/results/embedding_trajectory/
```

and manuscript-facing figures under:

```text
remedy/figures/
```

This experiment is designed to test the narrower mechanistic statement:

> **Controlled movement in projected text-representation space produces ordered movement in localization.**

It should not be interpreted as complete causal identification of the full segmentation pipeline.

---

## Reproducibility and Provenance

The repository preserves multiple layers of reproducibility information:

- experiment and analysis scripts;
- exact language conditions;
- frozen development and held-out protocols;
- patient/slice partitions;
- frozen numerical results;
- bootstrap outputs;
- manuscript figure source data;
- figure provenance tables;
- SHA-256 checksums;
- upstream Git-state information;
- relevant upstream source modifications;
- frozen Python environments;
- independent manuscript-level QC.

The manuscript-facing results are therefore traceable from frozen result records to figure source data and QC assertions.

---

## Data, Models, and Large Artifacts

Raw medical datasets, pretrained model checkpoints, Hugging Face caches, Python environments, and large generated intermediate arrays are **not redistributed** through this repository.

They should be obtained from their original sources and remain subject to their respective licenses, terms of use, and access requirements.

In particular, this repository intentionally excludes large model/checkpoint formats and generated arrays from Git tracking.

---

## Upstream Implementations

This work builds on the released implementations of:

- **MedCLIP-SAM**
- **MedCLIP-SAMv2**
- **Text3DSAM**

The complete upstream repositories are not duplicated here.

Exact upstream Git states, environment snapshots, file inventories, and relevant local source modifications are recorded under:

```text
docs/
```

---

## Scope and Limitations

This repository studies **case-level reliability under language conditioning**.

It does **not** propose a new segmentation architecture, and the results should not be interpreted as showing that language-guided medical segmentation is universally unstable.

Important limitations include:

- language conditions are prespecified rather than an exhaustive sample of possible clinical phrasing;
- the detailed failure-mechanism analysis is concentrated on the evaluated 2D MedCLIP-SAM pipeline;
- the ground-truth-free risk signal is selective rather than definitive;
- some failures remain among retained cases;
- the automatic correction strategy evaluated during development did not improve performance;
- cross-system experiments differ in architecture, dataset, anatomy, dimensionality, and language conditions.

The evidence therefore supports **risk detection and mechanistic understanding** more strongly than automatic correction.

---

## Status

**Experimental phase: frozen.**

The principal observational, held-out reliability, transfer, mechanism-intervention, boundary-condition, figure-generation, and manuscript-QC analyses have been completed and frozen.

No additional substantive model experiments are planned unless a specific evidentiary gap requires further analysis.

---

## Citation

Citation information will be added upon manuscript release.

---

## License

See [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md) for repository-specific licensing and redistribution information.

Upstream code, datasets, pretrained models, and checkpoints remain subject to their original licenses and terms.
