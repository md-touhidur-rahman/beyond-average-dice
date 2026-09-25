# REMEDY PROTOCOL FREEZE — 2026-09-22

## 1. Purpose

This study tests whether the failure structure previously observed in Brain400 can motivate a ground-truth-free reliability mechanism for language-conditioned medical image segmentation.

The original Brain400 study established that severe segmentation divergence frequently coincides with changes in language-conditioned localization. The present extension asks whether cross-description localization consistency can be used to:

1. identify cases at elevated risk of language-conditioned failure;
2. rank description-specific localization hypotheses without ground truth; and
3. motivate a localization-level remedy that reduces failure on held-out patients.

## 2. Prior-data disclosure

All Brain400 analyses completed before 2026-09-22 are exploratory/hypothesis-generating for this extension.

The full Brain400 cohort has already informed the hypothesis that localization disagreement is associated with segmentation failure. The subsequent held-out partition is therefore not historically untouched external validation.

From this protocol freeze onward:

- patients are partitioned once into development and held-out evaluation sets;
- no patient may occur in both sets;
- method construction and selection use development patients only;
- held-out segmentation outcomes must not influence method selection;
- after held-out evaluation is opened, the method will not be modified and reevaluated on the same held-out patients as confirmatory evidence.

## 3. Source-data invariants

Authoritative input:

    ../results/brain_p20/brain400_p20.csv

Frozen checks:

- 400 slices;
- 174 patients;
- 20 language descriptions per slice;
- 76,000 unordered within-slice description pairs;
- 5,398 frozen failure-success pair observations.

Patient ID is parsed from filename using `pid-<ID>`.

## 4. Frozen failure definition

A slice is labeled catastrophic if at least one of its 20 descriptions has Dice < 0.10 and at least one description has Dice >= 0.50.

A patient-level stratification label is positive if that patient contains at least one catastrophic slice.

This stratification label is used only to balance prevalence across development and held-out partitions. It does not replace later analysis endpoints.

## 5. Patient partition

Patients, not slices, are the partitioning unit.

Target allocation:

- 70% development;
- 30% held-out evaluation.

The split is stratified by the frozen binary patient-level catastrophic label.

Random seed: 42.

The resulting patient assignment is written to disk and hashed before Experiment 1 outcomes are analyzed.

## 6. Experiment 1A — case-level GT-free failure detection

Question:

Can cross-description localization disagreement identify slices at elevated risk of catastrophic language-conditioned failure without using ground truth at inference time?

Primary outcome:

- AUPRC for the frozen catastrophic slice label.

Secondary outcome:

- AUROC.

Candidate GT-free localization signals are computed exclusively from the 20 predicted boxes:

1. one minus mean pairwise box IoU;
2. one minus median pairwise box IoU;
3. normalized box-center dispersion;
4. normalized box-area dispersion.

The development-set signal with the highest AUPRC is selected. Ties are resolved by higher AUROC and then by the candidate order above.

No Dice value or ground-truth mask is an input to any candidate signal.

## 7. Case-level intervention gate

Using the selected Experiment 1A signal, a development-only threshold is selected to achieve sensitivity >= 0.80 for catastrophic slices.

Among thresholds satisfying this constraint, choose the threshold with greatest specificity.

If no threshold reaches sensitivity >= 0.80, choose the threshold with maximal Youden J.

The threshold is frozen before held-out evaluation.

## 8. Experiment 1B — prompt-level trust ranking

For each slice and valid description-specific box b_k, define:

    C_k = mean_{j != k} IoU(b_k, b_j)

The medoid localization is the observed box with maximal C_k.

The selector uses localization boxes only and does not use Dice or ground truth.

Primary evaluation subset:

- catastrophic development slices.

Primary endpoint:

    regret_i = max_k Dice_ik - Dice_i,medoid

The primary summary is mean regret with patient-cluster bootstrap uncertainty.

A random-description selector provides a reference comparison.

Secondary descriptive endpoints may include top-1 oracle hit rate and rank association, but they cannot replace the frozen primary endpoint.

## 9. Medoid edge cases

Only valid boxes participate in medoid computation.

If two or more descriptions have exactly equal maximal agreement, select the lowest frozen prompt index.

Primary medoid analysis requires at least three valid description-specific boxes.

Cases with fewer than three valid boxes are reported separately and use the globally fixed baseline description for any downstream intervention.

The frequency of ties and insufficient-valid-box cases is reported.

## 10. Localization remedy

The primary remedy is consistency-gated medoid localization.

For a slice whose frozen case-level disagreement score does not exceed the frozen gate threshold:

- retain the globally fixed baseline description/localization.

For a slice whose score exceeds the threshold:

- replace the baseline localization with the medoid localization selected from the observed description-specific boxes.

The globally fixed baseline is P14, previously identified in the frozen Brain400 analysis as the best fixed description.

The remedy does not use ground truth during selection.

## 11. Prespecified comparator

Coordinate-wise median localization is retained as a sensitivity comparator, not as the primary remedy.

The primary rationale for medoid localization is that it selects an actually observed language-conditioned spatial hypothesis rather than synthesizing a new box.

## 12. Evaluation endpoints

Primary held-out remedy endpoint:

- paired difference in mean Dice relative to fixed P14.

Secondary endpoints:

- catastrophic failure rate;
- lower-tail Dice;
- proportion of slices improved;
- proportion degraded;
- worst observed degradation;
- case-level variability where applicable.

All comparisons are paired at slice level while uncertainty respects patient clustering.

## 13. Statistical uncertainty

Brain400 confidence intervals use patient-cluster bootstrap resampling.

Bootstrap replicates: 10,000.

Bootstrap random seed: 42.

All slices belonging to a sampled patient are retained together.

## 14. Brain600 role

Brain600 is not the primary validation cohort.

It differs from Brain400 in language regime and SAM configuration and is used only as a separate cross-regime stress test after the Brain400 method is frozen.

Brain600 must not be used to rescue or redesign a method after unfavorable Brain400 held-out results.

## 15. No-peeking and stopping rules

Development data may be used for the choices explicitly permitted above.

Before held-out evaluation:

- candidate signals are fixed;
- signal-selection rule is fixed;
- gate-selection rule is fixed;
- medoid definition is fixed;
- baseline is fixed;
- remedy is fixed;
- primary endpoint is fixed.

Held-out outcomes are evaluated only after these choices are frozen.

An unfavorable held-out result is retained and reported.

The held-out set must not subsequently be used to tune a replacement method and present that replacement as independently validated on the same patients.
