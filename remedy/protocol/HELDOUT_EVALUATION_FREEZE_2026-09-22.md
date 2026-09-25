# HELD-OUT EVALUATION FREEZE — 2026-09-22

This document freezes the method after development analysis and before any
Brain400 held-out segmentation outcome is accessed.

It supplements, but does not modify, the original frozen protocol:

REMEDY_PROTOCOL_FREEZE_2026-09-22.md
SHA256:
333321b13041e5985d2f09b937dbd46ad6e9ef1df3f8d560d7d60d3552c0f1d0

## 1. Partition

Development:
- 121 patients
- 268 slices
- 89 catastrophic slices

Held-out:
- 53 patients
- 132 slices
- 43 catastrophic slices

Patient partition SHA256:
a3ef91ccada0bd55c5f5dba2d036317d9c7f97eced77b05fc1e17b752819f3fe

Slice partition SHA256:
351fc37945228536809f1874acd07907d9dee87960c5a1ac327a8b73a2a25bf5

No held-out segmentation outcomes were accessed during method development.

## 2. Development-selected GT-free risk score

The prespecified Experiment 1A selection rule selected:

    risk = 1 - mean pairwise IoU among the 20 description-specific
           localization boxes

Development performance for the frozen catastrophic-slice target:

- AUPRC = 0.477679
- AUROC = 0.684389

No Dice value, ground-truth mask, SAM confidence, or patient outcome is an
input to this risk score.

Higher score indicates greater localization disagreement.

## 3. Frozen threshold

The threshold was selected using the prespecified development-only rule:

1. require catastrophic-slice sensitivity >= 0.80;
2. among qualifying thresholds maximize specificity;
3. deterministic threshold selection as implemented in the frozen
   development script.

Frozen threshold:

    0.264446087201

Development sensitivity:

    0.808989

Development specificity:

    0.480447

This threshold will not be changed after held-out evaluation.

## 4. Prompt-level trust result

On 89 catastrophic development slices, medoid localization selection reduced
regret relative to a uniformly random description:

Mean medoid regret:

    0.200817

Patient-cluster bootstrap 95% CI:

    [0.142524, 0.265000]

Mean random-description expected regret:

    0.278529

Mean paired medoid-minus-random regret:

    -0.077711

Patient-cluster bootstrap 95% CI:

    [-0.106583, -0.049444]

This establishes that cross-description localization agreement contains
information about relative localization quality, but does not establish that
the medoid outperforms the strong fixed P14 baseline.

## 5. Failed automatic correction

The originally proposed consistency-gated medoid replacement was tested on
development patients only.

Development result:

- fixed P14 mean Dice = 0.475400
- gated-medoid mean Dice = 0.446724
- difference = -0.028676
- fixed P14 Dice < 0.10 rate = 34.701%
- gated-medoid Dice < 0.10 rate = 37.687%
- worst observed slice-level change = -0.944994

Therefore consistency-gated medoid replacement is considered falsified as
the primary automatic remedy and will not be promoted as the held-out
correction method.

This negative result is retained rather than hidden.

## 6. Frozen held-out method

The held-out method is selective reliability / abstention using the already
selected GT-free localization-disagreement score and already selected
threshold.

For each held-out slice:

1. compute the 20 description-specific localization boxes;
2. compute risk = 1 - mean pairwise box IoU;
3. if risk > 0.264446087201, flag the slice as high risk;
4. otherwise retain it as lower risk.

The method does not choose a replacement segmentation.

P14 remains the frozen fixed segmentation baseline.

"Abstention" means that the automated system does not treat the flagged
prediction as an unqualified retained prediction. No claim is made here
about the performance of a human reviewer or unspecified fallback system.

## 7. Primary held-out endpoint

Primary endpoint:

AUPRC of the frozen localization-disagreement score for the frozen
catastrophic-slice target.

This evaluates whether GT-free localization disagreement transfers to unseen
patients as a failure-risk signal.

## 8. Key secondary held-out endpoints

The following are frozen secondary endpoints:

1. AUROC for catastrophic-slice detection.

2. At the frozen threshold:
   - sensitivity for catastrophic slices;
   - specificity for non-catastrophic slices;
   - fraction of slices flagged;
   - positive predictive value;
   - negative predictive value.

3. Selective reliability of fixed P14:
   - mean Dice among retained slices;
   - median Dice among retained slices;
   - Dice < 0.10 rate among retained slices.

4. Failure enrichment:
   - P14 Dice < 0.10 rate among flagged slices;
   - catastrophic-slice prevalence among flagged slices;
   - catastrophic-slice prevalence among retained slices.

5. Failure capture:
   - fraction of all catastrophic slices flagged;
   - fraction of all P14 Dice < 0.10 slices flagged.

## 9. Descriptive risk-coverage analysis

For comparability with development analysis, held-out descriptive
risk-coverage values will also be reported at fixed rejection fractions:

    5%, 10%, 20%, 30%, 40%, 50%

These fractions are descriptive only.

They are not alternative tuned operating thresholds and will not replace the
frozen threshold as the prespecified operating point.

## 10. Uncertainty

Uncertainty will respect patient clustering.

For held-out performance estimates, 10,000 patient-cluster bootstrap
replicates will be used with random seed 42.

Patients, rather than slices, are the resampling unit.

## 11. Interpretation constraints

A successful held-out result may support the claim that cross-description
localization disagreement provides a GT-free signal for identifying elevated
risk of language-conditioned segmentation failure.

It does not establish:
- that disagreement proves a segmentation is wrong;
- that agreement proves a segmentation is correct;
- that the method automatically repairs a failed segmentation;
- that flagged cases would necessarily be corrected by human review;
- clinical utility or clinical validation.

The failed medoid correction remains part of the development evidence.

## 12. No post-held-out tuning

After held-out outcomes are opened:

- the risk score will not be replaced;
- the threshold will not be changed;
- the target definition will not be changed;
- the primary endpoint will not be changed;
- another method will not be tuned on the same held-out patients and
  described as independently validated.

Unexpected or unfavorable held-out results will be retained and reported.
