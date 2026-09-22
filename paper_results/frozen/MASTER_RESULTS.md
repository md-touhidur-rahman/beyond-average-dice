# MASTER RESULTS — FROZEN SCIENTIFIC RECORD

Status: FROZEN FOR MANUSCRIPT
Rule: No manuscript number or claim should contradict this file.
Oracle quantities are retrospective and require ground truth.

============================================================
1. DEFINITIONS
============================================================

D_i,p:
Dice for case i under language condition p.

Case range:
max_p D_i,p - min_p D_i,p

Mean pairwise absolute sensitivity:
mean_{p<q} |D_i,p - D_i,q|

Best fixed condition:
argmax_p mean_i D_i,p

Retrospective oracle:
O_i = max_p D_i,p

Recoverability / oracle advantage:
mean_i O_i - performance of globally best fixed condition

Failure-success switch:
min_p D_i,p < 0.10 AND max_p D_i,p >= 0.50

Brain400 P20 catastrophic-tail pattern:
median_p D_i,p >= 0.50 AND min_p D_i,p < 0.10

IMPORTANT:
Failure-success switch, catastrophic-tail pattern, and range > 0.50
are different quantities and must never be used interchangeably.

============================================================
2. BROAD LANGUAGE CONDITIONS — BREAST97 P20
============================================================

N = 97 valid UDIAT cases.
Original N = 98.
benign_000062 excluded because of incompatible geometry.

20 released Breast P2 descriptions.

IMPORTANT CONFOUND:
Descriptions refer to mammography whereas evaluated UDIAT images are
ultrasound. Preserve this explicitly in Methods and Discussion.

These prompts are released semantically related descriptions.
Do NOT call them certified strict paraphrases.

Mean per-case pairwise |Delta Dice|:
0.104712673
95% CI: [0.080906833, 0.130152992]

Median per-case pairwise |Delta Dice|:
0.051957651

Mean within-case range:
0.340558194
95% CI: [0.272531423, 0.411040993]

Median range:
0.182057734

Maximum range:
0.959937378

Range > .01: 83.5%
Range > .05: 71.1%
Range > .10: 58.8%
Range > .20: 46.4%
Range > .30: 41.2%
Range > .50: 35.1%

Best fixed condition:
P13

Best fixed mean Dice:
0.499668543

Retrospective oracle mean Dice:
0.603951313

Oracle advantage:
+0.104282769
95% CI: [0.060423386, 0.135280023]

Bootstrap:
10,000 case-level resamples
RNG seed 42
best fixed condition reselected within each replicate

PERMITTED INTERPRETATION:
Broad released language conditions reveal substantial case-level
variation and retrospective recoverability.

NOT PERMITTED:
- strict paraphrase robustness
- modality mismatch is irrelevant
- oracle selection is deployable

============================================================
3. BROAD LANGUAGE CONDITIONS — BRAIN400 P20
============================================================

N = 400 slices
Patients = 174

20 released Brain P2 semantically closely related descriptions.
Do NOT claim guaranteed strict paraphrases.

Mean per-case pairwise |Delta Dice|:
0.111660177
95% CI: [0.098059708, 0.125045531]

Median:
0.07534

Mean within-case range:
0.385860607
95% CI: [0.345674955, 0.426065003]

Median range:
0.29086

Maximum range:
0.97263

Range > .01: 77.5%
Range > .05: 71.0%
Range > .10: 65.0%
Range > .20: 57.25%
Range > .30: 48.75%
Range > .50: 40.5%

Best fixed:
P14

Best fixed mean Dice:
0.476014

Retrospective oracle:
0.564146509

Oracle advantage:
+0.088132485
95% CI: [0.071647682, 0.105346473]

Bootstrap:
10,000 patient-cluster resamples
174 patients
RNG seed 42
slice-level estimand
best fixed reselected per replicate

PERMITTED:
Brain400, which does not share the Breast modality-text mismatch,
shows similarly substantial case-level sensitivity.

NOT PERMITTED:
This proves the Breast mismatch has no effect.

============================================================
4. BRAIN400 HEAVY-TAIL CHARACTERIZATION
============================================================

Cases with range > .50:
162 / 400

Among these 162:

Mean exact-zero prompts:
3.623

Median exact-zero prompts:
1

Mean prompts with Dice < .05:
4.278

Median:
2

Mean prompts with Dice >= .50:
12.599

Median:
16

P20 catastrophic-tail cases:
87 / 162 = 53.7%

Rescue-tail:
median Dice < .20 AND max Dice >= .50
28 / 162 = 17.3%

Mixed:
>=3 prompts Dice < .10 AND >=3 prompts Dice >= .50
66 / 162 = 40.7%

PERMITTED:
Many highly variable cases are not uniformly poor; successful and
catastrophic conditions coexist within the same case.

============================================================
5. BRAIN400 LOCALIZATION PROPAGATION
============================================================

Prompt-pair observations:
76,000

IMPORTANT:
These observations are non-independent.
Do NOT treat n=76,000 as inferential sample size.

Correlation:
box IoU vs |Delta Dice|

Pearson:
-0.5477065

Spearman:
-0.3963634

When box IoU >= .90:
mean |Delta Dice| = 0.0114
median |Delta Dice| = 0.0006

Among |Delta Dice| > .50:
N = 6,843

box IoU < .10:
66.3%

box IoU < .25:
74.3%

box IoU < .50:
90.0%

PERMITTED:
Large segmentation differences strongly coincide with localization
disagreement.

NOT PERMITTED:
Localization disagreement causally mediates the effect.

============================================================
6. BRAIN400 FAILURE-SUCCESS COMPONENT DECOMPOSITION
============================================================

Failure-success criterion:
exactly one condition Dice < .10
and the other Dice >= .50

Pairs:
5,398

Unique images:
132

Unique patients:
88

Unique case-prompt SAM evaluations:
2,340

Localization distribution:

box IoU < .10:
4,520 / 5,398 = 83.7347%

.10-.25:
253 = 4.69%

.25-.50:
315 = 5.84%

.50-.75:
190 = 3.52%

.75-.90:
95 = 1.76%

>= .90:
25 = 0.4631%

Therefore:

box IoU < .25:
88.4216%

box IoU < .50:
94.2571%

box IoU >= .75:
2.2230%

SAM rerun sanity:
max |old Dice - rerun Dice| = 0
mean difference = 0
N > 1e-6 = 0

ALL 5,398 PAIRS:

Mean selected |Delta Dice|:
0.7614825146

Median:
0.8008398320

Mean retrospective oracle-candidate |Delta Dice|:
0.7462894547

Median:
0.8076685078

Mean reduction:
0.0151930599

Different selected candidate index:
48.3327%

Failing side retrospective oracle candidate >= .20:
7.0211%

Failing side retrospective oracle candidate >= .50:
4.7240%

Failing-side selection gap > .05:
9.2997%

Failing-side selection gap > .20:
6.7988%

DOMINANT LOW-OVERLAP REGIME, box IoU < .10:

N = 4,520
images = 115
patients = 81

Selected mean |Delta Dice|:
0.767893

Oracle-candidate mean |Delta Dice|:
0.793607

Failing side oracle candidate >= .50:
0.0221%

Failing-side selection gap > .20:
0.1106%

RARE HIGH-OVERLAP REGIME, box IoU >= .90:

N = 25 pairs
6 images
6 patients

Selected mean |Delta Dice|:
0.679701

Oracle-candidate mean |Delta Dice|:
0.130282

Different selected candidate index:
96%

Failing side retrospective oracle candidate >= .50:
88%

Failing-side selection gap > .20:
100%

CRITICAL MANUSCRIPT REQUIREMENT:
Every substantive discussion of the >=.90 subgroup must prominently
state N=25 pairs from 6 images / 6 patients.

PERMITTED:
Catastrophic failures exhibit two descriptive/component-level regimes.
Most coincide with severe localization switching. The rare
high-localization-overlap residual subgroup is predominantly associated
with SAM-B candidate availability/selection behavior.

NOT PERMITTED:
- causal mediation
- candidate selection explains all failures
- high-overlap mechanism is common
- n=5,398 independent observations

============================================================
7. BRAIN600 CONTROLLED REFORMULATION / INFORMATION ABLATION
============================================================

N = 600 images.

Patient IDs unavailable.
Bootstrap unit = image.
Acknowledge possible optimism if multiple images originate from the
same patient.

Same cohort/model/inference configuration across H0-L5.

H0:
released original image-specific description

H1:
observation verb normalized

H2:
introductory imaging phrase normalized

L3:
standardized semantic reformulation preserving descriptive lesion tail

L4:
compressed and standardized lesion description retaining principal
lesion attributes and tumor identity; reporting/epistemic wording also
normalized

L5:
broad tumor-subtype description with case-specific morphology/location
removed

IMPORTANT:
L5 is NOT a paraphrase.
L4 is NOT pure shortening.
H0-L5 are NOT a scalar severity ladder.

Example — MENINGIOMA:

H0:
"A brain imaging study showing a homogenous, dural-based mass
suggestive of a meningioma tumor."

L3:
"Brain MRI demonstrates a homogenous, dural-based mass suggestive of
a meningioma tumor."

L4:
"A homogenous, dural-based mass consistent with a meningioma tumor."

L5:
"A brain MRI showing a meningioma tumor."

Example — GLIOMA:

H0:
"A brain imaging study revealing a diffuse, infiltrative mass
suggestive of a glioma tumor."

L3:
"Brain MRI demonstrates a diffuse, infiltrative mass suggestive of a
glioma tumor."

L4:
"A diffuse, infiltrative mass consistent with a glioma tumor."

L5:
"A brain MRI showing a glioma tumor."

Example — PITUITARY:

H0:
"A brain imaging scan showing a well-circumscribed, smoothly contoured
sellar mass indicative of a pituitary tumor."

L3:
"Brain MRI demonstrates a well-circumscribed, smoothly contoured sellar
mass indicative of a pituitary tumor."

L4:
"A well-circumscribed, smoothly contoured sellar mass consistent with
a pituitary tumor."

L5:
"A brain MRI showing a pituitary tumor."

------------------------------------------------------------
INTERNAL CONSISTENCY
------------------------------------------------------------

Previously frozen H0-H2 means:

H0 = 0.753830
H1 = 0.751326
H2 = 0.747262

Extended six-condition analysis:

H0 = 0.753830
H1 = 0.751326
H2 = 0.747262

Exact consistency at reported precision.

H0-H2 predictions were reused, not regenerated.

------------------------------------------------------------
CONDITION PERFORMANCE
------------------------------------------------------------

H0 mean Dice:
0.753830

H1:
0.751326

H2:
0.747262

L3:
0.740973

L4:
0.722931

L5:
0.715053

------------------------------------------------------------
H0-RELATIVE MEAN ABSOLUTE CHANGE
------------------------------------------------------------

H1:
0.020296
95% CI [0.014246, 0.027275]

H2:
0.029997
95% CI [0.022890, 0.037996]

L3:
0.033596
95% CI [0.025457, 0.042595]

L4:
0.080111
95% CI [0.066815, 0.093859]

L5:
0.072436
95% CI [0.059929, 0.085784]

------------------------------------------------------------
H0-RELATIVE SIGNED CHANGE
------------------------------------------------------------

H1:
-0.002504
95% CI [-0.009313, 0.004263]

H2:
-0.006568
95% CI [-0.014920, 0.001384]

L3:
-0.012858
95% CI [-0.022185, -0.004253]

L4:
-0.030899
95% CI [-0.045713, -0.015812]

L5:
-0.038777
95% CI [-0.053143, -0.024717]

------------------------------------------------------------
H0-RELATIVE LARGE CHANGES
------------------------------------------------------------

             >.10      >.20      >.50      failure-success
H1           4.17%     2.50%     0.67%       0.67%
H2           7.17%     3.83%     1.17%       1.00%
L3           7.50%     3.83%     1.67%       1.67%
L4          19.17%    11.83%     4.17%       3.50%
L5          17.33%    10.17%     4.17%       3.33%

------------------------------------------------------------
LOCALIZATION AGREEMENT WITH H0
------------------------------------------------------------

Mean box IoU:

H1 = 0.950444
H2 = 0.911022
L3 = 0.901447
L4 = 0.801115
L5 = 0.815897

Pearson(box IoU, |Delta Dice|):

H1 = -0.507591
H2 = -0.640317
L3 = -0.565096
L4 = -0.670352
L5 = -0.663577

Spearman:

H1 = -0.807861
H2 = -0.747556
L3 = -0.665864
L4 = -0.617284
L5 = -0.644946

------------------------------------------------------------
ALL SIX CONDITIONS
------------------------------------------------------------

Mean within-case range:
0.132872
95% CI [0.116237, 0.150268]

Median range:
0.036751

Maximum:
0.977128

Range > .01:
67.5%

Range > .05:
44.8333%

Range > .10:
33.1667%

Range > .20:
20.3333%

Range > .30:
13.8333%

Range > .50:
8.0%

Best fixed condition:
H0

Best fixed mean Dice:
0.753830

Retrospective oracle:
0.792847

Oracle advantage:
+0.039017
95% CI [0.030225, 0.047372]

PERMITTED:
Within the frozen Brain600 system/cohort, minor lexical/reformulation
conditions produced comparatively small typical deviations, whereas
compression/standardization and removal of case-specific descriptive
information produced substantially larger case-level divergence.

NOT PERMITTED:
- monotonic language-severity relationship
- L5 is an equivalent-semantic paraphrase
- L4 isolates shortening alone
- causal attribution solely to lexical distance

============================================================
8. BRAIN600 CASE VULNERABILITY
============================================================

Primary descriptive predictor:
GT lesion size / foreground burden

Spearman lesion area vs six-condition range:
rho = 0.112783
nominal p = 0.005681
bootstrap 95% CI [0.033028, 0.192705]

Spearman lesion area vs mean pairwise absolute sensitivity:
rho = 0.112946
bootstrap 95% CI [0.033259, 0.192977]

Lesion fraction vs range:
rho = 0.114831
bootstrap 95% CI [0.035042, 0.194226]

Lesion fraction vs mean pairwise:
rho = 0.114470
bootstrap 95% CI [0.034512, 0.194276]

Quartile mean ranges:

Q1 smallest:
0.135482

Q2:
0.142642

Q3:
0.099458

Q4 largest:
0.153905

Range > .50:

Q1: 9.33%
Q2: 10.00%
Q3: 4.00%
Q4: 8.67%

For range > .50 cases:
median lesion area = 2193.5 px

Other cases:
2991.0 px

PERMITTED:
Lesion size shows only a weak positive rank association with
sensitivity, while quartile and extreme-tail analyses show no
consistent monotonic size gradient. Lesion size alone does not provide
a simple explanation for case-level sensitivity.

NOT PERMITTED:
- smaller lesions are generally more sensitive
- lesion size has no association whatsoever

============================================================
9. TEXT3DSAM BOUNDARY CONDITION
============================================================

Dataset:
AMOS liver CT

N = 30

Conditions:
4 semantically equivalent dataset-authored liver descriptions

Mean Dice:

P0 = 0.91436015
P1 = 0.91425480
P2 = 0.91443154
P3 = 0.91431402

Mean within-case range:
0.000389620

Mean pairwise absolute sensitivity:
0.000209918

Maximum range:
0.00121674

Cases with range > .01:
0 / 30

Retrospective oracle:
0.91452455

Best fixed:
P2 = 0.91443154

Oracle advantage:
+0.00009301

PERMITTED:
The large variation observed under broad MedCLIP-SAMv2 language
conditions did not reproduce in this evaluated Text3DSAM liver setting.

NOT PERMITTED:
- Text3DSAM is intrinsically more robust
- architecture causes the difference
- universal near-invariance of Text3DSAM

============================================================
10. CROSS-REGIME INTERPRETATION
============================================================

SUPPORTED:

Language-conditioned segmentation reliability is regime-dependent.

Broad semantically related descriptions can expose substantial and
heavy-tailed case-level variation.

At the evaluated strong Brain600 operating point, minor controlled
lexical/form reformulations are typically much more stable, while
compression/standardization and removal of descriptive information
increase case-level divergence.

A separate Text3DSAM liver setting provides a near-invariant boundary
condition.

UNSUPPORTED:

Any single causal explanation for differences across Breast97,
Brain400, Brain600, and Text3DSAM.

These settings differ in multiple dimensions including dataset,
language regime, system, anatomy/modality, dimensionality and
segmentation configuration.

============================================================
11. GLOBAL CLAIM BLACKLIST
============================================================

DO NOT CLAIM:

- first demonstration of prompt sensitivity
- first prompt-robustness study
- first uncertainty-aware SAM analysis
- first localization failure analysis
- universal language instability
- architecture superiority
- causal mediation by localization
- deployable oracle prompt selection
- strict paraphrase robustness for P20
- Text3DSAM is inherently robust
- smaller lesions cause failures
- monotonic language-severity response
- L5 preserves equivalent semantics
- L4 changes only prompt length

============================================================
12. CORE MANUSCRIPT THESIS
============================================================

"Aggregate segmentation performance can conceal severe case-specific
failures under language conditioning. These failures are heavy-tailed,
partially recoverable under alternative descriptions, strongly
associated with changes in upstream localization in the evaluated
CLIP-to-SAM pathway, and dependent on the evaluated language regime
and operating setting."

============================================================
13. FROZEN CONTRIBUTIONS
============================================================

C1 — Case-level reliability characterization

Characterize language-conditioned medical segmentation beyond
aggregate Dice using within-case sensitivity, catastrophic
failure-success transitions, and retrospective fixed-versus-oracle
recoverability.

C2 — Failure-component decomposition

Trace catastrophic MedCLIP-SAMv2 prompt-conditioned differences
through localization agreement and downstream SAM-B candidate
behavior, identifying a dominant severe-localization-disagreement
regime and a rare high-overlap residual regime.

C3 — Controlled language analysis

Within a frozen Brain600 cohort/system/configuration, compare original
descriptions, controlled lexical/reformulation conditions,
compression/standardization, and removal of case-specific descriptive
information.

C4 — Boundary characterization

Show that measured language-conditioned reliability differs
substantially across broad P20 conditions, controlled Brain600
conditions, and the evaluated Text3DSAM liver setting, without
assigning those cross-setting differences to one causal factor.

============================================================
14. REPRODUCIBILITY NOTES
============================================================

Brain600 L3 SHA256:
e0324e896c3685cdfe41befe383bfc5074208aa69c3239f6a5d3bac6af2cd6d1

Brain600 L4 SHA256:
af7d58480cdc0fbd7795fdc0ad5fa0212ecb3c4b9b81fed328e601740b8aac59

Brain600 L5 SHA256:
7bc0f2268c4ed680030dfa857838405bb7275ad1be250516381f892dc3973963

Prompt manifest SHA256:
2e460618f572872979974c9949d9b9ca2f57db537851497eaef1c39b4a2383d6

Frozen Brain600 analysis protocol SHA256:
9ad00afb5cada848fb264266c6aada92c19a3f2ce4926691a597f750e6e25a69

Brain600 bootstrap:
10,000 image resamples
RNG 42

Brain400 bootstrap:
patient-cluster bootstrap
174 patients
10,000 resamples
RNG 42
