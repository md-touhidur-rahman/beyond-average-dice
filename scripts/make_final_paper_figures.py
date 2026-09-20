from pathlib import Path
import csv
import numpy as np

# Use non-interactive backend on HPC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MED = ROOT / "MedCLIP-SAM"
T3D = ROOT / "Text3DSAM"

OUT = ROOT / "paper_results"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def get_numeric(rows, candidates):
    for c in candidates:
        if c in rows[0]:
            return np.array([float(r[c]) for r in rows], dtype=float), c
    raise KeyError(
        f"None of {candidates} found.\n"
        f"Available columns:\n{list(rows[0].keys())}"
    )

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# Load frozen datasets
# ------------------------------------------------------------
brain = read_csv(MED / "part2/action400_full_matrix.csv")
breast = read_csv(
    MED / "part2/breast98_results/breast98_results.csv"
)
t3d = read_csv(
    T3D / "part2/text3dsam_amos30_liver_language.csv"
)

assert len(brain) == 400
assert len(breast) == 97
assert len(t3d) == 30

# Detect action column names robustly
B1, b1n = get_numeric(brain, ["A1_dice", "dice_A1", "A1"])
B2, b2n = get_numeric(brain, ["A2_dice", "dice_A2", "A2"])
B3, b3n = get_numeric(brain, ["A3_dice", "dice_A3", "A3"])

R1, r1n = get_numeric(breast, ["A1_dice", "dice_A1", "A1"])
R2, r2n = get_numeric(breast, ["A2_dice", "dice_A2", "A2"])
R3, r3n = get_numeric(breast, ["A3_dice", "dice_A3", "A3"])

T = np.array([
    [
        float(r["dice_P0"]),
        float(r["dice_P1"]),
        float(r["dice_P2"]),
        float(r["dice_P3"]),
    ]
    for r in t3d
])

print("Detected Brain:", b1n, b2n, b3n)
print("Detected Breast:", r1n, r2n, r3n)

# ------------------------------------------------------------
# Derived quantities
# ------------------------------------------------------------
brain_lang = np.abs(B1 - B2)
brain_down = np.abs(B2 - B3)

breast_lang = np.abs(R1 - R2)
breast_down = np.abs(R2 - R3)

brain_oracle = np.maximum.reduce([B1, B2, B3])
breast_oracle = np.maximum.reduce([R1, R2, R3])

brain_best_fixed = max(B1.mean(), B2.mean(), B3.mean())
breast_best_fixed = max(R1.mean(), R2.mean(), R3.mean())

t3d_range = T.max(axis=1) - T.min(axis=1)

t3d_pairwise = np.stack([
    np.abs(T[:, i] - T[:, j])
    for i in range(4)
    for j in range(i + 1, 4)
], axis=1)

t3d_pairwise_case = t3d_pairwise.mean(axis=1)

# ------------------------------------------------------------
# Figure 1
# Case-level language sensitivity
# ------------------------------------------------------------
plt.figure(figsize=(6.6, 4.3))

bp = plt.boxplot(
    [brain_lang, breast_lang],
    labels=["Brain\nN=400", "Breast\nN=97"],
    showfliers=False,
    widths=0.55,
)

rng = np.random.default_rng(42)

for x, vals in enumerate([brain_lang, breast_lang], start=1):
    jitter = rng.normal(0, 0.045, len(vals))
    plt.scatter(
        np.full(len(vals), x) + jitter,
        vals,
        s=10,
        alpha=0.35,
    )

plt.ylabel(r"Absolute Dice change $|A_1-A_2|$")
plt.title("Case-level sensitivity to language/localization strategy")
plt.grid(axis="y", alpha=0.2)

savefig("fig1_language_sensitivity")

# ------------------------------------------------------------
# Figure 2
# Language intervention vs downstream SAM intervention
# ------------------------------------------------------------
plt.figure(figsize=(6.8, 4.4))

labels = ["Brain", "Breast"]
lang_means = [brain_lang.mean(), breast_lang.mean()]
down_means = [brain_down.mean(), breast_down.mean()]

x = np.arange(2)
w = 0.34

plt.bar(
    x - w/2,
    lang_means,
    width=w,
    label="Language/localization change\n|A1−A2|",
)
plt.bar(
    x + w/2,
    down_means,
    width=w,
    label="SAM-B→SAM-H change\n|A2−A3|",
)

plt.xticks(x, labels)
plt.ylabel("Mean absolute Dice change")
plt.title("Controlled intervention magnitude")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.2)

for i in range(2):
    ratio = lang_means[i] / down_means[i]
    ymax = max(lang_means[i], down_means[i])
    plt.text(
        i,
        ymax + 0.008,
        f"{ratio:.2f}×",
        ha="center",
        fontsize=10,
    )

savefig("fig2_language_vs_segmenter")

# ------------------------------------------------------------
# Figure 3
# Fixed action versus per-case oracle
# ------------------------------------------------------------
plt.figure(figsize=(6.8, 4.4))

fixed = [brain_best_fixed, breast_best_fixed]
oracle = [brain_oracle.mean(), breast_oracle.mean()]

x = np.arange(2)
w = 0.34

plt.bar(x - w/2, fixed, width=w, label="Best fixed action")
plt.bar(x + w/2, oracle, width=w, label="Per-case oracle")

plt.xticks(x, ["Brain", "Breast"])
plt.ylabel("Mean Dice")
plt.title("Action heterogeneity leaves recoverable headroom")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.2)

for i in range(2):
    gain = oracle[i] - fixed[i]
    plt.text(
        i,
        max(fixed[i], oracle[i]) + 0.015,
        f"+{100*gain:.2f} Dice pts",
        ha="center",
        fontsize=10,
    )

savefig("fig3_oracle_headroom")

# ------------------------------------------------------------
# Figure 4
# Independent architecture contrast
#
# IMPORTANT:
# Different perturbation definitions.
# Brain/Breast = |A1-A2| strategy change.
# Text3DSAM = mean pairwise difference among 4 prompts.
# ------------------------------------------------------------
plt.figure(figsize=(7.0, 4.5))

values = [
    brain_lang.mean(),
    breast_lang.mean(),
    t3d_pairwise_case.mean(),
]

names = [
    "MedCLIP-SAM\nBrain",
    "MedCLIP-SAM\nBreast",
    "Text3DSAM\nAMOS liver",
]

x = np.arange(3)
plt.bar(x, values, width=0.58)

plt.xticks(x, names)
plt.ylabel("Mean absolute Dice variation")
plt.title("Language sensitivity is pipeline-dependent")
plt.grid(axis="y", alpha=0.2)

for i, v in enumerate(values):
    plt.text(
        i,
        v + 0.004,
        f"{v:.4f}",
        ha="center",
        fontsize=10,
    )

plt.figtext(
    0.5,
    -0.03,
    "Perturbations differ: MedCLIP-SAM compares single-prompt vs "
    "prompt-ensemble localization; Text3DSAM compares four "
    "dataset-authored semantically equivalent prompts.",
    ha="center",
    fontsize=8,
    wrap=True,
)

savefig("fig4_pipeline_dependence")

# ------------------------------------------------------------
# Main result table
# ------------------------------------------------------------
table_rows = [
    {
        "dataset": "Brain",
        "pipeline": "MedCLIP-SAM",
        "N": 400,
        "A1_or_P0_mean": B1.mean(),
        "A2_or_best_alt_mean": B2.mean(),
        "language_sensitivity": brain_lang.mean(),
        "downstream_sensitivity": brain_down.mean(),
        "ratio": brain_lang.mean() / brain_down.mean(),
        "oracle": brain_oracle.mean(),
        "best_fixed": brain_best_fixed,
        "oracle_gain": brain_oracle.mean() - brain_best_fixed,
    },
    {
        "dataset": "Breast",
        "pipeline": "MedCLIP-SAM",
        "N": 97,
        "A1_or_P0_mean": R1.mean(),
        "A2_or_best_alt_mean": R2.mean(),
        "language_sensitivity": breast_lang.mean(),
        "downstream_sensitivity": breast_down.mean(),
        "ratio": breast_lang.mean() / breast_down.mean(),
        "oracle": breast_oracle.mean(),
        "best_fixed": breast_best_fixed,
        "oracle_gain": breast_oracle.mean() - breast_best_fixed,
    },
    {
        "dataset": "AMOS liver",
        "pipeline": "Text3DSAM",
        "N": 30,
        "A1_or_P0_mean": T[:,0].mean(),
        "A2_or_best_alt_mean": T.mean(axis=0).max(),
        "language_sensitivity": t3d_pairwise_case.mean(),
        "downstream_sensitivity": np.nan,
        "ratio": np.nan,
        "oracle": T.max(axis=1).mean(),
        "best_fixed": T.mean(axis=0).max(),
        "oracle_gain":
            T.max(axis=1).mean() - T.mean(axis=0).max(),
    },
]

fields = list(table_rows[0].keys())

with open(OUT / "table1_headline_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(table_rows)

# ------------------------------------------------------------
# Console verification
# ------------------------------------------------------------
print()
print("=== FINAL CHECK ===")

print(
    "Brain language/downstream/ratio:",
    brain_lang.mean(),
    brain_down.mean(),
    brain_lang.mean()/brain_down.mean(),
)

print(
    "Breast language/downstream/ratio:",
    breast_lang.mean(),
    breast_down.mean(),
    breast_lang.mean()/breast_down.mean(),
)

print(
    "Text3DSAM pairwise/range:",
    t3d_pairwise_case.mean(),
    t3d_range.mean(),
)

print(
    "Brain oracle gain:",
    brain_oracle.mean() - brain_best_fixed,
)

print(
    "Breast oracle gain:",
    breast_oracle.mean() - breast_best_fixed,
)

print(
    "Text3DSAM oracle gain:",
    T.max(axis=1).mean() - T.mean(axis=0).max(),
)

print()
print("Saved publication assets to:", OUT)
