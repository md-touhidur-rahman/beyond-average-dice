from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
)

# ============================================================
# OUTPUT
# ============================================================

OUT = Path("beyond-average-dice/remedy/figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

STEM = OUT / "fig_method_workflow"

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

NAVY = "#17324D"
BLUE = "#2C7FB8"
LIGHT_BLUE = "#EAF4FB"

GREEN = "#2A9D8F"
LIGHT_GREEN = "#EAF7F3"

ORANGE = "#D97706"
LIGHT_ORANGE = "#FFF4E5"

PURPLE = "#7C5AC7"
LIGHT_PURPLE = "#F2EDFB"

RED = "#C84C4C"
LIGHT_RED = "#FBECEC"

GRAY = "#5E6872"
MID_GRAY = "#AAB2B9"
LIGHT_GRAY = "#F5F6F7"
BLACK = "#202428"
WHITE = "#FFFFFF"

# ============================================================
# HELPERS
# ============================================================

def rounded_box(
    ax, x, y, w, h,
    facecolor=WHITE,
    edgecolor=MID_GRAY,
    lw=1.2,
    radius=0.018,
    zorder=1,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def txt(
    ax, x, y, s,
    size=10.5,
    weight="normal",
    color=BLACK,
    ha="center",
    va="center",
    rotation=0,
    zorder=10,
    linespacing=1.15,
):
    ax.text(
        x, y, s,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=zorder,
        linespacing=linespacing,
    )


def arrow(
    ax, x1, y1, x2, y2,
    color=NAVY,
    lw=1.5,
    mutation_scale=13,
    style="-|>",
    connectionstyle="arc3",
    zorder=5,
):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    ax.add_patch(a)
    return a


def section_label(ax, x, y, number, title, color):
    circ = Circle(
        (x, y),
        0.017,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="none",
        zorder=10,
    )
    ax.add_patch(circ)
    txt(ax, x, y, str(number), size=10.5, weight="bold", color=WHITE)
    txt(
        ax, x + 0.028, y, title,
        size=12.2, weight="bold", color=NAVY,
        ha="left"
    )


def mini_image_icon(ax, x, y, w=0.042, h=0.050):
    """Abstract image icon: deliberately not a medical-image rendering."""
    r = Rectangle(
        (x - w/2, y - h/2), w, h,
        transform=ax.transAxes,
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        linewidth=1.0,
        zorder=5,
    )
    ax.add_patch(r)

    c1 = Circle(
        (x - 0.007, y + 0.006),
        0.010,
        transform=ax.transAxes,
        facecolor="#D5DADF",
        edgecolor="none",
        zorder=6,
    )
    c2 = Circle(
        (x + 0.009, y - 0.006),
        0.008,
        transform=ax.transAxes,
        facecolor="#B9C1C8",
        edgecolor="none",
        zorder=6,
    )
    ax.add_patch(c1)
    ax.add_patch(c2)


def heatmap_icon(ax, x, y):
    """Abstract localization field."""
    for radius, color in [
        (0.025, LIGHT_ORANGE),
        (0.018, "#F7C873"),
        (0.011, "#E9943A"),
        (0.0055, RED),
    ]:
        c = Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
            zorder=6,
        )
        ax.add_patch(c)


def bbox_icon(ax, x, y):
    r = Rectangle(
        (x - 0.021, y - 0.017),
        0.042, 0.034,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor=ORANGE,
        linewidth=2.0,
        zorder=7,
    )
    ax.add_patch(r)


def mask_icon(ax, x, y):
    c1 = Circle(
        (x - 0.007, y),
        0.016,
        transform=ax.transAxes,
        facecolor=GREEN,
        edgecolor="none",
        alpha=0.9,
        zorder=6,
    )
    c2 = Circle(
        (x + 0.008, y + 0.006),
        0.012,
        transform=ax.transAxes,
        facecolor=GREEN,
        edgecolor="none",
        alpha=0.9,
        zorder=6,
    )
    ax.add_patch(c1)
    ax.add_patch(c2)


# ============================================================
# CANVAS
# ============================================================

# Wide enough for a two-column figure while keeping labels readable.
fig, ax = plt.subplots(figsize=(15.8, 9.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ============================================================
# TITLE
# ============================================================

txt(
    ax, 0.5, 0.965,
    "Study design for evaluating language-conditioned segmentation reliability",
    size=17,
    weight="bold",
    color=NAVY,
)

txt(
    ax, 0.5, 0.933,
    "Observational case-level evaluation, failure localization, ground-truth-free risk assessment, "
    "and controlled representation intervention",
    size=10.6,
    color=GRAY,
)

# ============================================================
# 1. CORE COMPUTATIONAL PATHWAY
# ============================================================

rounded_box(
    ax, 0.055, 0.695, 0.89, 0.185,
    facecolor=LIGHT_BLUE,
    edgecolor="#9DCBE6",
    lw=1.3,
)

section_label(
    ax, 0.080, 0.850,
    1, "Language-conditioned segmentation pathway", BLUE
)

# Input image
mini_image_icon(ax, 0.115, 0.775)
txt(ax, 0.115, 0.724, "Medical image", size=9.5, weight="bold")

# Description
rounded_box(
    ax, 0.185, 0.744, 0.120, 0.064,
    facecolor=WHITE,
    edgecolor=BLUE,
    lw=1.1,
)
txt(
    ax, 0.245, 0.776,
    "Target\ndescription",
    size=9.6,
    weight="bold",
)

# Text representation
rounded_box(
    ax, 0.355, 0.744, 0.125, 0.064,
    facecolor=WHITE,
    edgecolor=BLUE,
    lw=1.1,
)
txt(
    ax, 0.4175, 0.776,
    "Projected text\nrepresentation",
    size=9.2,
    weight="bold",
)

# Localization heatmap
heatmap_icon(ax, 0.555, 0.777)
txt(
    ax, 0.555, 0.724,
    "Localization\nheatmap",
    size=9.4,
    weight="bold",
)

# Bounding box
bbox_icon(ax, 0.685, 0.777)
txt(
    ax, 0.685, 0.724,
    "Spatial prompt\n(bounding box)",
    size=9.4,
    weight="bold",
)

# SAM
rounded_box(
    ax, 0.760, 0.744, 0.075, 0.064,
    facecolor=LIGHT_ORANGE,
    edgecolor=ORANGE,
    lw=1.1,
)
txt(ax, 0.7975, 0.776, "SAM", size=11, weight="bold")

# Mask
mask_icon(ax, 0.895, 0.777)
txt(
    ax, 0.895, 0.724,
    "Segmentation\nmask",
    size=9.4,
    weight="bold",
)

arrow(ax, 0.140, 0.777, 0.182, 0.777)
arrow(ax, 0.307, 0.777, 0.352, 0.777)
arrow(ax, 0.482, 0.777, 0.525, 0.777)
arrow(ax, 0.585, 0.777, 0.658, 0.777)
arrow(ax, 0.708, 0.777, 0.757, 0.777)
arrow(ax, 0.837, 0.777, 0.866, 0.777)

# ============================================================
# 2. THREE PRIMARY OBSERVATIONAL QUESTIONS
# ============================================================

section_label(
    ax, 0.080, 0.650,
    2, "Observational reliability analyses", GREEN
)

# Branch 1
rounded_box(
    ax, 0.055, 0.405, 0.270, 0.205,
    facecolor=LIGHT_GREEN,
    edgecolor="#9ED8CA",
    lw=1.2,
)

txt(
    ax, 0.190, 0.577,
    "A. Case-level sensitivity",
    size=11.5,
    weight="bold",
    color=NAVY,
)

txt(
    ax, 0.190, 0.535,
    "Breast97  •  Brain400  •  Brain600",
    size=9.7,
    weight="bold",
)

txt(
    ax, 0.190, 0.477,
    "Multiple descriptions of the same target\n"
    "→ within-case performance distribution\n\n"
    "Quantify:\n"
    "• within-case sensitivity\n"
    "• severe-failure tails\n"
    "• retrospective recoverability",
    size=9.3,
    ha="center",
    va="center",
)

# Branch 2
rounded_box(
    ax, 0.365, 0.405, 0.270, 0.205,
    facecolor=LIGHT_ORANGE,
    edgecolor="#EAC88C",
    lw=1.2,
)

txt(
    ax, 0.500, 0.577,
    "B. Failure localization",
    size=11.5,
    weight="bold",
    color=NAVY,
)

txt(
    ax, 0.500, 0.535,
    "Brain400 description pairs",
    size=9.7,
    weight="bold",
)

txt(
    ax, 0.500, 0.477,
    "Compare pairs spanning segmentation\n"
    "failure ↔ success\n\n"
    "Trace divergence through:\n"
    "• localization-box overlap\n"
    "• preserved-localization residuals\n"
    "• downstream SAM candidates",
    size=9.3,
    ha="center",
    va="center",
)

# Branch 3
rounded_box(
    ax, 0.675, 0.405, 0.270, 0.205,
    facecolor=LIGHT_PURPLE,
    edgecolor="#C8B8EB",
    lw=1.2,
)

txt(
    ax, 0.810, 0.577,
    "C. Ground-truth-free risk",
    size=11.5,
    weight="bold",
    color=NAVY,
)

txt(
    ax, 0.810, 0.535,
    "Localization disagreement",
    size=9.7,
    weight="bold",
)

txt(
    ax, 0.810, 0.477,
    "Development-only signal/threshold selection\n"
    "→ freeze method\n"
    "→ patient-disjoint Brain400 evaluation\n"
    "→ transfer to Brain600\n\n"
    "Evaluate discrimination and\n"
    "selective reliability",
    size=9.3,
    ha="center",
    va="center",
)

# arrows from core pathway down
arrow(
    ax, 0.555, 0.694, 0.190, 0.613,
    color=GREEN,
    lw=1.3,
    connectionstyle="arc3,rad=0.10",
)
arrow(
    ax, 0.685, 0.694, 0.500, 0.613,
    color=ORANGE,
    lw=1.3,
    connectionstyle="arc3,rad=0.06",
)
arrow(
    ax, 0.555, 0.694, 0.810, 0.613,
    color=PURPLE,
    lw=1.3,
    connectionstyle="arc3,rad=-0.08",
)

# ============================================================
# 3. CONTROLLED INTERVENTION
# ============================================================

rounded_box(
    ax, 0.055, 0.115, 0.585, 0.235,
    facecolor=LIGHT_RED,
    edgecolor="#E4AAAA",
    lw=1.25,
)

section_label(
    ax, 0.080, 0.320,
    3, "Development-only representation intervention", RED
)

txt(
    ax, 0.105, 0.274,
    "12 development Brain cases",
    size=9.8,
    weight="bold",
    ha="left",
)

txt(
    ax, 0.105, 0.241,
    "Projected text representations",
    size=9.2,
    color=GRAY,
    ha="left",
)

# Endpoint A
rounded_box(
    ax, 0.105, 0.167, 0.070, 0.045,
    facecolor=WHITE,
    edgecolor=RED,
    lw=1.0,
)
txt(ax, 0.140, 0.1895, "A", size=10.5, weight="bold")

# dose circles
dose_x = [0.220, 0.280, 0.340, 0.400, 0.460]
dose_labels = ["0", "0.25", "0.50", "0.75", "1"]

for i, (x, lab) in enumerate(zip(dose_x, dose_labels)):
    frac = i / 4
    # Keep vector styling simple and deterministic.
    face = "#F8DADA" if i < 4 else "#F1BBBB"
    c = Circle(
        (x, 0.190),
        0.017,
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=RED,
        linewidth=1.0,
        zorder=7,
    )
    ax.add_patch(c)
    txt(ax, x, 0.148, rf"$\alpha={lab}$", size=8.5)

arrow(ax, 0.177, 0.190, 0.201, 0.190, color=RED, lw=1.2)
for x1, x2 in zip(dose_x[:-1], dose_x[1:]):
    arrow(ax, x1 + 0.019, 0.190, x2 - 0.019, 0.190,
          color=RED, lw=1.0)

# Endpoint B
rounded_box(
    ax, 0.505, 0.167, 0.070, 0.045,
    facecolor=WHITE,
    edgecolor=RED,
    lw=1.0,
)
txt(ax, 0.540, 0.1895, "B", size=10.5, weight="bold")
arrow(ax, 0.478, 0.190, 0.503, 0.190, color=RED, lw=1.2)

txt(
    ax, 0.345, 0.125,
    "Prespecified doses → localization response",
    size=9.2,
    weight="bold",
)

# ============================================================
# 4. BOUNDARY CONDITION
# ============================================================

rounded_box(
    ax, 0.675, 0.115, 0.270, 0.235,
    facecolor=LIGHT_GRAY,
    edgecolor="#C4C9CE",
    lw=1.25,
)

section_label(
    ax, 0.700, 0.320,
    4, "Cross-system boundary condition", GRAY
)

txt(
    ax, 0.810, 0.268,
    "Text3DSAM",
    size=11.2,
    weight="bold",
    color=NAVY,
)

txt(
    ax, 0.810, 0.220,
    "Liver segmentation under\n"
    "semantically equivalent descriptions",
    size=9.6,
)

txt(
    ax, 0.810, 0.160,
    "Tests whether pronounced language\n"
    "sensitivity is universal across\n"
    "text-guided segmentation systems",
    size=9.2,
    color=GRAY,
)

# ============================================================
# FOOTER
# ============================================================

txt(
    ax, 0.500, 0.060,
    "Observational analyses characterize where language-conditioned variation occurs; "
    "the intervention separately tests whether controlled movement of the projected "
    "text representation produces ordered movement of localization.",
    size=9.2,
    color=GRAY,
)

# ============================================================
# SAVE
# ============================================================

fig.savefig(
    f"{STEM}.pdf",
    bbox_inches="tight",
    pad_inches=0.10,
)

fig.savefig(
    f"{STEM}.svg",
    bbox_inches="tight",
    pad_inches=0.10,
)

fig.savefig(
    f"{STEM}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.10,
)

plt.close(fig)

print("=== METHOD WORKFLOW FIGURE CREATED ===")
for ext in ("pdf", "svg", "png"):
    p = Path(f"{STEM}.{ext}")
    print(f"{p}  {p.stat().st_size} bytes")
