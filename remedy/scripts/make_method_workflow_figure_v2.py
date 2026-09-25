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

STEM = OUT / "fig_method_workflow_v2"

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK = "#1F2933"
MUTED = "#66717D"
LINE = "#9AA6B2"
PALE = "#F7F9FA"

BLUE = "#2474A6"
BLUE_BG = "#EEF6FB"

GREEN = "#238A78"
GREEN_BG = "#EDF8F5"

ORANGE = "#C87516"
ORANGE_BG = "#FFF6E9"

PURPLE = "#7054B8"
PURPLE_BG = "#F5F1FB"

RED = "#B94A48"
RED_BG = "#FCF1F1"

WHITE = "#FFFFFF"


# ============================================================
# HELPERS
# ============================================================

def box(ax, x, y, w, h, fc=WHITE, ec=LINE, lw=1.1, r=0.012):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=10, weight="normal",
         color=INK, ha="center", va="center"):
    ax.text(
        x, y, s,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=1.18,
        zorder=10,
    )


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.35):
    p = FancyArrowPatch(
        (x1, y1), (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=5,
    )
    ax.add_patch(p)
    return p


def panel_letter(ax, x, y, letter):
    text(ax, x, y, letter, size=18, weight="bold", ha="left")


def separator(ax, y):
    ax.plot(
        [0.045, 0.955], [y, y],
        transform=ax.transAxes,
        color="#D6DCE1",
        lw=0.9,
        zorder=1,
    )


def image_symbol(ax, x, y):
    r = Rectangle(
        (x - 0.025, y - 0.030),
        0.050, 0.060,
        transform=ax.transAxes,
        facecolor=PALE,
        edgecolor=LINE,
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(r)

    c1 = Circle(
        (x - 0.008, y + 0.006), 0.011,
        transform=ax.transAxes,
        facecolor="#CBD3D9",
        edgecolor="none",
        zorder=5,
    )
    c2 = Circle(
        (x + 0.010, y - 0.008), 0.008,
        transform=ax.transAxes,
        facecolor="#AEB8C1",
        edgecolor="none",
        zorder=5,
    )
    ax.add_patch(c1)
    ax.add_patch(c2)


def heatmap_symbol(ax, x, y):
    for radius, color in [
        (0.026, "#FBE4B8"),
        (0.019, "#F5BF68"),
        (0.012, "#E58B32"),
        (0.006, "#C94F4F"),
    ]:
        ax.add_patch(
            Circle(
                (x, y), radius,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )


def bbox_symbol(ax, x, y):
    ax.add_patch(
        Rectangle(
            (x - 0.025, y - 0.020),
            0.050, 0.040,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=ORANGE,
            linewidth=2.0,
            zorder=5,
        )
    )


def mask_symbol(ax, x, y):
    ax.add_patch(
        Circle(
            (x - 0.006, y),
            0.018,
            transform=ax.transAxes,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.90,
            zorder=5,
        )
    )
    ax.add_patch(
        Circle(
            (x + 0.010, y + 0.005),
            0.013,
            transform=ax.transAxes,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.90,
            zorder=5,
        )
    )


# ============================================================
# CANVAS
# ============================================================

fig, ax = plt.subplots(figsize=(14.0, 9.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ============================================================
# PANEL A — COMPUTATIONAL PATHWAY
# ============================================================

panel_letter(ax, 0.045, 0.955, "A")
text(
    ax, 0.078, 0.955,
    "Language-conditioned inference pathway",
    size=13.5, weight="bold", ha="left"
)

# Elements aligned on ONE horizontal baseline.
Y = 0.845

image_symbol(ax, 0.100, Y)
text(ax, 0.100, 0.792, "Medical image", size=9.5, weight="bold")

box(ax, 0.170, 0.815, 0.120, 0.060, fc=BLUE_BG, ec=BLUE)
text(ax, 0.230, Y, "Target\ndescription", size=9.6, weight="bold")

box(ax, 0.340, 0.815, 0.135, 0.060, fc=BLUE_BG, ec=BLUE)
text(ax, 0.4075, Y, "Projected text\nrepresentation",
     size=9.3, weight="bold")

heatmap_symbol(ax, 0.555, Y)
text(ax, 0.555, 0.792, "Localization", size=9.5, weight="bold")

bbox_symbol(ax, 0.675, Y)
text(ax, 0.675, 0.792, "Bounding box", size=9.5, weight="bold")

box(ax, 0.745, 0.815, 0.075, 0.060, fc=ORANGE_BG, ec=ORANGE)
text(ax, 0.7825, Y, "SAM", size=10.2, weight="bold")

mask_symbol(ax, 0.900, Y)
text(ax, 0.900, 0.792, "Mask", size=9.5, weight="bold")

arrow(ax, 0.127, Y, 0.167, Y)
arrow(ax, 0.293, Y, 0.337, Y)
arrow(ax, 0.478, Y, 0.523, Y)
arrow(ax, 0.587, Y, 0.645, Y)
arrow(ax, 0.704, Y, 0.742, Y)
arrow(ax, 0.823, Y, 0.868, Y)

separator(ax, 0.745)

# ============================================================
# PANEL B — OBSERVATIONAL STUDY
# ============================================================

panel_letter(ax, 0.045, 0.705, "B")
text(
    ax, 0.078, 0.705,
    "Observational reliability characterization",
    size=13.5, weight="bold", ha="left"
)

# ---- B1 -----------------------------------------------------

box(
    ax, 0.055, 0.420, 0.275, 0.235,
    fc=GREEN_BG, ec="#82C6B9"
)

text(
    ax, 0.1925, 0.620,
    "Case-level sensitivity",
    size=11.5, weight="bold", color=GREEN
)

text(
    ax, 0.1925, 0.574,
    "Breast97   •   Brain400   •   Brain600",
    size=9.5, weight="bold"
)

text(
    ax, 0.1925, 0.515,
    "Multiple descriptions\n"
    "for the same target",
    size=9.7
)

arrow(ax, 0.1925, 0.482, 0.1925, 0.455, color=GREEN)

text(
    ax, 0.1925, 0.438,
    "Within-case sensitivity\n"
    "Severe-failure tail\n"
    "Retrospective recoverability",
    size=9.3, weight="bold"
)

# ---- B2 -----------------------------------------------------

box(
    ax, 0.3625, 0.420, 0.275, 0.235,
    fc=ORANGE_BG, ec="#E3B36E"
)

text(
    ax, 0.500, 0.620,
    "Failure localization",
    size=11.5, weight="bold", color=ORANGE
)

text(
    ax, 0.500, 0.574,
    "Brain400",
    size=9.5, weight="bold"
)

text(
    ax, 0.500, 0.526,
    "Segmentation\nfailure  ↔  success",
    size=9.7
)

arrow(ax, 0.500, 0.486, 0.500, 0.459, color=ORANGE)

text(
    ax, 0.500, 0.438,
    "Localization-box overlap\n"
    "Preserved-localization residuals\n"
    "Downstream SAM candidates",
    size=9.3, weight="bold"
)

# ---- B3 -----------------------------------------------------

box(
    ax, 0.670, 0.420, 0.275, 0.235,
    fc=PURPLE_BG, ec="#B8A5E3"
)

text(
    ax, 0.8075, 0.620,
    "Ground-truth-free risk",
    size=11.5, weight="bold", color=PURPLE
)

text(
    ax, 0.8075, 0.574,
    "Localization disagreement",
    size=9.5, weight="bold"
)

# Vertical process — no prose collision.
text(ax, 0.8075, 0.532, "Development", size=9.3)
arrow(ax, 0.8075, 0.515, 0.8075, 0.493, color=PURPLE)

text(ax, 0.8075, 0.477, "Freeze", size=9.3, weight="bold")
arrow(ax, 0.8075, 0.461, 0.8075, 0.442, color=PURPLE)

text(
    ax, 0.8075, 0.427,
    "Brain400 held-out  →  Brain600 transfer",
    size=8.8, weight="bold"
)

separator(ax, 0.375)

# ============================================================
# PANEL C — CONTROLLED TEST + BOUNDARY CONDITION
# ============================================================

panel_letter(ax, 0.045, 0.335, "C")
text(
    ax, 0.078, 0.335,
    "Controlled mechanism test and cross-system boundary condition",
    size=13.5, weight="bold", ha="left"
)

# ---- Mechanism ---------------------------------------------

box(
    ax, 0.055, 0.075, 0.585, 0.215,
    fc=RED_BG, ec="#DEA09F"
)

text(
    ax, 0.085, 0.258,
    "Representation intervention",
    size=11.3, weight="bold", color=RED, ha="left"
)

text(
    ax, 0.085, 0.222,
    "12 development Brain cases",
    size=9.3, weight="bold", ha="left"
)

# A -> doses -> B
box(ax, 0.100, 0.140, 0.065, 0.045, fc=WHITE, ec=RED)
text(ax, 0.1325, 0.1625, "A", size=10.5, weight="bold")

xs = [0.225, 0.295, 0.365, 0.435, 0.505]
labs = ["0", "0.25", "0.50", "0.75", "1"]

arrow(ax, 0.168, 0.1625, 0.202, 0.1625, color=RED)

for i, (x, lab) in enumerate(zip(xs, labs)):
    ax.add_patch(
        Circle(
            (x, 0.1625), 0.015,
            transform=ax.transAxes,
            facecolor="#F8DCDC" if i < 4 else "#F0BDBC",
            edgecolor=RED,
            linewidth=1.0,
            zorder=5,
        )
    )
    text(ax, x, 0.119, rf"$\alpha={lab}$", size=8.2)

for x1, x2 in zip(xs[:-1], xs[1:]):
    arrow(
        ax, x1 + 0.017, 0.1625,
        x2 - 0.017, 0.1625,
        color=RED, lw=1.05
    )

box(ax, 0.555, 0.140, 0.055, 0.045, fc=WHITE, ec=RED)
text(ax, 0.5825, 0.1625, "B", size=10.5, weight="bold")
arrow(ax, 0.522, 0.1625, 0.552, 0.1625, color=RED)

text(
    ax, 0.355, 0.093,
    "Prespecified interpolation doses  →  localization response",
    size=9.0, weight="bold"
)

# ---- Boundary condition ------------------------------------

box(
    ax, 0.675, 0.075, 0.270, 0.215,
    fc=PALE, ec="#BFC7CE"
)

text(
    ax, 0.705, 0.258,
    "Boundary condition",
    size=11.3, weight="bold", color=MUTED, ha="left"
)

text(
    ax, 0.810, 0.215,
    "Text3DSAM",
    size=10.7, weight="bold"
)

text(
    ax, 0.810, 0.169,
    "Liver segmentation",
    size=9.4
)

arrow(ax, 0.810, 0.148, 0.810, 0.127, color=MUTED)

text(
    ax, 0.810, 0.103,
    "Semantically equivalent descriptions",
    size=8.9, weight="bold"
)

# ============================================================
# SAVE
# ============================================================

fig.savefig(
    f"{STEM}.pdf",
    bbox_inches="tight",
    pad_inches=0.08,
)

fig.savefig(
    f"{STEM}.svg",
    bbox_inches="tight",
    pad_inches=0.08,
)

fig.savefig(
    f"{STEM}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08,
)

plt.close(fig)

print("=== METHOD WORKFLOW V2 CREATED ===")
for ext in ("pdf", "svg", "png"):
    p = Path(f"{STEM}.{ext}")
    print(f"{p}  {p.stat().st_size} bytes")
