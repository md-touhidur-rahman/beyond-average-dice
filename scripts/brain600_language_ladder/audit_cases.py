import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "MedCLIP-SAMv2/parent_repro_brain600"
RES = ROOT / "beyond-average-dice/results/brain600_language_ladder"
PROMPT_DIR = BASE / "language_ladder_prompts"

CASES = ["457", "762", "2816", "1401", "2982"]
CONDS = ["H0","H1","H2","L3","L4","L5"]

# ---------------------------------------------------------
# Load frozen case metrics
# ---------------------------------------------------------

df = pd.read_csv(RES / "case_metrics.csv")
df["case"] = df["case"].astype(str)

# ---------------------------------------------------------
# Prompt JSON discovery
# Print exact files used so nothing is silently assumed.
# ---------------------------------------------------------

def find_jsons(root):
    return sorted(root.rglob("*.json"))

all_json = find_jsons(BASE)

print("\n==========================================")
print("AVAILABLE PROMPT JSON FILES")
print("==========================================")
for p in all_json:
    print(p.relative_to(ROOT))

# Candidate mapping based on known frozen protocol.
# If a path does not exist, abort rather than guess.

candidates = {
    "H0": [
        BASE / "prompts.json",
        BASE / "brain_prompts.json",
        BASE / "descriptions.json",
    ],
    "H1": [
        BASE / "highop/H1_verb/prompts.json",
        BASE / "highop/H1_verb/brain_prompts.json",
        BASE / "highop/H1_verb/descriptions.json",
    ],
    "H2": [
        BASE / "highop/H2_intro/prompts.json",
        BASE / "highop/H2_intro/brain_prompts.json",
        BASE / "highop/H2_intro/descriptions.json",
    ],
    "L3": [
        PROMPT_DIR / "L3_semantic.json",
        PROMPT_DIR / "L3.json",
    ],
    "L4": [
        PROMPT_DIR / "L4_concise.json",
        PROMPT_DIR / "L4.json",
    ],
    "L5": [
        PROMPT_DIR / "L5_broad.json",
        PROMPT_DIR / "L5.json",
    ],
}

def resolve(cond):
    existing = [p for p in candidates[cond] if p.exists()]

    if len(existing) == 1:
        return existing[0]

    # Try discovery by condition token, but NEVER silently choose
    token = cond.lower()
    matches = [
        p for p in all_json
        if token in str(p).lower()
    ]

    if len(matches) == 1:
        return matches[0]

    print(f"\nERROR resolving {cond}")
    print("Candidate existing:", existing)
    print("Token matches:", matches)
    raise RuntimeError(
        f"Could not uniquely resolve prompt JSON for {cond}. "
        "Do not guess; inspect paths above."
    )

prompt_paths = {c: resolve(c) for c in CONDS}

print("\n==========================================")
print("RESOLVED PROMPT FILES")
print("==========================================")
for c,p in prompt_paths.items():
    print(c, "->", p.relative_to(ROOT))

# ---------------------------------------------------------
# Robust prompt JSON extraction
# ---------------------------------------------------------

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

prompt_data = {c: load_json(p) for c,p in prompt_paths.items()}

def lookup_prompt(obj, case):
    """
    Supports dictionaries keyed by stem/filename and simple lists.
    Aborts if exact matching cannot be established.
    """
    if isinstance(obj, dict):
        # direct stem
        if case in obj:
            v = obj[case]
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                for k in ["prompt","text","description","caption"]:
                    if k in v:
                        return str(v[k])

        # filename-like keys
        hits = []
        for k,v in obj.items():
            if Path(str(k)).stem == case:
                hits.append(v)

        if len(hits) == 1:
            v = hits[0]
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                for k in ["prompt","text","description","caption"]:
                    if k in v:
                        return str(v[k])

    if isinstance(obj, list):
        hits = []
        for item in obj:
            if not isinstance(item, dict):
                continue

            identifiers = []
            for k in ["case","id","image","filename","file","image_path"]:
                if k in item:
                    identifiers.append(Path(str(item[k])).stem)

            if case in identifiers:
                hits.append(item)

        if len(hits) == 1:
            for k in ["prompt","text","description","caption"]:
                if k in hits[0]:
                    return str(hits[0][k])

    raise RuntimeError(f"Cannot uniquely find prompt for case={case}")

# ---------------------------------------------------------
# Box IoU from coarse masks
# ---------------------------------------------------------

coarse_dirs = {
    "H0": BASE / "coarse",
    "H1": BASE / "highop/H1_verb/coarse",
    "H2": BASE / "highop/H2_intro/coarse",
    "L3": BASE / "language_ladder/L3_semantic/coarse",
    "L4": BASE / "language_ladder/L4_concise/coarse",
    "L5": BASE / "language_ladder/L5_broad/coarse",
}

def file_map(d):
    return {
        p.stem: p for p in d.iterdir()
        if p.is_file() and not p.name.startswith(".")
    }

coarse_maps = {c:file_map(d) for c,d in coarse_dirs.items()}

def bbox(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise RuntimeError(f"Cannot read {path}")
    if x.ndim == 3:
        x = x[...,0]
    ys,xs = np.where(x > 0)
    if len(xs) == 0:
        return None
    return (
        int(xs.min()), int(ys.min()),
        int(xs.max()), int(ys.max())
    )

def box_iou(a,b):
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b

    ix1=max(ax1,bx1)
    iy1=max(ay1,by1)
    ix2=min(ax2,bx2)
    iy2=min(ay2,by2)

    iw=max(0,ix2-ix1+1)
    ih=max(0,iy2-iy1+1)

    inter=iw*ih
    aa=(ax2-ax1+1)*(ay2-ay1+1)
    bb=(bx2-bx1+1)*(by2-by1+1)

    return inter/(aa+bb-inter) if aa+bb-inter else 0.0

# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------

rows=[]

print("\n==========================================")
print("CASE AUDIT")
print("==========================================")

for case in CASES:
    r = df[df["case"] == case]

    if len(r) != 1:
        raise RuntimeError(
            f"Expected exactly one metrics row for {case}, got {len(r)}"
        )

    r = r.iloc[0]

    h0_box = bbox(coarse_maps["H0"][case])

    print("\n------------------------------------------")
    print("CASE", case)
    print("------------------------------------------")

    for cond in CONDS:
        prompt = lookup_prompt(prompt_data[cond], case)
        b = bbox(coarse_maps[cond][case])
        iou = box_iou(h0_box,b)
        dice = float(r[f"dice_{cond}"])

        print(f"\n{cond}")
        print("Dice:", f"{dice:.6f}")
        print("Box:", b)
        print("Box IoU vs H0:", f"{iou:.6f}")
        print("Prompt:", prompt)

        rows.append({
            "case":case,
            "condition":cond,
            "dice":dice,
            "box":str(b),
            "box_iou_vs_H0":iou,
            "prompt":prompt,
        })

audit = pd.DataFrame(rows)

out = RES / "representative_case_audit.csv"
audit.to_csv(out,index=False)

print("\n==========================================")
print("WROTE")
print("==========================================")
print(out)
