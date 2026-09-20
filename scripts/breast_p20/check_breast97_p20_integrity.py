import csv
import numpy as np

OLD = "part2/breast98_results/breast98_results.csv"
NEW = "part2/breast97_p20/breast97_p20.csv"

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

old = load(OLD)
new = load(NEW)

assert len(old) == 97, len(old)
assert len(new) == 97, len(new)

old_by_id = {r["filename"]: r for r in old}
new_by_id = {r["filename"]: r for r in new}

assert set(old_by_id) == set(new_by_id)

diffs = []

for fn in sorted(old_by_id):
    a1 = float(old_by_id[fn]["A1_dice"])
    p01 = float(new_by_id[fn]["P01_dice"])
    diffs.append((abs(a1-p01), fn, a1, p01))

diffs.sort(reverse=True)

arr = np.array([x[0] for x in diffs])

print("N =", len(arr))
print("max |P01 - frozen A1| =", arr.max())
print("mean |P01 - frozen A1| =", arr.mean())
print("median |P01 - frozen A1| =", np.median(arr))
print("exact matches =", int((arr == 0).sum()), "/", len(arr))

print("\nLargest 10 differences:")
for d, fn, a1, p01 in diffs[:10]:
    print(
        f"{fn:28s} "
        f"A1={a1:.12f} "
        f"P01={p01:.12f} "
        f"diff={d:.12g}"
    )

# Strict reproducibility requirement.
# Tiny floating noise is acceptable; substantive differences are not.
if arr.max() <= 1e-6:
    print("\nINTEGRITY CHECK: PASS")
else:
    print("\nINTEGRITY CHECK: INVESTIGATE")
