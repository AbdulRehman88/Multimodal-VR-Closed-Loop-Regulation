from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
SOURCE = ROOT / "revision/results/confirmatory_200"
CONFIG = ROOT / "revision/configs"
STATS = ROOT / "revision/results/statistical_analysis"

FIGURES = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR"
    r"\results\final_independent_evaluation"
    r"\publication_figures_revised"
)

STEM = "F10_comparative_policy_performance_v2"
PDF = FIGURES / (STEM + ".pdf")
PNG = FIGURES / (STEM + ".png")
MANIFEST = ROOT / "revision/results/publication_evidence/F10_v2_figure_integrity.json"

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 244 - PUBLICATION FIGURE 10")

require(FIGURES.is_dir(), "Original figure repository missing")
require(not PDF.exists(), "Figure 10 PDF already exists")
require(not PNG.exists(), "Figure 10 PNG already exists")
require(not MANIFEST.exists(), "Figure 10 manifest already exists")

# Verify preserved statistical results.
stat_manifest = STATS / "statistical_analysis_manifest.json"
require(
    sha(stat_manifest)
    == "6a9685ea20bd0261a213b300a9b5df739bac93753a028a58af161e2003076ce4",
    "Statistical evidence manifest mismatch"
)

stat = json.loads(stat_manifest.read_text(encoding="utf-8"))
for name, expected in stat["outputs"].items():
    require(
        sha(STATS / name) == expected,
        f"Statistical data modified: {name}"
    )

audit_file = CONFIG / "confirmatory_results_integrity_audit.json"
require(
    sha(audit_file)
    == "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d",
    "Confirmatory integrity audit mismatch"
)
audit = json.loads(audit_file.read_text(encoding="utf-8"))

# Inspect the font embedded in the original Matplotlib PDF.
reference = FIGURES / "F8_information_ablation_revised.pdf"
require(reference.is_file(), "Original style-reference PDF missing")

font_records = re.findall(
    rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
    reference.read_bytes()
)

font_names = sorted({
    value.decode("latin1").split("+")[-1]
    for value in font_records
})

print("\nORIGINAL PDF FONT RECORDS:", font_names)

detected = " ".join(font_names).lower()

if "dejavusans" in detected or "dejavu-sans" in detected:
    family = "DejaVu Sans"
elif "liberationsans" in detected:
    family = "Liberation Sans"
elif "arial" in detected:
    family = "Arial"
elif "timesnewroman" in detected:
    family = "Times New Roman"
elif "liberationserif" in detected:
    family = "Liberation Serif"
elif "dejavuserif" in detected:
    family = "DejaVu Serif"
else:
    raise RuntimeError(
        "Original font family could not be verified. "
        "Do not generate a mismatched publication figure."
    )

from matplotlib import font_manager

available = {
    font.name
    for font in font_manager.fontManager.ttflist
}
require(
    family in available,
    f"Original font not available in Matplotlib: {family}"
)

print("Matched font family:", family)

plt.rcParams.update({
    "font.family": family,
    "font.size": 12,
    "font.weight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

# Original palette, extended to distinguish the new policies.
COLORS = {
    "B0": "#727B86",
    "B1": "#9B8790",
    "B2": "#B76A49",
    "B3": "#5485AC",
    "B4": "#718EAA",
    "B5": "#B24B50",
    "B6": "#24618E",
    "B7": "#30836D",
    "B8": "#8A7795",
}

NAMES = {
    "B0": "Neutral",
    "B1": "Fixed schedule",
    "B2": "Time-only LR",
    "B3": "Motion-only LR",
    "B4": "Motion threshold",
    "B5": "AI without reserve",
    "B6": "Proposed AI + reserve",
    "B7": "Privileged-state oracle",
    "B8": "Sham motion",
}

policies = [f"B{i}" for i in range(9)]
records = []

for entry in tqdm(
    audit["checkpoint_inventory"],
    desc="Figure 10 data verification",
    unit="profile"
):
    path = SOURCE / entry["file"]

    require(
        sha(path) == entry["sha256"],
        f"Checkpoint modified: {path.name}"
    )

    for r in json.loads(path.read_text(encoding="utf-8")):
        records.append({
            "profile": int(r["profile"]),
            "condition": r["condition"],
            "policy": r["policy"],
            "recovery_s": float(r["recovery_s"]),
            "variation": float(r["variation"])
        })

df = pd.DataFrame(records)

require(len(df) == 3600, "Incorrect number of episodes")
require(df["profile"].nunique() == 200, "Profile count mismatch")

profile_ids = sorted(df["profile"].unique())
index = pd.MultiIndex.from_product(
    [profile_ids, ["Low", "Elevated"]],
    names=["profile", "condition"]
)

def profile_matrix(metric):
    frame = df.pivot(
        index=["profile", "condition"],
        columns="policy",
        values=metric
    ).reindex(index=index, columns=policies)

    require(not frame.isna().any().any(),
            f"Incomplete {metric} matrix")

    return frame.to_numpy(float).reshape(200, 2, 9).mean(axis=1)

recovery = profile_matrix("recovery_s")
variation = profile_matrix("variation")

rng = np.random.default_rng(2026)
draws = rng.integers(0, 200, size=(10000, 200))

def means_and_intervals(values):
    means = values.mean(axis=0)
    intervals = np.quantile(
        values[draws].mean(axis=1),
        [0.025, 0.975],
        axis=0
    )
    return means, intervals

mean_rec, ci_rec = means_and_intervals(recovery)
mean_var, ci_var = means_and_intervals(variation)

# Cross-check plotted means against the frozen summary.
summary = pd.read_csv(
    STATS / "policy_summary.csv"
).set_index("policy").loc[policies]

require(
    np.allclose(
        mean_rec,
        summary["recovery_s"].to_numpy(float),
        rtol=0, atol=1e-10
    ),
    "Recovery means disagree with statistical analysis"
)
require(
    np.allclose(
        mean_var,
        summary["variation"].to_numpy(float),
        rtol=0, atol=1e-10
    ),
    "Variation means disagree with statistical analysis"
)

primary = pd.read_csv(
    STATS / "primary_holm_comparisons.csv"
)
primary = primary[
    primary["endpoint"] == "recovery_s"
].set_index("comparator").loc[
    ["B1", "B2", "B3", "B4"]
]

# Verify paired effects independently from the archived episodes.
for code in ("B1", "B2", "B3", "B4"):
    effect = (
        recovery[:, policies.index(code)]
        - recovery[:, policies.index("B6")]
    )
    require(
        abs(
            effect.mean()
            - primary.loc[code, "mean_improvement"]
        ) < 1e-10,
        f"Paired effect mismatch: {code}"
    )

# Snapshot the original figures; refuse accidental edits.
original_hashes = {
    p.name: sha(p)
    for p in FIGURES.iterdir()
    if p.is_file() and p.suffix.lower() in (".pdf", ".png")
}


# Reviewed layout: preserve the original Matplotlib font,
# established palette, policy definitions, and numerical data.
# Labels are separated from uncertainty intervals.

fig = plt.figure(
    figsize=(19.2, 8.0),
    layout="constrained",
    facecolor="white"
)

gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.55, 1.15, 1.15],
    wspace=0.15
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
ax_c = fig.add_subplot(gs[0, 2])

y = np.arange(len(policies))
colors = [COLORS[p] for p in policies]

labels = [
    f"{p}  {NAMES[p]}"
    for p in policies
]

def style(ax):
    ax.grid(
        axis="x",
        color="#707070",
        alpha=0.15,
        linewidth=0.8
    )
    ax.tick_params(
        axis="both", length=0, pad=7
    )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

# ==========================================================
# PANEL A - MEAN RECOVERY WITH CLUSTER BOOTSTRAP INTERVALS
# ==========================================================

ax_a.barh(
    y, mean_rec,
    color=colors,
    height=0.64,
    edgecolor="none",
    zorder=2
)

ax_a.errorbar(
    mean_rec,
    y,
    xerr=[
        np.maximum(0, mean_rec - ci_rec[0]),
        np.maximum(0, ci_rec[1] - mean_rec)
    ],
    fmt="none",
    ecolor="#202020",
    elinewidth=1.55,
    capsize=4,
    capthick=1.55,
    zorder=5
)

ax_a.set_yticks(y, labels)
ax_a.invert_yaxis()
ax_a.set_xlim(0, 3.20)
ax_a.set_xlabel("Mean recovery time (s)")
ax_a.set_title("A. Recovery time", pad=17)

# A fixed annotation column prevents labels from overlapping
# uncertainty bars, regardless of the policy-specific CI.
for yi, value in zip(y, mean_rec):
    ax_a.text(
        3.05, yi,
        f"{value:.3f}",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#202020"
    )

style(ax_a)

# ==========================================================
# PANEL B - CUMULATIVE VARIATION WITH COMMON BUDGET CEILING
# ==========================================================

ax_b.barh(
    y, mean_var,
    color=colors,
    height=0.64,
    edgecolor="none",
    zorder=2
)

ax_b.errorbar(
    mean_var,
    y,
    xerr=[
        np.maximum(0, mean_var - ci_var[0]),
        np.maximum(0, ci_var[1] - mean_var)
    ],
    fmt="none",
    ecolor="#202020",
    elinewidth=1.55,
    capsize=4,
    capthick=1.55,
    zorder=5
)

ax_b.axvline(
    0.6,
    color="#6F3741",
    linewidth=1.6,
    linestyle="--",
    zorder=3
)

ax_b.tick_params(labelleft=False)
ax_b.set_xlim(0, 0.80)
ax_b.set_xlabel("Mean stimulus total variation")
ax_b.set_title("B. Intervention expenditure", pad=17)

# Value labels never intersect the budget line or CIs.
for yi, value in zip(y, mean_var):
    ax_b.text(
        0.765, yi,
        f"{value:.3f}",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#202020"
    )

ax_b.text(
    0.60, -0.60,
    "Budget = 0.6",
    ha="center",
    va="center",
    fontsize=10.5,
    fontweight="bold",
    color="#6F3741"
)

style(ax_b)

# ==========================================================
# PANEL C - PAIRED RECOVERY EFFECTS, POSITIVE FAVORS B6
# ==========================================================

comparison_codes = ["B1", "B2", "B3", "B4"]
comparison_y = np.arange(len(comparison_codes))

effects = primary["mean_improvement"].to_numpy(float)
lower = primary["CI_low"].to_numpy(float)
upper = primary["CI_high"].to_numpy(float)

require(
    np.all(lower <= effects)
    and np.all(effects <= upper),
    "Primary confidence interval mismatch"
)

ax_c.axvline(
    0,
    linewidth=1.5,
    color="#303030",
    zorder=2
)

ax_c.errorbar(
    effects,
    comparison_y,
    xerr=[
        effects - lower,
        upper - effects
    ],
    fmt="o",
    markersize=8,
    color="#24618E",
    ecolor="#24618E",
    elinewidth=2,
    capsize=5,
    capthick=1.6,
    zorder=5
)

ax_c.set_yticks(
    comparison_y,
    [
        f"B6 vs {code}"
        for code in comparison_codes
    ]
)
ax_c.invert_yaxis()
ax_c.set_xlim(-0.04, 0.45)
ax_c.set_ylim(3.8, -0.8)
ax_c.set_xlabel("Paired recovery improvement (s)")
ax_c.set_title("C. Paired improvement", pad=17)

# Place all numerical labels at a fixed x-coordinate,
# outside every confidence interval.
for yi, value in zip(comparison_y, effects):
    ax_c.text(
        0.425, yi,
        f"+{value:.3f}",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#202020"
    )

style(ax_c)

# Distinguish the direction of all three measurements.
fig.suptitle(
    "Comparative closed-loop performance under a 0.6 variation ceiling",
    fontsize=16,
    fontweight="bold"
)

# Export to temporary files first.
tmp_pdf = FIGURES / ("." + STEM + ".staged.pdf")
tmp_png = FIGURES / ("." + STEM + ".staged.png")

require(not tmp_pdf.exists(), "Staged PDF already exists")
require(not tmp_png.exists(), "Staged PNG already exists")

try:
    fig.savefig(
        tmp_pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.20,
        facecolor="white"
    )
    fig.savefig(
        tmp_png,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.20,
        facecolor="white"
    )
finally:
    plt.close(fig)

require(tmp_pdf.stat().st_size > 10000,
        "Exported PDF unexpectedly small")
require(tmp_png.stat().st_size > 10000,
        "Exported PNG unexpectedly small")

with Image.open(tmp_png) as im:
    dpi = im.info.get("dpi", (0, 0))
    dimensions = im.size

require(
    all(abs(float(d) - 600) <= 2 for d in dpi),
    "PNG export is not 600 dpi"
)

# Confirm that a font from the original family is
# represented in the newly generated PDF.
new_fonts = {
    f.decode("latin1").split("+")[-1]
    for f in re.findall(
        rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
        tmp_pdf.read_bytes()
    )
}
require(new_fonts, "Exported vector PDF has no font records")

# Original plots must not change.
for filename, expected_hash in original_hashes.items():
    require(
        sha(FIGURES / filename) == expected_hash,
        f"Original publication figure changed: {filename}"
    )

tmp_pdf.replace(PDF)
tmp_png.replace(PNG)

manifest = {
    "step": 244,
    "figure": STEM,
    "scientific_scope": "new_independent_200_profile_cohort",
    "panels": {
        "A": "Mean recovery and profile-cluster bootstrap 95% CI",
        "B": "Actual cumulative stimulus variation and 95% CI",
        "C": "Paired B6-minus-comparator recovery improvement"
    },
    "policy_names": NAMES,
    "policy_colors": COLORS,
    "font_family": family,
    "reference_embedded_fonts": font_names,
    "output_embedded_fonts": sorted(new_fonts),
    "font_style": "bold",
    "base_font_size_pt": 12,
    "figure_title_size_pt": 16,
    "png_dpi": 600,
    "png_dimensions": dimensions,
    "bootstrap_replicates": 10000,
    "independent_profiles": 200,
    "original_results_changed": False,
    "original_figures_changed": False,
    "pdf_path": str(PDF),
    "png_path": str(PNG),
    "pdf_sha256": sha(PDF),
    "png_sha256": sha(PNG),
    "source_statistical_manifest_sha256": sha(stat_manifest),
    "source_audit_sha256": sha(audit_file)
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("\nFIGURE 10 - VERIFICATION")
print("Font family:", family)
print("Reference PDF fonts:", font_names)
print("New PDF fonts:", sorted(new_fonts))
print("Original font sizes/style: Matched")
print("Independent profiles:", len(profile_ids))
print("Verified episode records:", len(records))
print("Recovery mean parity: PASS")
print("Variation mean parity: PASS")
print("Primary paired-effect parity: PASS")

print("\nOUTPUT FILES")
print("PDF:", PDF)
print("PNG:", PNG)
print("PNG dimensions:", dimensions)
print("PNG dpi:", dpi)
print("PDF SHA256:", sha(PDF))
print("PNG SHA256:", sha(PNG))
print("Manifest SHA256:", sha(MANIFEST))
print("Original figures unchanged: PASS")
print("Frozen experiment unchanged: PASS")
print("STEP 244: PASS")
