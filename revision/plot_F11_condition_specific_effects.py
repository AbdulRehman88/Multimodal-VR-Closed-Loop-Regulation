from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
CONFIG = ROOT / "revision/configs"
SOURCE = ROOT / "revision/results/confirmatory_200"
FIGURES = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR"
    r"\results\final_independent_evaluation"
    r"\publication_figures_revised"
)
EVIDENCE = ROOT / "revision/results/publication_evidence"

STEM = "F11_condition_specific_policy_effects"
PDF = FIGURES / (STEM + ".pdf")
PNG = FIGURES / (STEM + ".png")
MANIFEST = EVIDENCE / "F11_figure_integrity.json"

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 246 - FIGURE 11: CONDITION-SPECIFIC EFFECTS")

require(FIGURES.is_dir(), "Original figure repository missing")

for path in (PDF, PNG, MANIFEST):
    require(not path.exists(), f"Output already exists: {path}")

audit_file = CONFIG / "confirmatory_results_integrity_audit.json"
require(
    sha(audit_file)
    == "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d",
    "Frozen audit mismatch"
)

audit = json.loads(audit_file.read_text(encoding="utf-8"))
require(audit["episodes"] == 3600, "Episode count mismatch")

reference = FIGURES / "F10_comparative_policy_performance_v2.pdf"
require(
    sha(reference)
    == "e3b1a73ed9140eb374afda4cca53a67566a1e1d0803c2a9604d011ed0cf50586",
    "Approved Figure 10 reference mismatch"
)

font_records = re.findall(
    rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
    reference.read_bytes()
)
font_names = {
    f.decode("latin1").split("+")[-1]
    for f in font_records
}
require(
    "TimesNewRomanPS-BoldMT" in font_names,
    "Original figure font not verified"
)
require(
    "Times New Roman" in {
        f.name for f in font_manager.fontManager.ttflist
    },
    "Times New Roman not installed"
)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

records = []

for entry in tqdm(
    audit["checkpoint_inventory"],
    desc="Verifying Figure 11 inputs",
    unit="profile"
):
    path = SOURCE / entry["file"]
    require(
        sha(path) == entry["sha256"],
        f"Frozen checkpoint modified: {path.name}"
    )

    for r in json.loads(path.read_text(encoding="utf-8")):
        if r["policy"] in ("B1", "B2", "B3", "B4", "B6"):
            records.append({
                "profile": int(r["profile"]),
                "condition": r["condition"],
                "policy": r["policy"],
                "recovery_s": float(r["recovery_s"])
            })

df = pd.DataFrame(records)
require(len(df) == 2000, "Unexpected subset size")
require(df["profile"].nunique() == 200, "Profile count mismatch")
require(
    not df.duplicated(
        ["profile", "condition", "policy"]
    ).any(),
    "Duplicate cases"
)

comparators = ["B1", "B2", "B3", "B4"]
names = [
    "B1  Fixed schedule",
    "B2  Time-only LR",
    "B3  Motion-only LR",
    "B4  Motion threshold"
]
colors = [
    "#9B8790",
    "#B76A49",
    "#5485AC",
    "#718EAA"
]

pivot = df.pivot(
    index=["profile", "condition"],
    columns="policy",
    values="recovery_s"
)

profiles = sorted(df["profile"].unique())
require(len(profiles) == 200, "Wrong cluster count")

rng = np.random.default_rng(2026)
draws = rng.integers(0, 200, size=(10000, 200))

effects = {}
intervals = {}

for code in comparators:
    low_values = (
        pivot.loc[
            pd.IndexSlice[profiles, "Low"], code
        ].to_numpy()
        -
        pivot.loc[
            pd.IndexSlice[profiles, "Low"], "B6"
        ].to_numpy()
    )

    elevated_values = (
        pivot.loc[
            pd.IndexSlice[profiles, "Elevated"], code
        ].to_numpy()
        -
        pivot.loc[
            pd.IndexSlice[profiles, "Elevated"], "B6"
        ].to_numpy()
    )

    require(
        low_values.shape == elevated_values.shape == (200,),
        "Incorrect paired condition vectors"
    )

    for key, values in (
        ("Low", low_values),
        ("Elevated", elevated_values),
        ("Interaction", elevated_values - low_values)
    ):
        mean = float(values.mean())
        ci = np.quantile(
            values[draws].mean(axis=1),
            [0.025, 0.975]
        )

        effects[(key, code)] = mean
        intervals[(key, code)] = ci

# Cross-check against the archived Step 237 report.
endpoint_report = (
    ROOT / "revision/results/endpoint_dependence_audit.json"
)
require(
    sha(endpoint_report)
    == "fe3b814014aa94d4aa62a170b4783970a5c53aebcee52233bd3210b52a609127",
    "Condition interaction report changed"
)
historical = json.loads(endpoint_report.read_text())

for row in historical["condition_interactions"]:
    if row["endpoint"] != "recovery_s":
        continue

    code = row["comparator"]

    for key, reported in (
        ("Low", row["low_mean_improvement"]),
        ("Elevated", row["elevated_mean_improvement"]),
        ("Interaction", row["elevated_minus_low"])
    ):
        require(
            abs(effects[(key, code)] - reported) < 1e-10,
            f"Effect verification failed: {key}/{code}"
        )

# Snapshot original publication files to prevent overwriting.
original_hashes = {
    path.name: sha(path)
    for path in FIGURES.iterdir()
    if path.is_file()
    and path.suffix.lower() in (".pdf", ".png")
}

fig, axes = plt.subplots(
    1, 3,
    figsize=(18.0, 6.4),
    layout="constrained"
)

panels = [
    (
        "Low",
        "A. Low initialization",
        (-0.15, 0.34)
    ),
    (
        "Elevated",
        "B. Elevated initialization",
        (-0.15, 0.78)
    ),
    (
        "Interaction",
        "C. Elevated - Low interaction",
        (-0.40, 0.80)
    )
]

y = np.arange(4)

for ax, (key, title, limits) in zip(axes, panels):
    means = np.array([
        effects[(key, code)] for code in comparators
    ])

    ci_low = np.array([
        intervals[(key, code)][0] for code in comparators
    ])

    ci_high = np.array([
        intervals[(key, code)][1] for code in comparators
    ])

    ax.axvline(
        0, color="#333333",
        linewidth=1.6, zorder=2
    )

    for j, code in enumerate(comparators):
        ax.errorbar(
            means[j], y[j],
            xerr=[
                [max(0, means[j] - ci_low[j])],
                [max(0, ci_high[j] - means[j])]
            ],
            fmt="o",
            markersize=8,
            color=colors[j],
            ecolor=colors[j],
            elinewidth=2.1,
            capsize=5,
            capthick=1.6,
            zorder=5
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_ylim(3.65, -0.65)
    ax.set_xlim(*limits)
    ax.set_title(title, pad=17)
    ax.set_xlabel(
        "Recovery improvement (s)"
        if key != "Interaction"
        else "Difference in improvement (s)"
    )
    ax.grid(
        axis="x", alpha=0.16,
        color="#707070", linewidth=0.8
    )
    ax.tick_params(axis="both", length=0, pad=7)

    for tick in (
        list(ax.get_xticklabels())
        + list(ax.get_yticklabels())
    ):
        tick.set_fontweight("bold")

fig.suptitle(
    "Condition-dependent recovery effects of motion-time AI control",
    fontsize=16,
    fontweight="bold"
)

# Export without replacing any previously approved figure.
temp_pdf = FIGURES / ("." + STEM + ".staged.pdf")
temp_png = FIGURES / ("." + STEM + ".staged.png")

require(not temp_pdf.exists(), "Staged PDF exists")
require(not temp_png.exists(), "Staged PNG exists")

try:
    fig.savefig(
        temp_pdf, format="pdf",
        bbox_inches="tight",
        pad_inches=0.22,
        facecolor="white"
    )
    fig.savefig(
        temp_png, format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.22,
        facecolor="white"
    )
finally:
    plt.close(fig)

require(
    temp_pdf.stat().st_size > 10000,
    "PDF unexpectedly small"
)
require(
    temp_png.stat().st_size > 10000,
    "PNG unexpectedly small"
)

with Image.open(temp_png) as image:
    dimensions = image.size
    dpi = image.info.get("dpi", (0, 0))

require(
    all(abs(float(v) - 600) < 2 for v in dpi),
    "PNG DPI mismatch"
)

exported_fonts = {
    f.decode("latin1").split("+")[-1]
    for f in re.findall(
        rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
        temp_pdf.read_bytes()
    )
}
require(
    "TimesNewRomanPS-BoldMT" in exported_fonts,
    "Exported PDF font differs from Figure 10"
)

for filename, expected in original_hashes.items():
    require(
        sha(FIGURES / filename) == expected,
        f"Existing publication figure changed: {filename}"
    )

temp_pdf.replace(PDF)
temp_png.replace(PNG)

data_manifest = {
    "figure": STEM,
    "step": 246,
    "scientific_role": "secondary_condition_specific",
    "comparison": "comparator_recovery_minus_B6_recovery",
    "positive_direction": "favors_B6",
    "interaction": "Elevated_improvement_minus_Low_improvement",
    "independent_profiles": 200,
    "paired_conditions": ["Low", "Elevated"],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 2026,
    "font": "Times New Roman Bold",
    "dpi": 600,
    "png_dimensions": dimensions,
    "effects": {
        code: {
            key: {
                "mean": effects[(key, code)],
                "CI95": [
                    float(v) for v in intervals[(key, code)]
                ]
            }
            for key in ("Low", "Elevated", "Interaction")
        }
        for code in comparators
    },
    "pdf_sha256": sha(PDF),
    "png_sha256": sha(PNG),
    "source_audit_sha256": sha(audit_file),
    "source_interaction_audit_sha256": sha(endpoint_report),
    "existing_figures_unchanged": True
}

MANIFEST.write_text(
    json.dumps(data_manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("\nCONDITION-SPECIFIC EFFECTS AND INTERACTIONS")

for code in comparators:
    print(
        f"{code}: "
        f"Low={effects[('Low',code)]:+.6f}, "
        f"Elevated={effects[('Elevated',code)]:+.6f}, "
        f"Interaction={effects[('Interaction',code)]:+.6f}"
    )

print("\nEXPORT INTEGRITY")
print("Font:", sorted(exported_fonts))
print("PNG dimensions:", dimensions)
print("PNG DPI:", dpi)
print("PDF:", PDF)
print("PNG:", PNG)
print("PDF SHA256:", sha(PDF))
print("PNG SHA256:", sha(PNG))
print("Manifest SHA256:", sha(MANIFEST))
print("Existing figures unchanged: PASS")
print("Frozen scientific results unchanged: PASS")
print("STEP 246: PASS")
