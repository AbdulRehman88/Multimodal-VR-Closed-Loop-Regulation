from pathlib import Path
import sys, json, hashlib, re, importlib.util
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
sys.path.insert(0, str(ROOT))

from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.controllers.baselines import ControllerParameters
from src.evaluation.control_metrics import evaluate_controller

CONFIG = ROOT / "revision/configs"
RESULTS = ROOT / "revision/results"
FIGURES = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR"
    r"\results\final_independent_evaluation"
    r"\publication_figures_revised"
)
EVIDENCE = RESULTS / "publication_evidence"

STEM = "F12_neutral_return_reserve_ablation"
PDF = FIGURES / (STEM + ".pdf")
PNG = FIGURES / (STEM + ".png")
MANIFEST = EVIDENCE / "F12_figure_integrity.json"

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def array_sha(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()

print("STEP 247 - FIGURE 12: NEUTRAL-RETURN RESERVE")

for path in (PDF, PNG, MANIFEST):
    require(not path.exists(), f"Output exists: {path}")

audit_file = CONFIG / "confirmatory_results_integrity_audit.json"
relapse_file = RESULTS / "post_recovery_relapse_audit.json"
dependence_file = RESULTS / "endpoint_dependence_audit.json"

require(
    sha(audit_file)
    == "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d",
    "Confirmatory audit mismatch"
)
require(
    sha(relapse_file)
    == "2f6d6e37349551bdc0c48fbbf173aaae66ab6a950b74bcabc8e7ab268d03b264",
    "Relapse audit mismatch"
)
require(
    sha(dependence_file)
    == "fe3b814014aa94d4aa62a170b4783970a5c53aebcee52233bd3210b52a609127",
    "Resource-parity audit mismatch"
)

audit = json.loads(audit_file.read_text(encoding="utf-8"))
relapse = json.loads(relapse_file.read_text(encoding="utf-8"))
dependence = json.loads(dependence_file.read_text(encoding="utf-8"))

require(relapse["replayed_trajectories"] == 800,
        "Incomplete historical replay audit")
require(
    dependence["B5_B6_resource_audit"]["equal_total_variation"] == 400,
    "Cohort resource-parity mismatch"
)

reference = FIGURES / "F10_comparative_policy_performance_v2.pdf"
require(
    sha(reference)
    == "e3b1a73ed9140eb374afda4cca53a67566a1e1d0803c2a9604d011ed0cf50586",
    "Approved figure-style reference changed"
)

fonts = {
    x.decode("latin1").split("+")[-1]
    for x in re.findall(
        rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
        reference.read_bytes()
    )
}
require("TimesNewRomanPS-BoldMT" in fonts, "Font mismatch")
require(
    "Times New Roman" in {
        f.name for f in font_manager.fontManager.ttflist
    },
    "Times New Roman unavailable"
)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "font.weight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True
})

# Previously audited adverse episode; illustrative, not typical.
PID = 200003
CONDITION = "Low"
INITIAL = 2.0

profiles = pd.read_csv(
    CONFIG / "confirmatory_profiles_200.csv"
).set_index("profile")

ledger = pd.read_csv(
    CONFIG / "confirmatory_episode_seeds.csv"
)

selected = ledger.loc[
    (ledger["profile"] == PID)
    & (ledger["condition"] == CONDITION)
]
require(len(selected) == 1, "Episode seed missing")
seed = int(selected.iloc[0]["episode_seed"])

checkpoint = (
    RESULTS / "confirmatory_200" / f"profile_{PID}.json"
)
entry = next(
    x for x in audit["checkpoint_inventory"]
    if int(x["profile"]) == PID
)
require(sha(checkpoint) == entry["sha256"],
        "Illustrative checkpoint changed")

saved = {
    (r["condition"], r["policy"]): r
    for r in json.loads(checkpoint.read_text(encoding="utf-8"))
}

p = profiles.loc[PID]
state_params = StateParameters(
    baseline=float(p["baseline"]),
    relaxation_rate=float(p["relaxation_rate"]),
    stimulus_gain=float(p["stimulus_gain"])
)
motion_params = MotionParameters(
    frequency_hz=float(p["motion_frequency_hz"]),
    state_sensitivity=float(p["motion_state_sensitivity"])
)
coupled_params = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05
)

model = joblib.load(
    ROOT / "results/frozen_one_second_ai_baseline"
    / "motion_time_unweighted.joblib"
)

path = ROOT / "revision/simulator_B5_budget_only.py"
spec = importlib.util.spec_from_file_location(
    "figure12_budget_simulator", path
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

common = dict(
    seed=seed,
    state_params=state_params,
    motion_params=motion_params,
    coupled_params=coupled_params,
    controller_params=ControllerParameters(),
    initial_state=INITIAL,
    control_rate_hz=10,
    model=model,
    policy="ai_neutral_return",
    maximum_update_change=0.05,
    maximum_total_variation=0.6
)

runs = {
    "B5": module.simulate_motion_feedback(
        **common, reserve_enabled=False
    ),
    "B6": simulate_motion_feedback(**common)
}

metrics = {}
margins = {}
variations = {}

for code, run in runs.items():
    old = saved[(CONDITION, code)]

    require(
        array_sha(run["state"]) == old["state_sha256"],
        f"State reproduction failed: {code}"
    )

    motion_hash = hashlib.sha256(
        np.ascontiguousarray(run["position"]).tobytes()
        + np.ascontiguousarray(run["velocity"]).tobytes()
    ).hexdigest()
    require(
        motion_hash == old["motion_sha256"],
        f"Motion reproduction failed: {code}"
    )
    require(
        np.array_equal(
            np.asarray(run["stimulus"])[::9],
            np.asarray(old["stimulus_at_updates"])
        ),
        f"Stimulus reproduction failed: {code}"
    )

    metrics[code] = evaluate_controller(
        run, hold_duration_s=1.0
    )
    updates = np.asarray(run["stimulus"])[::9]
    increments = np.abs(np.diff(np.r_[0.5, updates]))
    cumulative = np.cumsum(increments)
    distances = np.abs(updates - 0.5)

    margins[code] = 0.6 - cumulative - distances
    variations[code] = float(increments.sum())

require(
    abs(variations["B5"] - variations["B6"]) < 1e-10,
    "Paired variation not equal"
)
require(
    float(metrics["B5"]["opposite_boundary_overshoot"]) > 0,
    "Selected example is not an overshoot episode"
)

summary = relapse["policy_summary"]

require(
    summary["B5"]["relapsed_after_qualifying_window_count"] == 77,
    "B5 relapse result mismatch"
)
require(
    summary["B6"]["relapsed_after_qualifying_window_count"] == 0,
    "B6 relapse result mismatch"
)
require(
    summary["B5"]["final_state_outside_count"] == 90,
    "B5 terminal-state result mismatch"
)
require(
    summary["B6"]["final_state_outside_count"] == 0,
    "B6 terminal-state result mismatch"
)

# Preserve every prior figure.
original_hashes = {
    f.name: sha(f)
    for f in FIGURES.iterdir()
    if f.is_file() and f.suffix.lower() in (".pdf", ".png")
}

RED = "#B24B50"
BLUE = "#24618E"
GRAY = "#727B86"
TIME = np.arange(1080) / 90.0
CONTROL_TIME = np.arange(120) / 10.0

fig, axes = plt.subplots(
    2, 2,
    figsize=(17.5, 9.4),
    layout="constrained",
    facecolor="white"
)

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

def style(ax, grid_axis="both"):
    ax.grid(
        axis=grid_axis, color="#707070",
        alpha=0.15, linewidth=0.8
    )
    ax.tick_params(axis="both", length=0, pad=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

# A. Exact simulated hidden-state trajectories.
ax_a.axhspan(4, 6, color="#30836D", alpha=0.10)
for boundary in (4, 6):
    ax_a.axhline(
        boundary, color="#30836D",
        linestyle="--", linewidth=1.1
    )
ax_a.plot(
    TIME, runs["B5"]["state"],
    color=RED, linewidth=2.1,
    label="B5  AI without reserve"
)
ax_a.plot(
    TIME, runs["B6"]["state"],
    color=BLUE, linewidth=2.1,
    label="B6  Proposed AI + reserve"
)
ax_a.set(
    xlim=(0, 12),
    ylim=(0, 10),
    xlabel="Elapsed time (s)",
    ylabel="Simulated hidden state",
    title="A. Matched state trajectories"
)
ax_a.legend(loc="upper right", frameon=False)
style(ax_a)

# B. Exact stimulus allocation.
for code, color in (("B5", RED), ("B6", BLUE)):
    ax_b.step(
        TIME,
        runs[code]["stimulus"],
        where="post",
        linewidth=2.0,
        color=color,
        label=code
    )
ax_b.axhline(
    0.5, linestyle=":", color=GRAY,
    linewidth=1.2, label="Neutral stimulus"
)
ax_b.set(
    xlim=(0, 12),
    ylim=(0, 1),
    xlabel="Elapsed time (s)",
    ylabel="Normalized stimulus",
    title="B. Intervention allocation"
)
ax_b.legend(loc="upper right", frameon=False)
style(ax_b)

# C. Reserve margin is a feasibility quantity, not
# terminal stimulus neutrality or an observed safety score.
for code, color in (("B5", RED), ("B6", BLUE)):
    ax_c.step(
        CONTROL_TIME,
        margins[code],
        where="post",
        linewidth=2.0,
        color=color,
        label=code
    )
ax_c.axhline(
    0, color="#333333",
    linewidth=1.3, linestyle="--"
)
ax_c.set(
    xlim=(0, 12),
    xlabel="Elapsed time (s)",
    ylabel="Remaining neutral-return margin",
    title="C. Return-capacity constraint"
)
ax_c.legend(loc="upper right", frameon=False)
style(ax_c)

# D. Full-cohort descriptive mechanisms.
labels = [
    "Return capacity\nunavailable",
    "Post-recovery\nrelapse",
    "Final state\noutside interval",
    "Failed to\nrecover"
]

b5_counts = [
    357,
    summary["B5"]["relapsed_after_qualifying_window_count"],
    summary["B5"]["final_state_outside_count"],
    400 - summary["B5"]["recovered_count"]
]

b6_counts = [
    0,
    summary["B6"]["relapsed_after_qualifying_window_count"],
    summary["B6"]["final_state_outside_count"],
    400 - summary["B6"]["recovered_count"]
]

require(b5_counts == [357, 77, 90, 13],
        "B5 aggregate values mismatch")
require(b6_counts == [0, 0, 0, 0],
        "B6 aggregate values mismatch")

y = np.arange(4)
rates_b5 = 100 * np.asarray(b5_counts) / 400
rates_b6 = 100 * np.asarray(b6_counts) / 400

ax_d.barh(
    y - 0.16, rates_b5,
    height=0.30, color=RED,
    label="B5  No reserve", zorder=3
)
ax_d.barh(
    y + 0.16, rates_b6,
    height=0.30, color=BLUE,
    label="B6  With reserve", zorder=3
)

for i, (rate, count) in enumerate(zip(rates_b5, b5_counts)):
    ax_d.text(
        min(rate + 1.2, 96), i - 0.16,
        f"{rate:.2f}% ({count}/400)",
        fontsize=11, fontweight="bold",
        va="center", ha="left"
    )
    ax_d.text(
        2, i + 0.16, "0/400",
        fontsize=10.5, fontweight="bold",
        va="center", ha="left", color=BLUE
    )

ax_d.set_yticks(y, labels)
ax_d.invert_yaxis()
ax_d.set_xlim(0, 112)
ax_d.set_xlabel("Episodes (%)")
ax_d.set_title(
    "D. Reserve-related outcomes (400 paired episodes)",
    pad=14
)
ax_d.legend(
    loc="lower right",
    frameon=False
)
style(ax_d, "x")

fig.suptitle(
    "Neutral-return reservation: allocation and regulation outcomes",
    fontsize=16,
    fontweight="bold"
)

tmp_pdf = FIGURES / ("." + STEM + ".staged.pdf")
tmp_png = FIGURES / ("." + STEM + ".staged.png")

require(not tmp_pdf.exists(), "Staged PDF already exists")
require(not tmp_png.exists(), "Staged PNG already exists")

try:
    fig.savefig(
        tmp_pdf, format="pdf",
        bbox_inches="tight",
        pad_inches=0.22
    )
    fig.savefig(
        tmp_png, format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.22
    )
finally:
    plt.close(fig)

require(tmp_pdf.stat().st_size > 10000, "PDF too small")
require(tmp_png.stat().st_size > 10000, "PNG too small")

with Image.open(tmp_png) as im:
    dimensions = im.size
    dpi = im.info.get("dpi", (0, 0))

require(
    all(abs(float(v) - 600) <= 2 for v in dpi),
    "PNG DPI mismatch"
)

output_fonts = {
    v.decode("latin1").split("+")[-1]
    for v in re.findall(
        rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
        tmp_pdf.read_bytes()
    )
}
require(
    "TimesNewRomanPS-BoldMT" in output_fonts,
    "Original PDF font not preserved"
)

for filename, expected in original_hashes.items():
    require(
        sha(FIGURES / filename) == expected,
        f"Existing figure changed: {filename}"
    )

tmp_pdf.replace(PDF)
tmp_png.replace(PNG)

manifest = {
    "step": 247,
    "figure": STEM,
    "role": "secondary_reserve_ablation_and_posthoc_diagnostics",
    "illustrative_episode": {
        "profile": PID,
        "condition": CONDITION,
        "episode_seed": seed,
        "selection": "previously audited adverse B5 example",
        "representative_random_selection": False,
        "B5_state_sha256": saved[(CONDITION, "B5")]["state_sha256"],
        "B6_state_sha256": saved[(CONDITION, "B6")]["state_sha256"],
        "B5_variation": variations["B5"],
        "B6_variation": variations["B6"]
    },
    "aggregate_episodes_per_policy": 400,
    "B5_counts": dict(zip(labels, b5_counts)),
    "B6_counts": dict(zip(labels, b6_counts)),
    "font": "Times New Roman Bold",
    "dpi": 600,
    "png_dimensions": dimensions,
    "pdf_sha256": sha(PDF),
    "png_sha256": sha(PNG),
    "source_audit_sha256": sha(audit_file),
    "source_relapse_audit_sha256": sha(relapse_file),
    "frozen_results_modified": False,
    "original_figures_modified": False
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("\nVERIFIED ILLUSTRATIVE EPISODE")
print("Profile:", PID, "Condition:", CONDITION)
print("Episode seed:", seed)
print("B5 cumulative variation:", variations["B5"])
print("B6 cumulative variation:", variations["B6"])
print("B5 opposite-boundary overshoot:",
      metrics["B5"]["opposite_boundary_overshoot"])
print("Exact state/motion/stimulus replay: PASS")

print("\nCOHORT MECHANISM COUNTS")
print("B5:", b5_counts)
print("B6:", b6_counts)
print("Late 3-s stability B5:",
      summary["B5"]["final_3s_stability_mean_pct"])
print("Late 3-s stability B6:",
      summary["B6"]["final_3s_stability_mean_pct"])

print("\nFIGURE OUTPUT")
print("PDF:", PDF)
print("PNG:", PNG)
print("PNG dimensions:", dimensions)
print("PNG DPI:", dpi)
print("PDF fonts:", sorted(output_fonts))
print("PDF SHA256:", sha(PDF))
print("PNG SHA256:", sha(PNG))
print("Manifest SHA256:", sha(MANIFEST))
print("Existing figures unchanged: PASS")
print("Frozen experimental results unchanged: PASS")
print("STEP 247: PASS")
