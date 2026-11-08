# Motion-Driven Closed-Loop Regulation Under Partial Observability

**Reproducible synthetic virtual-reality research software for causal motion-derived state estimation, budget-constrained feedback regulation, and matched controller evaluation.**

This repository contains the simulator, frozen machine-learning estimators, controller implementations, synthetic profile definitions, deterministic episode seeds, paired results, audit records, manuscript figures, and reproduction tools for a computational study of feedback regulation under partial observability.

The software release covers **two independently generated synthetic evaluation cohorts**: a three-policy study of **1,200 episodes** (Cohort A) and a separate nine-policy comparison of **3,600 episodes** (Cohort B). Their population-generation procedures differ; results are reported separately and should not be pooled as if they were one population.

- **Maintainer:** Abdul Rehman
- **Repository:** [AbdulRehman88/Multimodal-VR-Closed-Loop-Regulation](https://github.com/AbdulRehman88/Multimodal-VR-Closed-Loop-Regulation)
- **Release:** `v1.1.0` (documentation for the release candidate; tag and archive status must be confirmed independently)
- **License:** MIT for the original software; third-party packages retain their own licenses.
- **Scientific scope:** **Synthetic simulation only.** No human participants, measured physiological or emotional states, deployed VR application, hardware energy measurements, or clinical efficacy testing are represented by these experiments.

## 1. Research question and contributions

**Can causally available, incomplete motion observations support regulation of an unobserved synthetic state while respecting a strict intervention budget?**

This repository enables investigation of four connected questions:

1. **Closed-loop utility:** Does motion-informed feedback improve recovery and time spent in the regulated interval relative to constant-neutral intervention?
2. **Information dependence:** How do elapsed-time-only, motion-only, motion-threshold, and time-preserved sham-motion comparators perform relative to a motion-and-time estimator?
3. **Control-budget design:** What changes when the controller explicitly reserves enough remaining intervention variation for a feasible return to neutral?
4. **Performance boundaries:** How does the proposed controller compare with an information-privileged state oracle, and where does it exhibit adverse or condition-specific behavior?

The contribution is an **integrated causal simulation, estimation, constrained feedback, and leakage-aware evaluation framework**, not a new classification architecture, a proof of globally optimal control, or a real-world safety certification.

## 2. System and control protocol

### 2.1. Synthetic latent state

The simulator models a bounded latent variable on **[0, 10]**, with target **5** and regulated interval **[4, 6]**. Two initial conditions are considered:

| Condition | Initial synthetic state |
|---|---:|
| Low | 2 |
| Elevated | 8 |

“Low,” “Regulated,” and “Elevated” denote simulator-defined computational states, not clinically or psychologically validated conditions.

The synthetic population varies in latent-state and motion-related parameters. Motion is generated in three dimensions and depends on the simulated state and profile-specific parameters.

### 2.2. Causal observation and estimation

Motion is sampled at **90 Hz**. The controller consumes features computed from **completed one-second observation windows**; a decision cannot use an observation from its own future control time.

The frozen principal estimator is a `StandardScaler` + three-class `LogisticRegression` pipeline using **14 motion descriptors and elapsed time** (15 inputs). It predicts Low, Regulated, or Elevated classes; it **does not** regress the continuous latent-state value.

Elapsed time can encode the simulated recovery schedule. The comparative design therefore isolates time-only information (B2), motion-only information (B3), and time-preserved sham motion (B8).

### 2.3. Intervention limits

| Parameter | Frozen setting |
|---|---:|
| Sampling rate | 90 Hz |
| Episode duration | 12 s |
| Observation window | 1.0 s |
| Feedback frequency | 10 Hz |
| First feedback decision | After 1.0 s |
| Normalized stimulus range | [0, 1] |
| Neutral stimulus | 0.5 |
| Maximum absolute change per update | 0.05 |
| Total absolute stimulus-variation budget | 0.6 |

For intervention values \(u_k\), the expenditure is the sum of absolute changes at feedback updates (including the change from the initial neutral value). The **neutral-return reserve** limits expenditure so sufficient variation remains for a feasible return to `0.5`:

\[
V_k + |u_k - 0.5| \leq 0.6.
\]

This is a **budget-feasibility invariant** in the implemented simulator. It does not, by itself, guarantee that the intervention actually ends at neutral, that every episode recovers, or that the controller is safe in the physical world. Stimulus variation is **not** measured power or energy.

## 3. Evaluation cohorts: distinct provenance

| Design element | Cohort A | Cohort B |
|---|---|---|
| Purpose | Primary three-policy synthetic evaluation | Independent nine-policy comparative benchmark |
| Independent synthetic profiles | 200 | 200 |
| Initial conditions per profile | Low and Elevated | Low and Elevated |
| Policies | B0, B6, B7 | B0–B8 |
| Episodes per policy | 400 | 400 |
| Total policy episodes | **1,200** | **3,600** |
| Frozen evidence | `results/final_independent_evaluation/` | `revision/results/confirmatory_200/` |
| Profile-generation provenance | Original independent test population | Separately generated comparative population |

**Do not merge the two cohorts as though their profile identifiers or sampling distributions were identical.** They are separate synthetic populations with different generation provenance. Their within-cohort paired analyses are meaningful under their respective frozen protocols; a pooled effect estimate is not supplied here.

The principal development dataset used **60 training and 20 development-validation profiles**. Fitted classifiers and controller parameters were frozen before the independent evaluations. The development dataset and Cohort B seed ledger are included for the comparative reproduction pipeline.

## 4. Controllers and information controls

| ID | Policy | Main distinguishing feature |
|---|---|---|
| **B0** | Constant neutral | No active feedback; stimulus remains at `0.5` |
| **B1** | Fixed schedule | Prespecified intervention schedule |
| **B2** | Time-only classifier | Elapsed time without motion descriptors |
| **B3** | Motion-only classifier | Motion descriptors without elapsed time |
| **B4** | Motion-threshold policy | Rule-based feedback using motion thresholds |
| **B5** | AI, reserve disabled | Frozen motion-and-time feedback without neutral-return reservation |
| **B6** | AI, reserve enabled | Proposed frozen motion-and-time feedback with neutral-return reservation |
| **B7** | Privileged-state oracle | Direct access to the true **simulated** latent state, subject to the configured resource limits |
| **B8** | Time-preserved sham-motion policy | Replaces recipient motion descriptors with a deterministic training-only donor trajectory while preserving elapsed time |

**B5 versus B6:** They use the same frozen estimator and nominal per-update and total-variation limits. The intentionally manipulated factor is the neutral-return reserve.

**B7 is not deployable under the motion-only observation assumption.** It is an information-privileged reference, not a practical estimator or evidence of an optimal policy.

**B8 is a diagnostic control, not a newly fitted model.** Its 120 donor trajectories come exclusively from training profiles (60 profiles × two conditions). A deterministic hash of the episode seed chooses the donor; motion features retain coherent within-donor time ordering while the recipient’s elapsed-time input is unchanged. Donor selection does not use recipient outcomes or latent state.

## 5. Cohort A: three-policy results

The first independent synthetic cohort evaluates **400 episodes per policy**.

| Policy | Mean recovery time (s) | Regulated-state stability (%) |
|---|---:|---:|
| B0 — Neutral | 2.261917 | 81.150694 |
| **B6 — Reserved AI** | **2.036333** | **83.030556** |
| B7 — Privileged-state oracle | 1.894500 | 84.212500 |

Relative to neutral, B6 reduced mean recovery time by approximately **0.22558 s** and increased time in the regulated interval by approximately **1.87986 percentage points**. The privileged-state oracle performed better than B6 on both means.

The frozen adverse-intervention audit also records:

- **65/400** B6 episodes with an adverse paired outcome against neutral (slower recovery **or** lower stability).
- **167** episodes containing at least one opposite-direction action.
- **745** opposite-direction actions across those episodes.
- **373** episodes reaching the reserve boundary.

The two separate adverse criteria identified the same 65 episodes in this cohort. All 400 episodes recovered under each evaluated policy, although recovery times differed. These are synthetic, profile- and protocol-specific outcomes; they do not imply uniform individual-level benefit.

## 6. Cohort B: nine-policy comparative results

The independent comparative cohort evaluates **200 profiles × two initial conditions × nine policies = 3,600 episodes**. The values below are descriptive means or rates over **400 episodes per policy**.

| Policy | Mean recovery (s) | Stability (%) | Mean variation | Recovered (%) | Neutral-return feasible (%) |
|---|---:|---:|---:|---:|---:|
| B0 | 2.266861 | 81.109491 | 0.00000 | 100.00 | 100.00 |
| B1 | 2.301750 | 80.818750 | 0.60000 | 100.00 | 100.00 |
| B2 | 2.292667 | 80.894444 | 0.60000 | 100.00 | 100.00 |
| B3 | 2.141806 | 82.151620 | 0.28000 | 100.00 | 100.00 |
| B4 | 2.155056 | 82.041204 | 0.55475 | 100.00 | 100.00 |
| **B5** | **2.323139** | **72.554398** | **0.59025** | **96.75** | **10.75** |
| **B6** | **2.048389** | **82.930093** | **0.59025** | **100.00** | **100.00** |
| B7 | 1.905667 | 84.119444 | 0.55400 | 100.00 | 100.00 |
| B8 | 2.326556 | 80.612037 | 0.58875 | 100.00 | 100.00 |

B6 had better **overall mean recovery** than B0, B1, B2, B3, B4, B5, and B8 in this evaluation. B7, which accesses privileged latent-state information, achieved the best overall mean recovery and stability.

**Overall means do not establish universal superiority.** B6 was not best in every Low/Elevated subgroup or for every profile. In particular, the condition-specific results include unfavorable comparisons for B6 against some simpler controls.

### 6.1. Prespecified overall recovery contrasts

Positive values indicate **B6 recovery-time improvement** over the comparator.

| Contrast | B6 improvement (s) | Archived 95% confidence interval |
|---|---:|---|
| B6 vs B1 | 0.253361 | [0.218472, 0.289528] |
| B6 vs B2 | 0.244278 | [0.210222, 0.279946] |
| B6 vs B3 | 0.093417 | [0.072861, 0.113973] |
| B6 vs B4 | 0.106667 | [0.066804, 0.150862] |

The frozen comparative statistical analysis reports **Holm-adjusted p = 0.00008** for each of these four comparisons. These results use the study’s paired, profile-aware synthetic analysis protocol and do not establish effects in actual VR users.

### 6.2. Reserve ablation: B5 versus B6

B5 removes the neutral-return reserve while retaining the same nominal action bounds and frozen estimator as B6.

| Outcome | B5 (reserve off) | B6 (reserve on) |
|---|---:|---:|
| Mean recovery (s) | 2.323139 | 2.048389 |
| Stability (%) | 72.554398 | 82.930093 |
| Mean stimulus variation | 0.59025 | 0.59025 |
| Recovered (%) | 96.75 | 100.00 |
| Neutral-return feasible (%) | 10.75 | 100.00 |

Under B5, **357/400** episodes exhausted their ability to fund a return to neutral under the reserve criterion; the audit further identifies **77 relapses**, **90 terminal states outside [4, 6]**, and **13 unrecovered episodes**. These categories may overlap and must not be summed as independent events.

The paired comparison isolates the effect of the **implemented reserve rule under the synthetic dynamics**. It does not prove that the mathematical rule provides general safety outside this controlled model.

### 6.3. Information-source and sham controls

- **B2:** Tests what can be achieved using only elapsed time.
- **B3:** Tests feedback relying on motion without elapsed time.
- **B4:** Provides a simple rule-based motion policy.
- **B8:** Preserves the time input while replacing recipient motion with coherent, training-only sham-motion data.

These controls help distinguish information-source contributions from effects that could arise solely from elapsed time or generic intervention schedules. They do **not** validate a real-world physiological motion–state mapping.

## 7. Endpoints, pairing, and statistical interpretation

- **Recovery:** First qualifying entry into **[4, 6]** followed by continuous residence for **one second**. Unrecovered episodes receive the specified **12-second horizon** for recovery-time comparisons.
- **Stability:** Percentage of simulation samples within **[4, 6]**.
- **Variation:** Sum of absolute normalized intervention changes at controller updates.
- **Neutral-return feasibility:** Whether the remaining variation budget can fund returning the current stimulus to `0.5`; distinct from actually returning to neutral.
- **Opposite-direction action:** An action contrary to the corrective direction indicated by the true synthetic state, assessed retrospectively.
- **Adverse paired outcome:** An episode with worse recovery time or stability than its paired neutral reference, as defined by the frozen evaluation protocol.

The independent inferential unit is the **synthetic profile**. A profile contributes both initial conditions, and policy effects are paired within profile and condition; observation windows, motion frames, and controller updates are **not** independent replicate subjects.

Cohort A uses a **10,000-replicate profile-cluster bootstrap (seed 2026)** in its frozen statistical analysis. Cohort B has a separately frozen statistical protocol and multiplicity-adjusted contrasts. Inference from the two cohorts should be interpreted separately; confidence intervals on different endpoints from the same trajectories are not independent pieces of evidence.

## 8. Repository structure

The most important public paths are:

```text
configs/
  frozen_development_profiles.csv
  independent_synthetic_test_profiles.csv

src/
  controllers/
    simple_policies.py
    information_adapters.py
    sham_motion.py
  simulator/
    reserve_configurable.py
  models/
  evaluation/

revision/
  run_confirmatory_comparison.py      # Frozen historical runner; provenance
  run_portable_comparison.py           # Supported Cohort B reproduction entry point
  baseline_policies.py                # Frozen historical policy source
  simulator_B5_budget_only.py         # Frozen historical simulator source
  configs/
    comparative_protocol_lock.json
    confirmatory_profiles_200.csv
    confirmatory_episode_seeds.csv
    confirmatory_runner_lock.json
    confirmatory_results_integrity_audit.json
  models/
    B2_time_only.joblib
    B3_motion_only.joblib
  results/
    confirmatory_200/                 # 200 frozen profile checkpoints

results/
  frozen_one_second_ai_baseline/
    motion_time_unweighted.joblib
  motion_state_dataset/
    development_windows.csv
  final_independent_evaluation/
    final_results/
    publication_evidence/
    publication_figures_revised/

reproduce_archived_results.py
reproduce_final_evaluation.py
verify_matched_oracle.py
verify_statistical_inference.py

FULL_CANONICAL_REPRODUCTION_AUDIT.json
PORTABLE_COMPARATIVE_PREFLIGHT.json
CANONICAL_B8_PARITY.json
CANONICAL_RESERVE_SIMULATOR_PARITY.json
RELEASE_SHA256_MANIFEST.json
RELEASE_NOTES_v1.1.0.md

requirements-repro.txt
requirements-lock.txt
CITATION.cff
LICENSE
```

The `revision/` namespace includes **historical execution and audit records**. Some historical scripts retain original machine-specific paths; those are retained for provenance, not represented as universally portable entry points. The supported comparative runner is `revision/run_portable_comparison.py`.

## 9. Reproducibility

### 9.1. Python environment

The established clean Cohort A reproduction used **Python 3.11.9 on Windows**. The repository includes:

- `requirements-lock.txt`: pinned package-version inventory from the verified clean Cohort A environment.
- `requirements-repro.txt`: smaller direct-dependency specification.

Example Windows PowerShell setup from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

A pinned package list is not a wheel-hash lock and **does not guarantee bitwise reproducibility on every platform or future installation**. The full Cohort B numerical replay was verified in the established project environment; no separate cross-platform or clean-install test of that complete package is claimed.

### 9.2. Cohort A: tests and archived checks

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe reproduce_archived_results.py
.\.venv\Scripts\python.exe verify_statistical_inference.py
.\.venv\Scripts\python.exe verify_matched_oracle.py
.\.venv\Scripts\python.exe reproduce_final_evaluation.py
```

The verified original environment produced **30 passing unit tests**. Replaying the 1,200 archived policy episodes yielded **zero mismatched episodes**, with reported maximum absolute numerical difference **1.110 × 10^-16**. Additional inference and oracle audits are preserved in the release evidence.

### 9.3. Cohort B: portable nine-policy runner

From the repository root, using the prepared environment:

```powershell
.\.venv\Scripts\python.exe revision/run_portable_comparison.py --mode run
```

This runner resolves its project inputs relative to the repository and creates new records under:

```text
results/reproduced_comparative_200/
```

It must **not** overwrite the authoritative frozen records under:

```text
revision/results/confirmatory_200/
```

The original full comparative run is preserved by `revision/run_confirmatory_comparison.py` and its historical lock. The portable runner has been checked against that implementation and archived episode results.

**Full verification outcome:** All **3,600/3,600** Cohort B episodes were regenerated with exact agreement for archived recovery and stability metrics, budget-related outcomes, controller-update values, and state/motion trajectory SHA-256 hashes. No frozen checkpoint was modified.

The verification evidence includes:

| Audit file | Verified purpose |
|---|---|
| `PORTABLE_COMPARATIVE_PREFLIGHT.json` | Portable runner agreement for all nine policies in representative cases |
| `FULL_CANONICAL_REPRODUCTION_AUDIT.json` | **3,600/3,600** exact comparative episode reproductions |
| `CANONICAL_B8_PARITY.json` | 120/120 donor trajectories, 400/400 deterministic selections, and 12 × 4,440 held-out feature/prediction comparisons |
| `CANONICAL_RESERVE_SIMULATOR_PARITY.json` | 48 tested B5/B6 trajectory reproductions against frozen references |
| `revision/configs/confirmatory_results_integrity_audit.json` | SHA-256 inventory and integrity verification for 200 archived comparative checkpoints |
| `RELEASE_SHA256_MANIFEST.json` | File-level integrity manifest of the assembled release candidate |

The full 3,600-episode replay—not the smaller 48-case module parity test—is the basis for the final comparative reproduction claim.

## 10. Figures and analytical provenance

Manuscript-ready figures and their underlying analysis evidence are stored in:

```text
results/final_independent_evaluation/publication_figures_revised/
results/final_independent_evaluation/publication_evidence/
```

The comparative additions include PDF and PNG versions of:

- **F10:** `F10_comparative_policy_performance_v2`
- **F11:** `F11_condition_specific_policy_effects`
- **F12:** `F12_neutral_return_reserve_ablation`

Additional archived figures document the framework architecture, causal motion windows, state trajectories, controller comparisons, intervention behavior, heterogeneity, and adverse outcomes.

**Figures and executable regeneration are separate claims.** The archive provides the figure files and associated numerical evidence, but it does not guarantee that every manuscript figure can be recreated from one supported command without additional historical script setup.

## 11. Data integrity and frozen provenance

The principal frozen files include:

| File | Role |
|---|---|
| `results/frozen_one_second_ai_baseline/motion_time_unweighted.joblib` | Original motion-and-time estimator |
| `revision/models/B2_time_only.joblib` | Time-only comparison model |
| `revision/models/B3_motion_only.joblib` | Motion-only comparison model |
| `revision/configs/comparative_protocol_lock.json` | Frozen comparative design and integrity declarations |
| `revision/configs/confirmatory_profiles_200.csv` | Independent Cohort B profiles |
| `revision/configs/confirmatory_episode_seeds.csv` | Exact 400-case paired seed ledger |
| `revision/results/confirmatory_200/` | Original 200 profile checkpoint files |
| `revision/configs/confirmatory_results_integrity_audit.json` | Checkpoint inventory and verification |

Selected frozen SHA-256 values:

```text
Principal motion-and-time model:
1884007ac6581b6653c3227e464256b5ccc56c1e30acf847f543371312f36467

Cohort B profiles:
e2ab295d7adfafb0926947007d3cfa61541afb9d0c55c1f71df193f33aec5b23

Cohort B seed ledger:
05fe01c4d0cc28d272603c774a25ad811c77c1e14481d456295345c79a5f6143

Full 3,600-episode reproduction audit:
78ec617a43ee647bd8fe43f7b8203d5d9ce3ec1d5a38df13a28108bf51707617
```

The original cohort protocol and results are preserved. Historical evidence locks have not been silently rewritten to fit the newer portable entry point. To verify the published release, use the manifest and the **exact version/tag** under examination; earlier release manifests do not necessarily describe later documentation edits.

## 12. Limitations and claims boundary

The experimental findings are conditional on the **constructed stochastic dynamics, selected profile-generation procedures, model features, frozen estimator, action constraints, and outcome definitions**.

In particular, this software does **not** demonstrate:

- A physiologically valid, user-specific latent state.
- Stress, affect, attention, or emotion inference from measured human VR motion.
- Performance with real participants, headsets, eye trackers, or physical intervention hardware.
- Cross-device or cross-population generalization beyond the synthetic simulations.
- Measured hardware energy efficiency, user safety, therapeutic benefit, or clinical efficacy.
- Globally optimal control or uniformly superior behavior across profiles and initial conditions.
- Unbiased state identification in the absence of synthetic motion–state coupling assumptions.

The elapsed-time feature may act as a shortcut; the classifier probabilities were not established as calibrated estimates. The time-only, motion-only, and sham-motion controls address internal information dependence but cannot substitute for external measured-data validation. The two evaluation cohorts were separately generated, and their parameter sampling procedures are not presumed identical.

The reserve rule establishes a specific **mathematical budget constraint**, not universal safety or recovery.

## 13. Versioning, citation, and permanent archives

The preceding release is preserved as:

- **GitHub v1.0.1:** [Verified Research Software Archive](https://github.com/AbdulRehman88/Multimodal-VR-Closed-Loop-Regulation/releases/tag/v1.0.1)
- **Zenodo v1.0.1 version DOI:** [10.5281/zenodo.23999113](https://doi.org/10.5281/zenodo.23999113)

**The v1.0.1 DOI refers to the earlier three-policy software archive.** It must not be presented as a permanent deposit containing this full nine-policy `v1.1.0` package.

The version-specific DOI for **v1.1.0** should be added **only after** its GitHub release is tagged and Zenodo confirms the corresponding permanent deposit. Consult `CITATION.cff` for repository citation metadata, and cite the scientific article separately when available.

## 14. Contact and reproducibility questions

**Maintainer:** Abdul Rehman  
**GitHub:** [AbdulRehman88](https://github.com/AbdulRehman88)

For reproduction issues, open a GitHub issue and include the operating system, Python version, exact command, relevant console output, and the Git commit or release tag being evaluated.
