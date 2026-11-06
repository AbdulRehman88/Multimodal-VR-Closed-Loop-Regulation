# Motion-Driven Closed-Loop Regulation Under Partial Observability

**A reproducible synthetic framework for causal motion-derived state estimation and resource-constrained feedback regulation.**

This repository contains the frozen implementation, trained estimator, independent evaluation profiles, paired experimental results, and verification tools for a simulation-based study of motion-informed closed-loop regulation.

**Maintainer:** Abdul Rehman  
**Repository:** https://github.com/AbdulRehman88/Multimodal-VR-Closed-Loop-Regulation  
**Status:** Reproduced independent evaluation; publication software release preparation.

## 1. Research objective

The central question is whether incomplete, causal motion-derived information can support effective feedback regulation of an unobserved synthetic state under an explicit intervention budget.

The framework evaluates:

1. Whether motion-informed feedback improves recovery and regulated-state residence compared with constant-neutral intervention.
2. How much performance is lost relative to a resource-matched controller with direct access to the simulated state.
3. How prediction errors, incorrect intervention directions, profile heterogeneity, and resource saturation affect closed-loop outcomes.

The contribution is the integrated control and evaluation framework, not a novel classification algorithm or an optimal-control guarantee.

**Scientific scope:** All experiments are synthetic. This repository does not contain human-subject experiments, measured physiological states, a deployed VR application, or clinical efficacy evidence.

## 2. System overview

The experimental framework consists of:

- A profile-parameterized, bounded latent-state simulator.
- State-dependent, three-dimensional stochastic motion generation at 90 Hz.
- Completed, causal one-second observation windows.
- Fourteen motion descriptors and elapsed time (15 inputs).
- A frozen `StandardScaler + LogisticRegression` three-class estimator.
- A 10-Hz motion-informed feedback controller.
- A cumulative stimulus-variation budget with a reserved capacity for returning to neutral.
- A constant-neutral comparator and a matched privileged-state oracle.
- Independent profile evaluation, paired disturbances, cluster-aware inference, and adverse-intervention audits.

### Synthetic state

The hidden state is bounded to `[0, 10]`, with target `5` and regulated interval `[4, 6]`.

Two initial conditions are evaluated:

- Low: initial state `2`.
- Elevated: initial state `8`.

These labels represent simulated computational states, not validated psychological or physiological conditions.

### Motion inference

The classifier receives 14 causal motion features plus elapsed simulation time. Its output is one of:

- `0`: Low
- `1`: Regulated
- `2`: Elevated

The estimator does not predict continuous latent-state values. Its fitted parameters are frozen before independent evaluation.

### Feedback constraints

The normalized intervention is bounded to `[0, 1]` with neutral value `0.5`.

The AI and matched oracle operate under:

- Control frequency: 10 Hz.
- First feedback decision: 1.0 s.
- Maximum stimulus change per update: 0.05.
- Total stimulus-variation allowance: 0.6.
- Neutral-return reserve: enabled.

The reserve mechanism preserves sufficient remaining variation capacity to return to neutrality. It does not mathematically force terminal neutrality for every possible trajectory.

Stimulus variation is a normalized computational control-effort measure, not physical energy consumption.

## 3. Experimental design

| Parameter | Frozen value |
|---|---|
| Simulation frequency | 90 Hz |
| Episode duration | 12 s |
| Observation window | 90 samples / 1 s |
| Control frequency | 10 Hz |
| Estimator inputs | 15 |
| Training profiles | 60 |
| Development-validation profiles | 20 |
| Independent test profiles | 200 |
| Conditions per test profile | Low and Elevated |
| Test episodes per policy | 400 |
| Evaluated policies | Neutral, reserved AI, matched reserved oracle |
| Total policy episodes | 1,200 |
| Bootstrap replicates | 10,000 |
| Bootstrap seed | 2026 |
| Independent resampling unit | Synthetic profile |

Profile assignments are separated between development and independent evaluation. All policy comparisons retain the specified paired profile-condition disturbances.

## 4. Frozen evaluation results

The following values are means across the 400 independent test episodes per policy.

| Policy | Recovery (s) | Stability (%) | Stimulus variation |
|---|---:|---:|---:|
| Neutral | 2.261917 | 81.150694 | 0.0000 |
| Reserved AI | 2.036333 | 83.030556 | 0.5885 |
| Matched reserved oracle | 1.894500 | 84.212500 | 0.5535 |

Compared with neutral regulation, reserved AI improved mean recovery by approximately 0.22558 s and regulated-state stability by approximately 1.87986 percentage points.

The privileged-state oracle performed better than reserved AI under the matched resource restrictions.

The final reliability audit also identified:

- 65 of 400 AI episodes with an adverse outcome relative to neutral.
- 167 episodes containing at least one opposite-direction action.
- 745 opposite-direction actions in total.
- 373 episodes reaching the AI reserve boundary.

The formal adverse-outcome definition is slower recovery **or** lower stability relative to the paired neutral episode. In the final evaluation, these two individual criteria identified the same 65 episodes.

All 400 final episodes recovered under each evaluated policy; recovery time nevertheless differed between policies.

These findings apply only to the tested synthetic dynamics and profile population.

## 5. Repository structure

```text
configs/
  controller_tuning_protocol.json
  frozen_development_profiles.csv
  independent_synthetic_test_profiles.csv

src/
  controllers/
  evaluation/
  models/
  simulator/

tests/

results/
  frozen_one_second_ai_baseline/
    model_manifest.json
    motion_time_unweighted.joblib

  final_independent_evaluation/
    locked_protocol.json
    locked_protocol.sha256
    final_results/
    publication_evidence/
    publication_figures_revised/

reproduce_archived_results.py
reproduce_final_evaluation.py
verify_matched_oracle.py
verify_statistical_inference.py

requirements-repro.txt
requirements-lock.txt
STAGING_MANIFEST.json
LICENSE
CITATION.cff
```

The release includes the accepted experimental implementation and final evidence. Exploratory controllers, failed development attempts, and superseded experiments are not part of the publication workflow.

## 6. Reproducibility environment

The end-to-end evaluation was reproduced on Windows using Python 3.11.9.

The recommended reproducibility specification is:

`requirements-lock.txt`

This is the complete pinned package-version inventory from the clean environment in which all 1,200 episodes reproduced successfully. It is not a wheel-hash lock and does not independently establish cross-platform bitwise reproducibility.

For a new Windows PowerShell environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements-repro.txt` lists nine direct dependencies and is available for a lighter installation.

The older `requirements.txt` is a retained development-environment snapshot. It includes dependencies not needed by the accepted scikit-learn pipeline and is not the recommended installation file.

## 7. Reproducing the published evaluation

Run these commands from the repository root, using the installed Python environment.

### Unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Expected result: 30 passing tests.

### Archived numerical evidence

```powershell
.\.venv\Scripts\python.exe reproduce_archived_results.py
```

This verifies the frozen file manifest, 400 deterministic episode seeds, 1,200 paired policy records, principal means, paired-effect summaries, selected bootstrap results, adverse episodes, and resource audits.

### Statistical inference

```powershell
.\.venv\Scripts\python.exe verify_statistical_inference.py
```

This checks all 18 paired-effect contrasts and their archived intervals, plus 18 profile-bootstrap safety-rate intervals.

The paired-effect confidence intervals are evaluated with documented tolerances for differences between independently generated bootstrap random sequences. The safety-rate intervals were reproduced exactly in the verified environment.

### Full independent evaluation

```powershell
.\.venv\Scripts\python.exe reproduce_final_evaluation.py
```

This regenerates the complete 1,200-episode test from the frozen model, profile parameters, deterministic seeds, and implemented policies.

For each episode, it compares recovery status and six numerical outcomes against the archived records.

**Verified acceptance result:**

```text
Episodes verified: 1200
Numeric metrics per episode: 6
Mismatched episodes: 0
Maximum absolute error: 1.110e-16
STEP 201: PASS
```

This result was reproduced using a separately created clean environment. Numerical equivalence is established for that environment, not guaranteed across arbitrary operating systems or future dependency versions.

### Matched-oracle verification

```powershell
.\.venv\Scripts\python.exe verify_matched_oracle.py
```

The oracle-specific verification independently checks all 400 privileged-state episodes against the archived outcomes.

## 8. Evaluation definitions

**Recovery:** First qualifying entry into the regulated state interval followed by a continuous one-second residence. Unrecovered episodes receive the prescribed 12-second horizon for final recovery-time comparisons.

**Stability:** Percentage of simulation samples within the regulated interval.

**Stimulus variation:** Sum of absolute stimulus changes across control updates.

**Adverse outcome:** An AI episode having slower recovery or lower regulated-state stability than its paired neutral episode.

**Opposite-direction action:** An intervention inconsistent with the corrective direction indicated by the true synthetic state.

**Reserve-boundary event:** An episode in which the admissible neutral-return variation allowance reaches its boundary.

The true latent state is used by the simulator, matched oracle, and retrospective evaluation. It is not supplied to the motion-informed AI estimator.

## 9. Statistical methodology

The independent statistical unit is the synthetic profile, not an individual window or control decision.

Each profile contributes two initial-condition episodes. Policy effects are paired within the same profile and condition, with the two conditions grouped when computing overall profile effects.

The final analysis uses a 10,000-replicate profile-cluster bootstrap with seed 2026 and reports 95% confidence intervals.

The bootstrap intervals are descriptive and do not represent familywise-adjusted inference for multiple endpoints.

## 10. Figures, tables, and provenance

The final manuscript figure assets are archived under:

`results/final_independent_evaluation/publication_figures_revised/`

They include the motion-driven architecture, causal windows, representative trajectories, adverse-case illustration, primary controller comparisons, heterogeneity, intervention trade-offs, adverse-event summaries, and development information-source ablations.

Supporting tables, episode-level data, and figure provenance are preserved under:

`results/final_independent_evaluation/publication_evidence/`

The package contains archived publication figures and their numerical evidence. A complete executable regeneration pipeline for every figure is not presently included; figure availability must not be confused with source-level figure regeneration.

## 11. Integrity and provenance

The authoritative independent evaluation protocol is:

`results/final_independent_evaluation/locked_protocol.json`

The frozen model is:

`results/frozen_one_second_ai_baseline/motion_time_unweighted.joblib`

Frozen model SHA-256:

`1884007ac6581b6653c3227e464256b5ccc56c1e30acf847f543371312f36467`

Frozen protocol SHA-256:

`ba193b21e6e990489e395c91b72a8c0eabbba8a61af8f1c5165a52bb30df0008`

The `STAGING_MANIFEST.json` retains the integrity records for the original publication-staging artifacts. Documentation and reproduction scripts added afterward are tracked by Git and are not part of that original staging manifest.

The validated evaluation does not retrain the classifier or select controller parameters using the independent test results.

## 12. Limitations

This release establishes computational reproducibility of the implemented synthetic experiment in the tested environment.

It does **not** establish:

- Physiological validity of the synthetic latent state.
- Human stress, emotion, or sensory-state measurement.
- Transferability of the synthetic motion--state mapping to actual users.
- Real-world VR deployment or clinical effectiveness.
- Physical actuator cost or measured energy savings.
- Global optimality or universal safety of the feedback policy.

The frozen estimator was trained on neutral-control trajectories; its probabilities were not calibrated, and elapsed time may encode information related to the simulated recovery schedule. These limitations constrain interpretation of the closed-loop findings.

## 13. License and citation

Original software in this repository is released under the MIT License. Third-party dependencies retain their respective licenses. The scientific manuscript, figures, and data should be cited using the corresponding research publication and archived release, as applicable.

The citation metadata is supplied in `CITATION.cff`.

A permanent Zenodo version-specific software DOI will be added after archival publication. No DOI is claimed before registration.

## 14. Contact

**Repository maintainer:** Abdul Rehman

**GitHub:** https://github.com/AbdulRehman88

For reproducibility questions, please open a GitHub issue with the operating system, Python version, reproduction command, and relevant error details.