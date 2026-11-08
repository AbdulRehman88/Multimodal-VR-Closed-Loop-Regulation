# Multimodal VR Closed-Loop Regulation ? v1.1.0

## Evidence and scientific scope

This release contains two independently generated synthetic evaluation
cohorts, which must not be treated as the same profile population.

- Cohort A: 200 profiles, two initial conditions, three policies,
  1,200 evaluated episodes. Its original results and code remain
  preserved in the repository.
- Cohort B: 200 independently generated profiles, two initial
  conditions, nine policies (B0?B8), 3,600 evaluated episodes.
  The 200 frozen profile checkpoints are included.

The cohorts have distinct sampling provenance; neither constitutes
physical VR-user validation or cross-population equivalence.

## Numerical verification

The complete 3,600-episode cohort-B replay passed exact agreement
against the frozen checkpoints for endpoint values, control updates,
and stored state/motion hashes. See:

- FULL_CANONICAL_REPRODUCTION_AUDIT.json
- revision/configs/confirmatory_results_integrity_audit.json
- revision/results/confirmatory_200/

## Reproduction entry point

From the repository root, install requirements and run:

    python revision/run_portable_comparison.py --mode run

The verified portable runner uses repository-relative input paths.
Its output destination is results/reproduced_comparative_200/,
distinct from the frozen checkpoint directory.

The existing confirmatory runner lock binds the original historical
runner, which remains included as a provenance record.

## Important software distinction

The `revision/` folder retains some historically executed development,
audit and figure scripts. Certain historical scripts contain machine-
specific paths and are archived as provenance rather than claimed as
independently portable entry points. The supported nine-policy
reproduction entry point is run_portable_comparison.py.

The original cohort-A verification scripts are preserved from the
previous publication release. Their portability is not implied by
the cohort-B runner verification.

## Policy definitions

B0 neutral; B1 fixed schedule; B2 time-only classifier;
B3 motion-only classifier; B4 motion threshold;
B5 motion-and-time classifier without neutral-return reserve;
B6 motion-and-time classifier with reserve;
B7 privileged-state oracle; B8 time-preserved sham motion.

B5 and B6 use the same per-update and total-variation limits.
The privileged-state oracle is an upper-bound comparator,
not a deployable motion-only controller.

## Archive status

This file describes a release candidate. A Zenodo DOI must not be
claimed until the corresponding archive version is published.
