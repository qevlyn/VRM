# R5A-3B-P1 Relation Exposure Audit

R5A-3B-P1 is a strictly additive post-hoc diagnostic for R5A-3B-P0. It does not change architecture, candidate grammar, parse menu, relation menu, constraint definitions, budgets, salts, pilot manifest, or P0 reports.

The audit asks whether P0 actually exposed relation-conditioned candidates and hard constraints to a fair causal test. It reads only frozen P0 artifacts plus held-out train outputs for post-hoc target matching. It does not read ARC test outputs and does not rerun the pilot.

## Inputs

The audit reads these P0 artifacts from:

```text
/mnt/d/VRM/artifacts/r5a3b_relation_constraint_pilot/
```

- `pilot_manifest.json`
- `pre_hidden_pilot_commitment.json`
- `raw_candidate_universe_ledger.json`
- `condition_candidate_ledgers.json`
- `constraint_acquisition_report.json`
- `matched_random_prune_report.json`
- `pilot_recall_metrics.json`
- `runtime_profile.json`
- `final_report.json`

Candidate regeneration is not required if candidate provenance and output hashes are complete. P1 recomputes only parse/object-graph/relation-program-count metadata needed to distinguish exposure categories for folds without retained relation outputs.

## Outputs

Artifacts are written under:

```text
/mnt/d/VRM/artifacts/r5a3b_p1_exposure_audit/
```

Required outputs include environment reverification, P0 input integrity, ledger completeness, relation taxonomy, object-vs-relation surface, hit provenance, constraint efficacy, constraint eligibility, parse exposure, runtime exposure, representative cards, determinism verification, final report, and pytest result.

## Run

Canonical environment reverification and focused tests:

```bash
cd /mnt/d/VRM &&
/home/quyen/.venvs/vrm/bin/python -c 'import pytest, torch; print(pytest.__version__, torch.__version__)' &&
/home/quyen/.venvs/vrm/bin/python -m pytest -q tests/test_r5a3b_relation_constraint_pilot.py
```

P1 tests:

```bash
cd /mnt/d/VRM &&
/home/quyen/.venvs/vrm/bin/python -m pytest -q tests/test_r5a3b_relation_constraint_pilot.py tests/test_r5a3b_p1_exposure_audit.py
```

P1 audit:

```bash
cd /mnt/d/VRM &&
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.run_r5a3b_p1_exposure_audit
```
