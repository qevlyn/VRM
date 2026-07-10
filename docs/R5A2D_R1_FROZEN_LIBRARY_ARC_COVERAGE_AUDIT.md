# R5A-2D-R1 Final Frozen Compiled-Library ARC-Native Coverage Audit

This audit measures coverage of one frozen two-specialist library on one
precommitted local ARC corpus under strict train-pair validation.

It does not measure:
- ARC generality;
- the value of an arbitrarily large future specialist library;
- the human design/search cost of inventing new primitives;
- learned routing;
- learned program synthesis;
- test-time training;
- zero-compute ARC solving;
- performance after task-specific primitive edits;
- whether all future compiled-specialist libraries fail to scale.

A low or zero strict-LOO coverage result for this two-expert frozen library
measures only this library under this corpus and this strict support contract.

It does not establish that all future compiled-specialist libraries fail to scale,
nor does it establish that an expert which is LOO-infeasible would necessarily fail
when allowed to use the complete ARC training set.

## Frozen Library

The measured library is exactly:

- V2 transport/recolor executor, represented by the frozen R5A-0 V2 operation semantics and support-consistency over the existing `SEALED_CLASSES`.
- R5A-2C-R2 compiled topology executor, invoked through `execute_from_raw_grids` with its exact three-support-pair contract.

The identity baseline is reported separately as `output = input`.

No model is trained, no neural model is constructed, no checkpoint is read or written, and no support pair is duplicated, deleted, fabricated, cropped, padded, normalized, recolored, or otherwise adapted.

## Evidence Tiers

Full-support reproduction is exploratory only and is never counted as headline coverage.

Strict leave-one-out coverage requires every feasible train-pair fold to pass under the frozen raw interface contract. Topology support-contract infeasibility is labeled separately from executed topology prediction failure.

## Required Artifacts

The audit writes:

- `artifacts/r5a2d_r1_frozen_library_coverage/corpus_manifest_precommit.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/library_contract_report.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/task_ledger.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/coverage_profile.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/domain_attrition_report.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/strict_coverage_examples.md`
- `artifacts/r5a2d_r1_frozen_library_coverage/pre_score_commitment.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/optional_sealed_test_scoring.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/final_report.json`
- `artifacts/r5a2d_r1_frozen_library_coverage/pytest_result.txt`

Run:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.run_r5a2d_r1_frozen_library_coverage
pytest -q
```
