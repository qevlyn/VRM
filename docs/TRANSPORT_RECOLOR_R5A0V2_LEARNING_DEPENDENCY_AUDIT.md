# R5A-0 V2 Learning-Dependency and No-Oracle Audit

This is an additive audit for the frozen R5A-0 V2 transport-recolor result. It does not change the V2 model, generator, grammar, data, splits, checkpoints, losses, training code, historical artifacts, or any R5A-1A artifact.

The audit asks whether the existing V2 behavior depends on learned checkpoint weights, direct oracle/helper access, target labels, operation labels, or hidden metadata during model forward. It also records whether the current result is better interpreted as a fixed factorized scaffold rather than optimization-learned factorization.

## Added Files

- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a0v2_learning_dependency_audit.py`
- `/mnt/d/VRM/tests/test_transport_recolor_r5a0v2_learning_dependency_audit.py`
- `/mnt/d/VRM/docs/TRANSPORT_RECOLOR_R5A0V2_LEARNING_DEPENDENCY_AUDIT.md`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/learning_dependency_audit_report.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/learning_dependency_interventions.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/learning_dependency_static_scan.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/learning_dependency_dynamic_trace.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/learning_dependency_provenance.json`

## Scope

The audit reuses the frozen R5A-0 V2 train/dev/sealed suites and the frozen L0 initialization-reproduction random suite. It does not regenerate data.

It evaluates:

- selected trained checkpoints;
- five fresh initializations;
- interventions on every selected checkpoint and one fresh in-memory model;
- full trainable-parameter zeroing;
- full trainable-parameter reinitialization by copying from the model-declared fresh initialization for the requested seed;
- per-group zeroing, declared reinitialization, and deterministic perturbation;
- a read-only gradient probe on a small frozen batch;
- static source scan for oracle imports, oracle helper calls, target access, operation literals, and fixed semantic scaffold markers;
- dynamic oracle-call tracing and monkeypatching during model forward;
- separate tensor-packing alias tracing, so serialization helpers are not confused with forward-time oracle execution;
- explicit forward-input provenance.

## Parameter Groups

The audit groups trainable parameters as:

- `logit_scale_scalar`
- `embedding_lookup_tables`
- `pair_readout_latent_token`
- `relation_encoder_attention_projections`
- `relation_encoder_mlp_norms`
- `exact_pair_projector`
- `summary_pair_projector`
- `task_context_mlp`

Requested branches that do not exist in the current V2 implementation are recorded as absent rather than silently skipped.

## Provenance Boundary

The only values allowed to reach `SupportConditionedTransportRecolorR5A0V2.forward` are:

- `query_grids`
- `support_input_grids`
- `support_output_grids`

Target grids and operation/program metadata may be used after forward for loss, metrics, grouping, or audit reporting, but not as forward inputs.

The model may reuse immutable grid/color constants. Any reused constants are recorded separately from oracle helper dependencies.

Tensor packing may reuse serialization helpers such as `grid_to_list` to form raw tensors. The dynamic trace records this separately from model forward. Forward dependency conclusions are based on the already-packed tensors passed to the model.

## Commands

Run the audit:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.transport_recolor_r5a0v2_learning_dependency_audit \
  --pytest-result artifacts/transport_recolor_r5a0_v2/learning_dependency_pytest_result.txt
```

Run tests:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m pytest -q
```

## Verdicts

- `R5A0V2_LEARNING_DEPENDENCY_PASS`: trained checkpoints materially outperform fresh initialization, full reset destroys the trained solution, at least one learned group causally affects output grids, and no forbidden oracle/program/target dependency influences forward outputs.
- `R5A0V2_ORACLE_DEPENDENCY_FOUND`: target/oracle/program metadata reaches forward, an oracle helper is called during forward, or oracle-helper monkeypatching breaks forward.
- `R5A0V2_FIXED_SCAFFOLD_DOMINATES`: fresh or fully reset models retain near-trained behavior across frozen suites, provenance is valid, oracle-helper runtime dependency is not demonstrated, and protected files remain unchanged.
- `R5A0V2_LEARNING_DEPENDENCY_INCONCLUSIVE`: hard gates fail or reset/fresh behavior does not cleanly support a fixed-scaffold or learned-dependency verdict.

When the fixed-scaffold verdict is reached, the claim is limited: V2 is evidence that the hard-coded factorized scaffold solves this fixed grammar from raw support/query grids. It is not evidence that optimization learned the factorization.
