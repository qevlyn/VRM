# R5A-2A V2 Capability-Frontier Preflight

This is an additive, no-training preflight around the frozen R5A-0 V2 reference executor. It does not change V2 behavior, source, grammar, data, splits, training code, checkpoints, prior audits, R5A-1A artifacts, or any historical verdict.

The goal is to map the boundary between the declared V2 closure and four object-selection probe families that keep the normal V2 raw-grid interface:

- `largest_component_select_then_v2_transform`
- `nearest_marker_select_then_v2_transform`
- `unique_shape_select_then_v2_transform`
- `border_relation_select_then_v2_transform`

Each probe follows the same form: choose one object by an independently checked rule outside the declared V2 closure, apply an independently confirmed V2-compatible transform to that object, and leave all non-selected objects unchanged.

## Added Files

- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a2a_declared_closure.py`
- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a2a_frontier_probes.py`
- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a2a_frontier_audit.py`
- `/mnt/d/VRM/tests/test_transport_recolor_r5a2a_frontier_audit.py`
- `/mnt/d/VRM/docs/TRANSPORT_RECOLOR_R5A2A_CAPABILITY_FRONTIER_PREFLIGHT.md`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2a_frontier/probe_spec.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2a_frontier/declared_closure_report.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2a_frontier/generator_audit.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2a_frontier/v2_probe_evaluation.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2a_frontier/family_classification.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2a_frontier/final_report.json`

## Independent Declared Closure

The closure module is a handwritten audit-only transcription. It does not import the V2 model, V2 oracle helpers, model-forward helpers, or production dispatcher code.

Declared primitive semantics:

- `C`: rotate foreground cells clockwise around the `7x7` grid center.
- `H`: horizontal reflection.
- `V`: vertical reflection.
- `L`: recolor input colors `1..4` to `5..8` when current `x <= 3`.
- `K`: recolor input colors `1..4` to `5..8` when current `x <= 3 and y <= 3`.

The bounded declared program set is the audited R5A-0 V2 class table: five primitives plus eight two-step mixed programs. The finite signature domain is `7 x 7 x 9` single-cell states, lifted to collision-free whole grids under the recorded coordinate-bijection condition.

## Probe Panel

The probe panel uses 4-connected components, background `0`, marker color `8` where needed, and deterministic rejection checks for overlap, accidental component merging, selector ambiguity, transform ambiguity, closure representability, and split collision.

The V2-compatible transform programs used in the panel are:

```text
C
H
V
C->L
```

The deterministic composition split is:

- train-visible transforms: `C`, `H`
- held-out transforms: `V`, `C->L`

Every selector-family by transform cell has the same number of generated episodes.

## V2 Diagnostics

The audit evaluates frozen V2 only:

- fresh seeds `1001` through `1005`;
- one selected checkpoint;
- full-scene probe inputs;
- isolated-target probe inputs.

The isolated-target diagnostic is oracle-only. It is not an adapter, cropper, selector, or runtime pipeline. It answers only whether frozen V2 performs the transform when the selected object is already isolated inside the normal raw-grid interface.

## Commands

Run the audit:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.transport_recolor_r5a2a_frontier_audit \
  --pytest-result artifacts/transport_recolor_r5a2a_frontier/pytest_result.txt
```

Run tests:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m pytest -q
```

## Verdicts

Family classifications:

- `IN_CLOSURE`
- `OUT_OF_CLOSURE_V2_USEFUL`
- `OUT_OF_CLOSURE_V2_NOT_USEFUL`
- `OUT_OF_CLOSURE_PARTIAL_OR_AMBIGUOUS`
- `GENERATOR_INVALID`

Global verdicts:

- `R5A2A_FRONTIER_READY_FOR_TOOLBOX_EXPANSION_PREFLIGHT`
- `R5A2A_FRONTIER_BROADER_V2_BEHAVIOR_FOUND`
- `R5A2A_FRONTIER_NO_CLEAN_V2_EXTENSION_CANDIDATE`
- `R5A2A_FRONTIER_INCONCLUSIVE`

If no family reaches `OUT_OF_CLOSURE_V2_USEFUL`, the permitted next step is not to build the outer learner yet.
