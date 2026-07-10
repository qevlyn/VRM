# R5A-2C-R2 Compiled Topology Primitive Replication Audit

This additive audit tests whether the broad pattern `fixed mathematical substrate + support-consistency hypothesis selection + raw support/query grids -> exact task-specific output` can be replicated in a topology/reachability algebra distinct from V2 transport/recolor.

This audit measures only whether one additional narrow compiled specialist can
be implemented and validated under a declared finite topology hypothesis language.

It does not measure:
- the human design/search cost of discovering that hypothesis language;
- coverage of the language on real ARC tasks;
- routing among a large specialist library;
- search cost when a future library becomes large;
- ARC generality;
- zero-compute ARC solving;
- hidden-tool emergence.

## Added Files

- `/mnt/d/VRM/vrm/twin_tasks/compiled_topology_r5a2c_r2_data.py`
- `/mnt/d/VRM/vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py`
- `/mnt/d/VRM/vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py`
- `/mnt/d/VRM/vrm/twin_tasks/run_compiled_topology_r5a2c_r2_audit.py`
- `/mnt/d/VRM/tests/test_compiled_topology_r5a2c_r2.py`
- `/mnt/d/VRM/docs/COMPILED_TOPOLOGY_R5A2C_R2_AUDIT.md`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/closure_domain.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/shortcut_catalog.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/generator_audit.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/shortcut_audit.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/evaluation_report.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/ablation_report.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/final_report.json`

## Substrate

The shared core is parameter-free 4-connected fixed-point reachability:

```text
R_0 = initial_seed_mask
R_(t+1) = R_t OR (four_neighbors(R_t) AND passable_mask)
```

It is exercised under three support-conditioned conventions:

- `seed_reachable_recolor`
- `border_reachable_recolor`
- `enclosed_region_recolor`

These are one reachability primitive exercised under three support-conditioned conventions. They are not three independent primitive discoveries.

The executor receives only:

```text
support_input_grids
support_output_grids
query_input_grid
```

It enumerates complete semantic hypothesis tuples internally and requires exactly one tuple to match all three support pairs before applying that tuple to the query grid.

## Declared Domain

The finite domain is serialized to:

```text
/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/closure_domain.json
```

It records `7x7` grids, palette `0..8`, color-role domains, support/query counts, the finite hypothesis catalog, 4-connected adjacency, the wrong 8-connected ablation, and conjunction depth bound `2`.

## Shortcut Catalog

The shortcut catalog is fixed before evaluation and serialized to:

```text
/mnt/d/VRM/artifacts/transport_recolor_r5a2c_r2_topology/shortcut_catalog.json
```

It contains:

- 11 single rules
- 45 unordered pairwise conjunction rules
- 90 ordered lexicographic rules
- 146 total rules

No shortcut fitting, pruning, tuning, or mutation occurs after generation.

## Commands

Run the audit:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.run_compiled_topology_r5a2c_r2_audit \
  --pytest-result artifacts/transport_recolor_r5a2c_r2_topology/pytest_result.txt
```

Run tests:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m pytest -q
```

## Verdicts

- `R5A2C_R2_COMPILED_TOPOLOGY_PRIMITIVE_PASS`
- `R5A2C_R2_COMPILED_TOPOLOGY_PRIMITIVE_PARTIAL`
- `R5A2C_R2_COMPILED_TOPOLOGY_GENERATOR_INVALID`
- `R5A2C_R2_COMPILED_TOPOLOGY_INCONCLUSIVE`
