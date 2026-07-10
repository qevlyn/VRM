# R5A-3B-P2A Balanced Transport Allocation

R5A-3B-P2A is a strictly additive causal rerun that repairs one allocation/exposure confound from R5A-3B-P0/P1.

P0/P1 remain valid measurements of their declared lexical allocation policy. They tested a relation grammar whose four relation slots were consumed by the first four lexical actions. P2A tests whether two previously unexposed vector-transfer actions contribute under the same final candidate budget.

## Scope

P2A does not change parse hypotheses, relation predicates, renderer semantics, final candidate budget, program depth, support-pair protocol, ARC corpus scope, or the no-test-output firewall.

It does not test:

- TTT;
- neural guidance;
- LLM proposal;
- source-target edge binding;
- deeper programs;
- new relation predicates;
- a full ARC corpus audit.

## Frozen Budgets

- `MAX_PARSE_HYPOTHESES = 4`
- `MAX_FINAL_PROGRAMS_PER_PARSE = 8`
- `MAX_FINAL_CANDIDATES_PER_FOLD = 32`
- `MAX_PROGRAM_DEPTH = 2`
- `TOP_K = [1, 4, 16, 64, 128]`

Per parse, P2A keeps four P0-equivalent object programs and four balanced relation programs.

## Balanced Relation Buckets

The four relation slots are allocated in this order:

- `RELATION_TRANSPORT_COPY`: `copy_source_by_relation_vector`
- `RELATION_TRANSPORT_TRANSLATE`: `translate_target_by_relation_vector`
- `RELATION_RECOLOR`: `recolor_target_to_source`, `recolor_source_to_target`, `recolor_target_constant`
- `RELATION_STATE_EDIT`: `delete_target`, `keep_relation_endpoints`

Each bucket may internally score up to 10 legal microproposals, and each parse may score at most 40 relation microproposals. The final emitted relation candidates remain at most four per parse.

## Run

Canonical focused tests:

```bash
cd /mnt/d/VRM &&
/home/quyen/.venvs/vrm/bin/python -m pytest -q \
tests/test_r5a3b_relation_constraint_pilot.py \
tests/test_r5a3b_p1_exposure_audit.py \
tests/test_r5a3b_p2a_balanced_transport_allocation.py
```

Pilot:

```bash
cd /mnt/d/VRM &&
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.run_r5a3b_p2a_transport_pilot
```

Artifacts are written to:

```text
/mnt/d/VRM/artifacts/r5a3b_p2a_balanced_transport_allocation/
```
