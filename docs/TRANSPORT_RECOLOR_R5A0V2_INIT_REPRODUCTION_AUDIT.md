# R5A-0 V2 L0 Initialization-Reproduction Audit

This is an additive no-training audit for R5A-0 V2. It checks whether a freshly initialized `SupportConditionedTransportRecolorR5A0V2().eval()` reproduces trained/checkpoint-level final-grid behavior without loading a checkpoint, running an optimizer, invoking backward, or changing V2 behavior.

The audit does not modify the V2 model, generator, grammar, data, splits, checkpoints, losses, training code, historical verdicts, R5A-1A preregistration, R5A-1A preflight, or R5A-1A matchability work.

## Added Files

- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a0v2_init_reproduction_audit.py`
- `/mnt/d/VRM/tests/test_transport_recolor_r5a0v2_init_reproduction_audit.py`
- `/mnt/d/VRM/docs/TRANSPORT_RECOLOR_R5A0V2_INIT_REPRODUCTION_AUDIT.md`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/init_reproduction_audit_report.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/init_reproduction_random_suite.json`

## Suites

Suite A uses the existing fixed V2 split files:

- `train.jsonl`
- `dev.jsonl`
- `sealed.jsonl`
- combined train/dev/sealed

Suite B is a deterministic generated audit suite independent of model outputs. It uses the existing fixed grammar to label examples, but the model forward receives only normal V2 raw input tensors. The suite covers every current operation class, all 128 query types, all primary colors, all anchor/location categories, both distractor-count settings, and two deterministic variants per class/query-type pair.

Suite C contains the pre-existing pytest sanity fixtures that instantiate a fresh model on narrow generated unit examples. It is diagnostic only and is not benchmark evidence.

## Conditions

The audit evaluates:

- `fresh_init`: five explicit initialization seeds, no checkpoint load;
- `selected_checkpoint`: each existing selected V2 checkpoint;
- `fresh_same_seed_as_checkpoint`: fresh models reconstructed with the checkpoint seeds, no checkpoint load.

For every condition/suite pair, the report records state hash before eval, optimizer-step count for this process, exact by group, exact by class, macro exact, held-out mixed exact, and prediction hash.

## Forward Provenance

The only tensors passed into `SupportConditionedTransportRecolorR5A0V2.forward` are:

- `query_grids`
- `support_input_grids`
- `support_output_grids`

Target grids and operation labels are used only after forward for metric computation and grouping. If any target/oracle/program/class metadata reaches forward, the audit verdict is provenance-invalid.

## Commands

Run the audit:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.transport_recolor_r5a0v2_init_reproduction_audit \
  --out artifacts/transport_recolor_r5a0_v2/init_reproduction_audit_report.json \
  --random-suite-out artifacts/transport_recolor_r5a0_v2/init_reproduction_random_suite.json \
  --pytest-result artifacts/transport_recolor_r5a0_v2/init_reproduction_pytest_result.txt
```

Run tests:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m pytest -q
```

## Verdicts

- `R5A0V2_INIT_REPRODUCTION_FRESH_FAILS_TRAINED_SUCCEEDS`: fresh init materially underperforms checkpoints on held-out mixed and generated random scenes.
- `R5A0V2_INIT_REPRODUCTION_FRESH_EQUALS_TRAINED`: fresh init matches selected checkpoints across suites.
- `R5A0V2_INIT_REPRODUCTION_FIXTURE_ONLY`: fresh init succeeds only on narrow sanity fixtures.
- `R5A0V2_INIT_REPRODUCTION_PROVENANCE_INVALID`: target/oracle/program/class metadata reaches model forward.
- `R5A0V2_INIT_REPRODUCTION_INCONCLUSIVE`: evaluation path, checkpoint status, determinism, or provenance cannot be established.
