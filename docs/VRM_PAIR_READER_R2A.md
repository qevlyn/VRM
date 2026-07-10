# VRM Pair Reader R2A

VRM Pair Reader R2A is a narrow decoder-route integration experiment on the frozen Twin dataset.

R2A freezes both branches:

- the live VRM V1.1 path from `vrm/model.py`;
- a validated R1B `PairKeyValueReader` checkpoint.

Only a new output gate trains. The gate mixes the frozen legacy distribution and frozen reader distribution:

```text
final_probs = (1 - g) * legacy_probs + g * reader_probs
```

`force_gate=0` must reproduce the V1.1 output distribution. `force_gate=1` must reproduce the R1B reader output distribution. These controls prove the endpoints before the gate is trained.

R2A tests decoder-route compatibility, not ARC reasoning. The Twin dataset is retrieval-only by construction, so raw support-output copying is expected to dominate once the reader branch is available. A passing R2A result therefore means only that a frozen correspondence reader can be causally inserted beside the legacy decoder and selected by an output gate on this marker task.

The reader-disabled condition replaces reader probabilities with a uniform distribution while keeping the learned gate, legacy path, reader alpha, and inputs unchanged. This is a causal ablation: if the output changes, the trained route depends on the reader branch.

Passing R2A does not prove ARC is solved, dynamic branch selection works on ARC, or raw support-output copying is generally sufficient. The next test should be a mixed-route or rule-transfer benchmark where direct support-output copying is insufficient and the system must choose or compose routes based on task structure.

