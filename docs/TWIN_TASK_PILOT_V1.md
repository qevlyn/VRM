# VRM V1.2A - Counterfactual Twin-Task Suite Pilot

This pilot is a synthetic causal testbed. It is not ARC-2 training, not VRM V1.2 model training, and not evidence that VRM solves ARC.

## Causal Purpose

Each base instance creates two task episodes, Twin A and Twin B. The twins share the same query grid, the same two candidate output grids in the same order, the same support-input multiset, and the same support-output multiset. They differ only in which support output is paired with which support input.

Because the correct candidate label is complementary across the two twins, any representation that cannot see input-output pair correspondence should be unable to distinguish the twins above chance.

## Formal Twin Pair

For one base instance:

```text
support inputs: x0, x1, x2
output symbols: y0, y1, y2

Twin A rule: pA, a bijection from x_i to y_j
Twin B rule: pB, a derangement relative to pA
query: q = one of x0, x1, x2
target_A = pA(q)
target_B = pB(q)
candidates = {target_A, target_B}, same order in both twins
label_A + label_B = 1
```

## Exact Invariance Guarantees

For every accepted twin pair, the generator enforces exact equality of:

- query grid;
- candidate grids and candidate order;
- canonical multiset of full support-input grids;
- canonical multiset of full support-output grids;
- support count;
- grid dimensions;
- color vocabulary visible in unpaired bags;
- non-correspondence metadata visible to probes.

The generator enforces exact difference of:

- support input-output pair mapping;
- target grid;
- correct candidate label.

Multiset equality uses deterministic full-grid hashing with multiplicity counts, not histograms or approximate equality.

## Paired Correspondence Versus Unpaired Bags

The unpaired view contains only:

```text
query
candidate_0
candidate_1
bag of support input grids
bag of support output grids
```

It does not contain pair order, pair indices, rule IDs, base IDs, twin side, target grid, or label. Twin A and Twin B must serialize byte-identically under this view.

The paired relation-aware view contains coherent support pairs:

```text
(support input, assigned support output)
```

It may preserve pair membership because pair membership is the information under test. It still excludes rule IDs, template IDs, base IDs, twin side, target grid, and label.

## Empirical Audits

Exact invariance tests are the primary proof. Probes are secondary implementation-leak detectors.

Unpaired probes P1-P4 should be at chance because their feature payloads are identical for Twin A and Twin B while labels are opposite.

The paired probe P5 should be high accuracy because it can match the query marker to its paired support output and then select the matching candidate.

The deranged paired control P6 should stay at or below the 55% sealed threshold because support outputs are deterministically deranged within each episode, preserving bags while destroying the coherent correspondence needed by P5. Depending on the derangement, the simple equality probe can become anti-correlated rather than exactly chance.

## Why This Is Not A Full ARC Benchmark

The first family, `marker_to_output_bijection_v1`, is deliberately simple. It controls query-only and unpaired-support shortcuts, but it does not model the diversity, ambiguity, compositionality, or visual complexity of ARC-2.

Passing this pilot means only that the synthetic construction appears to isolate paired correspondence under the tested generator and probes.

## Commands

Generate:

```bash
python -m vrm.twin_tasks.run_pilot generate \
  --seed 20260628 \
  --train-pairs 8000 \
  --dev-pairs 1000 \
  --sealed-pairs 1000 \
  --out /mnt/d/VRM/artifacts/twin_task_pilot_v1
```

Audit:

```bash
python -m vrm.twin_tasks.run_pilot audit \
  --data /mnt/d/VRM/artifacts/twin_task_pilot_v1 \
  --seeds 20260628 20260629 20260630
```

Full:

```bash
python -m vrm.twin_tasks.run_pilot full \
  --seed 20260628 \
  --train-pairs 8000 \
  --dev-pairs 1000 \
  --sealed-pairs 1000 \
  --out /mnt/d/VRM/artifacts/twin_task_pilot_v1
```

Tests:

```bash
pytest -q
```

## Pass/Fail Criteria

The pilot passes only if:

- 100% of accepted twin pairs pass exact invariance checks;
- cross-split base-instance overlap is zero;
- cross-split twin-pair overlap is zero;
- duplicate complete unpaired representations with non-complementary labels are zero;
- P1-P4 sealed mean accuracy is at most 55% and compatible with chance;
- P5 sealed mean accuracy is at least 90%;
- P6 sealed mean accuracy is at most 55%.

If any condition fails, the audit verdict is `FAIL` and the generator or auditor must be investigated before expansion.

## Known Limitations

- The current family uses exactly three marker identities and three output symbols.
- Output symbols share geometry and differ only by color.
- The relation-aware probe uses exact grid equality features; future families should add richer perceptual encoders.
- Passing the sealed audit does not prove all future generator families are shortcut-free.
