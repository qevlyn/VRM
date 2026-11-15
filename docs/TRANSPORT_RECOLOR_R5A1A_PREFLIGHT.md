# Transport Recolor R5A-1A Preflight

R5A-1A is an architecture-control and preflight audit for a generic shared-renderer control on the frozen R5A-0 V2 transport/recolor benchmark. It does not train a model, resume training, create checkpoints, evaluate sealed examples, or modify the frozen V2 generator, grammar, splits, model, checkpoints, order audit, historical results, preregistered thresholds, or future outcome categories.

This preflight does not state or assume that the shared renderer is expected to pass or fail future training. Its purpose is to make future R5A-1 results interpretable before any such training occurs.

## Implemented Files

- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a1_shared_renderer.py`
- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a1_preflight.py`
- `/mnt/d/VRM/tests/test_transport_recolor_r5a1_preflight.py`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a1a_preflight/preflight_report.json`

## Shared Renderer Family

All three variants use the same generic architecture family. Only neutral capacity knobs differ: hidden size, attention heads, encoder layers, decoder layers, and MLP ratio.

Allowed inputs and computation:

- raw support input grids;
- raw support output grids;
- raw query input grids;
- generic color and positional embeddings;
- pair/example membership embeddings;
- masks and generic canvas bounds;
- learned output-cell query tokens;
- generic self-attention and cross-attention;
- generic MLPs;
- final per-cell color logits;
- generic pooled output-size logits based only on allowed raw support/query information.

Forbidden interfaces:

- explicit source-coordinate prediction;
- explicit pointer distribution or pointer supervision;
- copy gate or source-copy routing;
- explicit semantic transport representation;
- explicit color-transition representation;
- factor-labelled latent channels;
- separate geometry or recolor heads;
- auxiliary geometry/recolor losses;
- operation, program, class, split, seed, signature, oracle, target, factor metadata, or derived equivalents.

Generic attention is allowed. It is not treated as forbidden merely because it could internally learn correspondence. The prohibition is against a hand-specified semantic transport/color interface.

## Canonical Profiling Batch

The canonical profiling batch is selected deterministically from `/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/train.jsonl` only. It sorts training rows by `episode_id` and selects the first distinct operation classes until the fixed batch size is reached. The shared renderer receives only packed support/query tensors, masks, and canvas bounds; operation labels and other metadata are used only for audit reporting.

The preflight report records stable episode IDs, tensor shapes, token counts, padding counts, tensor SHA-256, source code hash, and artifact anchors.

## Capacity Search

The capacity ledger is deterministic and analytic. It does not use accuracy, runtime, training loss, validation metrics, or sealed outcomes.

FLOP convention:

- linear and matmul: `2*M*N*K`;
- attention includes Q/K/V projections, output projection, QK^T, and attention-value products;
- MLP totals include all linear layers;
- embedding lookup arithmetic is zero;
- embedding parameters are reported separately;
- softmax, masking, normalization, activation, indexing, reshape, and comparisons are excluded from the main total.

The preflight searches discrete candidate knobs for:

- `shared_small`: lower capacity, targeted below V2 parameters and FLOPs per valid output cell where feasible;
- `shared_matched`: targeted within +/-10% V2 parameters and +/-15% V2 FLOPs per valid output cell where feasible;
- `shared_large`: targeted at 1.5x-2.5x V2 decoder-side parameter and/or forward-FLOP budget where feasible.

If no semantically valid candidate meets the matched tolerances, the report records the full ledger and uses `R5A1A_PREFLIGHT_INCONCLUSIVE` unless another hard gate fails.

## Firewall Audit

The shared renderer forward path accepts only `SharedRendererInput`, an allowlisted dataclass containing:

- `query_grids`
- `support_input_grids`
- `support_output_grids`
- `memory_token_mask`
- `output_cell_mask`
- `canvas_hw`

The audit inventories the V2 batch/example surface and marks each field as allowed, forbidden, unused, or ambiguous. It also perturbs forbidden metadata, verifies the constructed tensors are unchanged, verifies eval-mode logits are unchanged for the same model, checks direct forbidden-field rejection, and scans the shared forward/data-packing source for forbidden metadata terms.

## Commands

Run the focused tests:

```bash
cd /mnt/d/VRM
pytest -q tests/test_transport_recolor_r5a1_preflight.py
```

Run the full project tests:

```bash
cd /mnt/d/VRM
pytest -q
```

Generate the preflight report:

```bash
cd /mnt/d/VRM
python3 -m vrm.twin_tasks.transport_recolor_r5a1_preflight \
  --out artifacts/transport_recolor_r5a1a_preflight/preflight_report.json \
  --pytest-result artifacts/transport_recolor_r5a1a_preflight/pytest_result.txt
```

## Verdict Meanings

- `R5A1A_PREFLIGHT_PASS`: all hard gates pass, matched budget tolerances are satisfied, full pytest passes, no training/checkpoint/sealed evaluation occurred.
- `R5A1A_PREFLIGHT_INCONCLUSIVE`: hard gates pass but an honest pre-training capacity match or full-pytest gate is not established.
- `R5A1A_PREFLIGHT_FAIL`: a hard preflight gate fails.

This document and report do not revise the existing R5A-1A preregistration thresholds or future outcome categories.
