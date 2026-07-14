# Frozen-Checkpoint L2 Order-Causality Diagnostic

## Scope and preregistration

This diagnostic analyzes the already frozen 193-episode sealed-L2 corpus. It does not train, fine-tune, select, or alter a model; generate scenes; search over programs; or inspect any coverage-L3 episode payload. The only program intervention is reversal of two active instructions. Thresholds and the precedence-ordered verdict rules are frozen in `configs/vrm2/inductor_order_causality.yaml` before execution.

The diagnostic compares the selected `SequentialResidualInductorV1` main checkpoint with the selected shuffled-order ablation checkpoint. Both execute through the certified `OracleInterpreterV1`. The generator is restricted to model-independent subset construction and cannot enter an inductor inference path.

## Frozen evidence and artifact resolution

The evaluator verifies the separate sealed support and query hashes, their byte-concatenated corpus hash, episode count, certified-interpreter archive and state hashes, and both inductor archives. A selected inductor checkpoint is resolved from its persisted report and then matched against checkpoint contents (seed, step, model version, ablation, configuration, and parameter count); filenames alone are not accepted.

The shuffled intervention audit compares main and shuffled training configurations, embedded checkpoint configurations, frozen training source, and deterministic target transformation. It requires that every non-identical L2 target swaps its two active instruction positions, that the NULL suffix stays unchanged, and that architecture, optimizer, seed, steps, data, decoder, aggregation, and checkpoint-selection policy are scientifically matched. Sealed metrics must not select the checkpoint.

## Model-independent subset funnel

Selection is nested and uses only frozen episode contents, ground-truth typed programs, the generator executor, and certified-interpreter fidelity:

1. Start with all 193 aligned sealed-L2 episodes.
2. Retain programs whose two active typed instructions are distinct.
3. Require forward and reversed programs to be schema-valid and generator-executable on all three support inputs and the query input.
4. Require the forward query output to equal the frozen target and differ exactly from the reversed query output.
5. Require at least one reversed support output to differ from its frozen target.
6. Require the certified interpreter to match generator output for both orientations on all four inputs.

The persisted subset manifest contains episode IDs, programs, grid/pair fingerprints, disambiguating-support counts, generator outputs, and interpreter-agreement audits. It contains no checkpoint identifier or model prediction. Construction is repeated and its deterministic content is compared before the manifest is accepted. Fewer than 30 final episodes forces `ORDER_CAUSALITY_INSUFFICIENT_COVERAGE`; no replacement evidence is generated.

## Isolation boundary

Stage A accepts support episodes only and emits hard program lengths, `torch.int64` typed programs, and field confidences. It has no query parameter. Support-order permutation `[2, 0, 1]` must preserve hard programs for both checkpoints.

Stage B runs as a new Python process. Its command accepts only the frozen query path, Stage-A prediction file, certified-interpreter checkpoint, output path, device, and batch size. It does not accept or load support artifacts. It validates canonical programs and passes only a query grid and hard integer typed IDs to the interpreter. Native and position-reversed predictions are evaluated on the same query.

The support-only selector is a separate diagnostic. It compares the native two-instruction candidate with its position reversal using the three supports only, lexicographically maximizing (1) exact reconstructed support grids and (2) correct support cells. A tie remains `ORIENTATION_AMBIGUOUS`; neither ground truth nor query breaks it. A second upper-bound selector receives the ground-truth instruction multiset but not its order.

## Metrics and classifications

Every prediction is classified exactly once as ordered exact, exact reversed, correct unordered multiset with other serialization, wrong multiset, or wrong length. Program recovery, native/reversed query execution, output changes, rescues, destructions, and the four joint native/reversed outcomes are reported for all sealed L2, query-order-sensitive, support-identifiable, and final-causal subsets. Orientation rates always include the correct-multiset denominator.

On the final subset, main-minus-shuffled full-precision deltas cover multiset exact, ordered exact, conditional orientation, native exact, reversed exact, rescue, destruction, and support-selected exact. Paired query outcomes and an exact McNemar two-sided descriptive probability are included without creating a new gate.

Identity ceiling is unordered-multiset exact; ordered realization is ordered-program exact; order loss is their difference. The behavioral decomposition reports native execution, predicted-multiset support selection, ground-truth-multiset support selection, and the certified oracle-program ceiling.

## Integrity and source freeze

The implementation records actual resolved paths and denies the three configured coverage-L3 payload paths. The audit must show zero opens. Before execution, the complete VRM test suite passes and source/config/document/test hashes are written to `source_manifest.json`. Every execution command verifies that manifest. No source may change until the post-run verification, and all three checkpoint hashes are rechecked.

Diagnostic runtime files are additive under `artifacts/vrm2/inductor/runtime/order_causality/`. Existing runtime files are rejected rather than overwritten. No training command is part of this protocol.

## Frozen verdict precedence

1. Structural, hash, boundary, shuffle, source, or L3-isolation failure: invalid (with the specific source-changed or shuffle-intervention verdict where applicable).
2. Final subset below 30: insufficient coverage.
3. Shuffled exceeds main by at least 0.05 in both native query exact and conditional correct orientation: mechanism contradicted.
4. For either checkpoint, order loss at least 0.15 and selector gain at least 0.10: order-decoding bottleneck supported.
5. For both checkpoints, identity ceiling below 0.50 and selector gain below 0.10: instruction-identity bottleneck supported.
6. Otherwise: mixed.

No result from this diagnostic authorizes another experiment automatically.
