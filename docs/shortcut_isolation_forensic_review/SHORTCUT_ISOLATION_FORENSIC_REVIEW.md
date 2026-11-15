# Shortcut-Isolation Forensic Review

## Executive conclusion

The current VRM/R6A/CompressARC program has not yet isolated a shortcut mechanism because it has not run an A/B/C causal shortcut experiment. Phase 1 establishes a real behavioral fact: query probabilities are sensitive when support outputs are swapped away from their original support inputs. But that intervention breaks the legitimate support input-output relation rather than preserving it while selectively destroying a declared nuisance route. The Phase 2 line then studies support-order equivariance and internal forward provenance under zero-noise, fresh-state conditions; it does not rerun loss, backward, optimizer, learned/adapted query behavior, or a certified production decoder. Therefore no existing experiment simultaneously satisfies semantic preservation, selective nuisance destruction, and behavioral causal effect.

The strongest current statement is: R6A has evidence of pairing sensitivity, evidence that v8's state action was invalid, evidence that v10 repairs the support-order state action under a narrow forward protocol, and evidence that v11.2 can reproduce byte/threshold trace markers. It does not have evidence that Cummax, Channel Layer, VRM, or any other route is a shortcut carrier, and it does not establish reasoning.

## Exact definition of shortcut isolation used in this review

A candidate mechanism can be called an isolated shortcut family only when all three conditions are met:

- A. Semantic preservation: the legitimate task-relevant relation R remains available to a correct solver.
- B. Selective nuisance destruction: a finite hypothesized nuisance route N, belonging to a declared shortcut family S, is broken or counterbalanced without removing R.
- C. Behavioral causal effect: the intervention changes an end-to-end task-relevant outcome such as query prediction, support reconstruction, optimization trajectory, final loss, discrete decoded output, or calibrated task-level score.

Internal tensor hashes, byte mismatches, ULP counts, or threshold events can support provenance hypotheses, but they are not by themselves shortcut isolation.

## Repository and evidence scope reviewed

Reviewed file inventory is in `docs/shortcut_isolation_forensic_review/reviewed_file_inventory.csv`.

- Files reviewed: 66.
- Source/test files reviewed: 23.
- Reports/manifests/artifact files reviewed: 43.
- Primary source scope: `r6a_compressarc/scripts/`, `third_party/compressarc_reference/`, selected `vrm/twin_tasks/`, and relevant tests.
- Primary evidence scope: Phase 1 reports/manifests/artifacts; v8, v10, and v11.2 artifacts; import-provenance manifests; earlier archive/docs context.

The generated Phase 2 Markdown reports are very large, so this review uses their underlying JSON/CSV artifacts and source code as primary evidence where possible. This avoids treating rendered report duplication as independent evidence.

## Claim-evidence reconstruction

The complete claim matrix is in `claim_evidence_matrix.csv`.

Key reconstruction:

- Phase 1 uses `select_swap` to choose output derangements by padded Hamming distance (`r6a_compressarc/scripts/run_phase1_multirepeat_pairing_audit.py:56-73`) and writes swapped challenges through `challenge_for_condition` (`r6a_compressarc/scripts/run_phase1_multirepeat_pairing_audit.py:200-203`; `r6a_compressarc/scripts/r6a_common.py:121-130`). It runs actual training steps (`r6a_compressarc/scripts/phase1_condition_solver.py:195-200`) and captures query probability tensors after training (`phase1_condition_solver.py:206-248`). This supports `PAIRING_SENSITIVE_QUERY_PILOT`, but not shortcut isolation.
- v8 attempted a support-order state action and traced forward divergences (`r6a_compressarc/scripts/r6a_phase2_support_permutation_first_divergence_v8_child.py:780-847`). v10 later showed v8 had applied the permutation to nonposterior tensors (`r6a_compressarc/artifacts/r6a_phase2_semantic_state_action_decoder_boundary_v10/v8_action_behavior_audit.json:status`). That invalidates the original AFFINE localization.
- v10 repairs the state action with a closed semantic table: only posterior mean/local-capacity leaves with `dims[0] == 1` are permuted (`r6a_compressarc/scripts/r6a_phase2_semantic_state_action_decoder_boundary_v10_child.py:277-280,382-389`). It verifies task action, collision witnesses, and affine commutation, then compares zero-noise forward traces (`r6a_phase2_semantic_state_action_decoder_boundary_v10_child.py:760-846`). Its own final outcome says the remaining mismatch is not evidence of a shortcut carrier, architecture route, or reasoning mechanism (`r6a_compressarc/artifacts/r6a_phase2_semantic_state_action_decoder_boundary_v10/final_outcome_classification.json`).
- v11.2 defines E_hash/E_threshold from event-point comparisons (`r6a_compressarc/scripts/r6a_phase2_two_pass_semantic_boundary_provenance_v11_2_child.py:629-682`) and E_bit through targeted replay (`r6a_phase2_two_pass_semantic_boundary_provenance_v11_2_child.py:692-708`). Its summary reports E_bit at `channel_layer[occurrence=1,dims=[1,1,0,0,0]]` for all seven cases, while E_threshold at `cummax[(1,1,1,1,1)]` occurs only for the three `604001fa` adjacent swaps. This is provenance evidence, not shortcut isolation.

## Actual codepath reconstruction

Structured codepath map is in `codepath_dependency_map.json`.

The production CompressARC path is:

```text
raw support pairs -> preprocessing.Task -> train.take_step objective/optimization -> posterior state -> ARCCompressor.forward -> logits/x_mask/y_mask -> solution_selection/evaluator
```

Source-grounded details:

- `preprocessing.Task` records `n_train`, `n_test`, shapes, masks, and `problem` tensors (`third_party/compressarc_reference/preprocessing.py:15-26,95-116,143-156`).
- `ARCCompressor.forward` starts from internally stored `multiposteriors`, `target_capacities`, and `decode_weights`; it then applies share, softmax, cummax, shift, direction, nonlinear, normalize, and final heads (`third_party/compressarc_reference/arc_compressor.py:94-158`).
- `channel_layer` samples or zero-replays latent channel values and returns KL (`third_party/compressarc_reference/layers.py:71-131`).
- `cummax` uses masks and directional geometry (`third_party/compressarc_reference/layers.py:418-452`).
- `train.take_step` is the actual loss/backward/optimizer path: it calls `model.forward`, computes reconstruction error over visible train inputs/outputs and test inputs, forms `loss`, calls `loss.backward()`, and then `optimizer.step()` (`third_party/compressarc_reference/train.py:49-108`).
- `solution_selection.Logger` tracks query logits and postprocesses predictions (`third_party/compressarc_reference/solution_selection.py:40-76,130-162`).

For Phase 2 forward audits:

- v8/v10/v11.2 zero-noise forward scans do not directly consume raw support input grids or raw support output grids inside `model.forward`. Raw task information has already entered through `Task`, masks/shapes/colors, and the initialized posterior state.
- v10 and v11.2 do not execute loss, backward, optimizer, or task adaptation. This is asserted in tests (`r6a_compressarc/tests/test_r6a_phase2_semantic_state_action_decoder_boundary_v10.py:163-168`; `r6a_compressarc/tests/test_r6a_phase2_two_pass_semantic_boundary_provenance_v11_2.py:174-182`) and in v11.2 `forbidden_path_certificate.json`.
- v11.2 explicitly states the decoder is not in forward scope: `AUDIT_DECODER_NOT_IN_FORWARD_SCOPE`, `E_discrete=decoder-out-of-scope`, and `audited_discrete_behavior_changed=false` (`r6a_compressarc/scripts/r6a_phase2_two_pass_semantic_boundary_provenance_v11_2_child.py:821-832`).

## Intervention semantics matrix

The intervention matrix is in `intervention_semantics_matrix.csv`.

The decisive distinction is that Phase 1 measures a behavioral effect but breaks the support relation, while Phase 2 preserves support-order semantics but mainly reads internal forward tensors. Neither combines preserved R, selectively broken N, and behavioral C.

## Why Phase 1 establishes pairing sensitivity but not shortcut isolation

Phase 1 is the only R6A line here that runs actual optimization and measures query behavior. The solver loads identical initial model/optimizer/RNG state per task/seed (`r6a_compressarc/scripts/phase1_condition_solver.py:176-184`), runs `train.take_step` for 1500 steps (`phase1_condition_solver.py:195-200`), saves final-100 query probabilities (`phase1_condition_solver.py:206-248`), and then computes evaluator-side NLL after solver completion (`r6a_compressarc/tests/test_r6a_phase1_multirepeat.py:180-186`).

However, the swap intervention writes a new challenge whose support outputs are assigned to different support inputs (`r6a_compressarc/scripts/r6a_common.py:121-130`). Tests verify the support inputs stay the same and output multiset stays the same (`r6a_compressarc/tests/test_r6a_phase1_multirepeat.py:110-123`), but that is not semantic preservation of the original relation. A correct solver should generally care which output belongs to which input. The Phase 1 result is therefore pairing sensitivity, not isolated shortcut dependence.

## Why v8 was invalidated

v8's support-order action was intended to permute example-axis state so that order changes could be compared after inverse alignment. But v10 found that v8's implementation applied the permutation to nonposterior tensors when a physical leading dimension happened to match the example count. The v10 audit records `V8_NONPOSTERIOR_PERMUTATION_OBSERVED` and lists `head_weights` and `mask_weights` as affected (`r6a_compressarc/artifacts/r6a_phase2_semantic_state_action_decoder_boundary_v10/v8_action_behavior_audit.json`). The source for that audit is `r6a_compressarc/scripts/r6a_phase2_semantic_state_action_decoder_boundary_v10_child.py:709-740`.

As a result, v8's original `FIRST_DIVERGENCE_AFFINE` and `FIRST_DIVERGENCE_NORMALIZE` labels do not validly localize production support-order behavior. They localize an implementation artifact in the state action.

## What v10 does and does not establish

v10 establishes a narrower and cleaner support-order result. It repairs the state action by permuting only eligible posterior mean/local-capacity tensors (`r6a_compressarc/scripts/r6a_phase2_semantic_state_action_decoder_boundary_v10_child.py:277-280,382-389`). It checks task action (`392-418`), direct affine commutation (`462-491`), trace comparison (`665-706`), and case execution (`760-846`). The summary reports task action pass/fail `[7,0]`, semantic state action `[7,0]`, affine certificate `[7,0]`, and collision witnesses `[1024,0]`.

What it does not establish: no shortcut family is declared, no nuisance is selectively destroyed, and no end-to-end training/optimization/query metric is measured. The tests explicitly guard against loss/backward/Adam/retraining/VRM paths (`r6a_compressarc/tests/test_r6a_phase2_semantic_state_action_decoder_boundary_v10.py:163-168`). Therefore `FIRST_DIVERGENCE_CUMMAX` is an internal forward mismatch label, not a shortcut carrier claim.

## What v11.2 does and does not establish

v11.2 improves provenance discipline by separating E_hash, E_bit, and E_threshold. E_hash is the first aggregate hash mismatch in pass 1; E_threshold is the first comparator over-tolerance event; E_bit is a targeted replay confirmation of exact byte mismatch (`r6a_compressarc/scripts/r6a_phase2_two_pass_semantic_boundary_provenance_v11_2_child.py:629-708`).

The result must be read carefully:

- All seven cases have E_bit at the same Channel Layer event. Because this includes controls and synthetic cases, E_bit is a baseline byte-instability or trace-alignment/capture marker, not a shortcut localization marker.
- Only `604001fa` adjacent swaps have E_threshold at Cummax. This may indicate case-specific amplification, a structural source-path candidate, alignment residue, or numerical sensitivity. It does not by itself prove structural source causality or shortcut use.
- The selective numerical oracle is limited to the replay cone and says `classification_rule_satisfied=false` (`r6a_compressarc/artifacts/r6a_phase2_two_pass_semantic_boundary_provenance_v11_2/selective_numerical_oracle.json`).
- The decoder scope certificate prevents final-grid claims (`r6a_compressarc/scripts/r6a_phase2_two_pass_semantic_boundary_provenance_v11_2_child.py:821-832`; tests at `r6a_compressarc/tests/test_r6a_phase2_two_pass_semantic_boundary_provenance_v11_2.py:152-155`).

Thus v11.2 establishes reproducible internal trace markers and useful candidate narrowing, but not shortcut isolation.

## Every identified blocker to shortcut isolation

The full gap analysis is in `shortcut_isolation_gap_analysis.json`.

1. No operational shortcut family: current use of shortcut is broad and verbal; no finite S and nuisance N are declared before intervention.
2. Intervention changes the wrong object: Phase 1 swaps support outputs, while Phase 2 permutes order/state conventions.
3. No semantic-preserving counterfactual: there is no paired C_preserve/C_break where R is proven intact and only N is broken.
4. No end-to-end behavioral readout in Phase 2: v10/v11.2 stop at forward tensors, labels, byte markers, and threshold events.
5. State/action confounding: v8 proves this risk concretely; v10 repairs the gross form but v11.2 still leaves alignment/capture/numerical explanations open.
6. Numerical baseline contamination: E_bit at Channel Layer in all seven cases means byte identity is too sensitive for shortcut localization.
7. No necessity/sufficiency test: no candidate route has been selectively blocked or substituted while preserving legitimate evidence.
8. No substitution test: no experiment asks whether a blocked shortcut family is replaced by another nuisance route.

## Classification of each blocker

- Conceptual: no finite shortcut family S; no declared nuisance variable N.
- Semantic-intervention: Phase 1 breaks the input-output relation; several support ablations in R5A remove legitimate evidence.
- Implementation: v8 nonposterior permutation invalidated its original localization.
- Numerical: Channel Layer E_bit in controls and tiny replay differences make byte identity an unstable causal criterion.
- Measurement: Phase 2 no loss/backward/optimizer/decoder/query metric.
- Causal-design: no A/B/C design, necessity test, sufficiency test, or substitution test.

## Minimal valid shortcut-isolation experiment

The minimal design is specified in `minimal_valid_shortcut_experiment_spec.md`.

In one sentence: define a finite shortcut family S and a legitimate relation R on tiny synthetic ARC-like tasks, build matched C_preserve/C_break conditions where R remains intact but nuisance N is counterbalanced, and require a predeclared change in held-out query exact/NLL plus preservation, necessity, and substitution tests.

## Explicit non-claims

This review does not claim that Cummax is a shortcut carrier. It does not claim that Channel Layer is a shortcut carrier. It does not claim that adding VRM is indicated. It does not claim that CompressARC is reasoning or not reasoning. It does not identify a new suspect layer. It does not rerun or reinterpret Phase 1 as a shortcut proof.

## Exact next decision gate

Before any layer lock, architecture change, or broad provenance scan, the next decision gate should be: can the project preregister S, R, N, C_preserve, C_break, and a behavioral outcome such that A/B/C can be scored before seeing results? If not, further trace localization will remain uninterpretable for shortcut isolation.

## WHAT IS ESTABLISHED

- Phase 1 establishes pairing-sensitive query behavior under the frozen 1500-step CompressARC protocol.
- v8's original state-action localization is invalidated by nonposterior permutation.
- v10 repairs the support-order semantic state action and narrows some forward mismatches under CPU float64 zero-noise conditions.
- v11.2 separates byte/hash/threshold markers and shows all seven E_bit markers at Channel Layer, with 604-only E_threshold markers at Cummax.
- Existing Phase 2 audits deliberately do not run loss, backward, optimizer, retraining, VRM, or architecture changes.

## WHAT IS NOT ESTABLISHED

- No existing result isolates a shortcut family.
- No existing result proves Cummax, Channel Layer, or any route is a shortcut carrier.
- No existing result proves reasoning.
- No existing result shows a behavioral causal effect from a semantic-preserving nuisance-only intervention.
- No existing result shows necessity, sufficiency, or substitution for a candidate shortcut route.

## WHY SHORTCUT ISOLATION FAILED

Shortcut isolation failed because the program split the required causal triad across different evidence types. Phase 1 has behavioral C but lacks A and B. v10/v11.2 have a semantic-preserving support-order setup for A but lack a declared nuisance family B and lack behavioral C. v8 additionally failed implementation validity. The result is a set of useful but non-composable facts: pairing sensitivity, support-order equivariance/provenance, and numerical/internal drift are not the same as shortcut dependence.

## MINIMUM EXPERIMENT NEEDED NEXT

Run a tiny preregistered A/B/C experiment with a finite shortcut family S, a legitimate relation R, and a nuisance N where C_preserve aligns R and N, C_break preserves R but breaks/counterbalances N, and success requires a predeclared held-out query exact/NLL effect plus preservation, necessity, and substitution checks.

## WHAT MUST NOT BE DONE YET

- Locking Cummax: not supported. Current evidence shows an internal 604-only threshold marker, not necessity or behavioral causality.
- Locking Channel Layer: not supported. E_bit appears at Channel Layer in controls and targets, making it a poor shortcut-localization marker.
- Adding VRM: not supported by the R6A shortcut-isolation evidence; VRM is a separate architecture line.
- Claiming reasoning: not supported; none of the evidence separates legitimate rule use from shortcut use.
- Claiming a shortcut carrier: not supported; no A/B/C shortcut family has been isolated.
- Running another broad provenance scan: not the next decision gate. Without a finite S/R/N causal design, more trace boundaries will still be difficult to interpret.

# CLAUDE CODE INDEPENDENT ADDENDUM

This addendum was produced by Claude Code as an independent forensic pass over the same repository. The analysis below was derived from source, scripts, tests, manifests, and artifacts **before** reading the Codex conclusions; each principal Codex conclusion was then audited against that independent evidence base. Nothing above this heading was modified. No experiment, test suite, training, inference, benchmark, audit, Git command, upload, or archive operation was executed; the only filesystem write in this session is this appended section.

Independent evidence base actually read for this addendum: 52 repository files, comprising 14 source/script files read line-by-line (`r6a_compressarc/scripts/phase1_condition_solver.py`, `run_phase1_multirepeat_pairing_audit.py`, `r6a_common.py`, the v8/v10/v11.2 child scripts, `third_party/compressarc_reference/{arc_compressor,layers,preprocessing,train,solution_selection}.py`, `vrm/model.py` header, `vrm/twin_tasks/r5a4b_support_semantics_audit.py` conditions/metrics), 3 test files, 10 reports, 5 manifests, 18 artifact JSON/CSV files (including `state_action_certificate.json`, `semantic_action_table.csv` aggregations, `first_divergence_trace.csv` rows, `cummax_microtrace.json`, `pass1_candidate_markers.json`, the v4 closure summary, and the symmetry-preflight S-cycle-invariance section), plus the two ARC dataset JSONs for task-shape verification.

## Verification of the principal Codex conclusions

**C1. "No existing experiment simultaneously satisfies A/B/C; therefore no shortcut is isolated" — CONFIRMED.**
Independent scoring agrees case by case. Phase 1 has C but fails A and B (the `ALL_PAIRING_SWAP` derangement destroys the input→output correspondence itself; `r6a_compressarc/scripts/r6a_common.py:121-130`). v10/v11.2 have A (task action and semantic state action certified) but fail B (no declared nuisance family — support order was never hypothesized as the mechanism behind the Phase 1 effect) and fail C (`audited_discrete_behavior_changed` is `false` for all seven cases in `artifacts/r6a_phase2_two_pass_semantic_boundary_provenance_v11_2/child_summary.json`; no loss/backward/optimizer per `forbidden_path_certificate.json`). v8 fails on implementation validity. No composite exists.

**C2. "Phase 1 establishes pairing-sensitive query behavior" — PARTIALLY_CONFIRMED, with a required stratification Codex does not state.**
The verdict machinery is sound (`run_phase1_multirepeat_pairing_audit.py:316-377`; 81/81 runs; Stratum A mean delta 1.6555, LOO all positive, `reports/r6a_phase1_decision_packet.md:12-104`). However, `reports/r6a_phase2_symmetry_preflight_report.md:14-134` proves that for `604001fa` the S condition also flips `in_out_same_size` true→false, changes `task.masks`, `task.shapes`, and the predicted query output shape, and switches every loss branch to `grid_size_uncertain` (`preprocessing.py:44-62,143-156`; `train.py:68-70`). Independent verification against the dataset JSONs shows S provably preserves all per-example shapes for only **3 of 6** Stratum A tasks (`3ed85e70`, `929ab4e9`, `981571dc` — all-outputs-same-shape) and disturbs the shape frame for `604001fa`, `79fb03f4`, `aa4ec2a5` (and Stratum B `1c786137`). So the honest claim is: *pairing-content sensitivity is cleanly established on the three frame-clean Stratum A tasks; on the other three, the measured delta conflates content-pairing destruction with shape-frame destruction.* The conclusion survives the confound — leave-one-task-out means remain positive (e.g., removing `604001fa` leaves mean delta 1.8189) and all six per-task means are positive — but the report-level verdict name `PAIRING_SENSITIVE_QUERY_PILOT` is broader than what is uniformly measured.

**C3. "The Phase 1 swap breaks the legitimate relation R" — CONFIRMED.**
The derangement re-assigns outputs to wrong inputs while `tests/test_r6a_phase1_multirepeat.py:110-123` verifies inputs and the output *multiset* are preserved. A correct solver is not invariant to this. One additional implication Codex leaves implicit: because S preserves the unpaired output multiset, a solver relying **only** on unpaired output-marginal statistics, or only on per-example memorization with pairing-independent query prediction, would show delta ≈ 0. The robust positive delta on frame-clean tasks therefore already *excludes* those two degenerate families as the sole account of query NLL — a narrow exclusion result, not an isolation result.

**C4. "v8 was invalidated because its state action permuted nonposterior tensors" — CONFIRMED in conclusion, CONTRADICTED in the cited evidence.**
The Codex review (main report §"Why v8 was invalidated"; `claim_evidence_matrix.csv` row 3; `shortcut_isolation_gap_analysis.json` blocker 2) grounds the invalidation in `artifacts/r6a_phase2_semantic_state_action_decoder_boundary_v10/v8_action_behavior_audit.json` and its `head_weights`/`mask_weights` records. That artifact is itself unreliable: v8 wrote `forward_reachable_state_inventory.csv` with naive comma-joins (`v8_child.py:1369-1395`), so fields such as `shape="[16, 2]"` contain unquoted commas; v10's `csv.DictReader` re-parse (`v10_child.py:709-741`) reads column-shifted values, and the `true` it attributes to `has_real_example_axis` for `head_weights/0` is actually the shifted `requires_grad` column (the truncated `physical_shape: "[16"` in the audit records is the tell). Positional parsing of the same CSV rows shows `head_weights/0` (shape `[16, 2]`, 16 ≠ n_examples) has `has_real_example_axis=false`; v8 did **not** permute head/mask weights.
The invalidation nevertheless stands on stronger, parse-independent evidence: v8's own `state_action_certificate.json` reports `example_axis_tensor_count = 232` for the n_examples=4 cases (`1f876c06`, synthetic oracle) versus the semantically correct **30** posterior leaves that v10 permutes (`semantic_state_action_certificate.json`, `permuted_alias_class_count=30` for every case) — i.e., ~202 non-posterior weight alias classes were wrongly permuted wherever a weight's leading dimension collided with n_examples=4 (the model's `cummax_dim = shift_dim = decoding_dim = 4`, `arc_compressor.py:20-27`). v10's `semantic_action_table.csv` counts 324 `accidental_leading_dim_equality` structural paths per n=4 case and **0** for `604001fa` (n_examples=5). The magnitude signature confirms corruption: `1f876c06_adjacent_1_2` first fails at event 3 with `max_abs = 3.539` and 448 over-tolerance cells (`first_divergence_trace.csv`) — weight-corruption scale, eleven orders above the ~1e-11 drift elsewhere.

**C5. "v8's AFFINE/NORMALIZE labels do not validly localize production behavior" — CONFIRMED, with a sharper account for 604001fa.**
For `604001fa` (n_examples=5, zero accidental collisions) v8's state action was actually **correct** (232 vs 30 discrepancy does not apply; its certificate shows 30). Its failing events (index ~888) have `max_abs = 3.5327e-11 / 1.7986e-10 / 7.0654e-11`, which match v10's `FIRST_DIVERGENCE_CUMMAX` magnitudes (`3.5328e-11 / 1.4131e-10 / 7.0653e-11`, `decoder_boundary_trace.json`, event 50) to four significant figures for the two directly comparable cases. v8 and v10 detected the **same numerical drift**; the label moved from AFFINE/NORMALIZE to CUMMAX only because v8 wrapped `affine` and `normalize` as boundary events (inner residual-block calls complete first), while v10's tracer does not. v8's single label vocabulary thus collapsed two utterly different phenomena — genuine harness-induced weight corruption (max_abs 3.54) and ~1e-10 numerical drift — under the same `FIRST_DIVERGENCE_AFFINE` name. A further note: the synthetic oracle was structurally blind to the v8 weight-permutation bug, because `apply_synthetic_fingerprints` (`v8_child.py:1004-1020`) writes patterns constant across channel dimensions, making a weight-row permutation mathematically invisible except through summation order (hence the oracle's failure registered only 9.57e-11, not catastrophe).

**C6. "v10/v11.2 are forward/provenance audits only — no loss, backward, optimizer, adaptation, or certified decoder" — CONFIRMED, and strengthened.**
Beyond the guards Codex cites, there is a structural point that makes the disconnect categorical: in every Phase 2 case the audited model is a **fresh, untrained initialization** (`reset_determinism(); ARCCompressor(task)`, `v10_child.py:767-770`; same in v8/v11.2). In CompressARC, pairing information reaches the posterior state **only through gradient descent on the reconstruction loss** (`train.py:50-108`; `forward()` reads no grids — its only task couplings are `task.masks`, `task.shapes`, and tensor sizes, `arc_compressor.py:94-158`, `layers.py:542-570`). At step 0 no gradient has flowed, so the audited forward has *never seen the pairing*. Phase 2 therefore could not have observed pairing-dependent computation even in principle; it audits a different causal object than the Phase 1 phenomenon. Additionally, the audited regime is CPU float64 zero-noise, while production is float32 CUDA stochastic (`arc_compressor.py:8-11`; `phase1_condition_solver.py:169-170`) — the equivariance certificate does not automatically transfer.

**C7. "E_bit at channel_layer in all seven cases means byte identity is a poor localization criterion" — CONFIRMED, with the mechanism named.**
Codex calls control-case E_bit "baseline byte-instability or trace-alignment/capture marker". The mechanism is identifiable from source: `channel_layer` and `normalize` reduce with `torch.mean` over `all_but_last_dim`, which **includes the example axis** (`layers.py:84,106-108,120-121,27-30`). The semantic state action physically reorders rows along that axis, so float summation order changes and last-ULP drift at the first example-carrying leaf (`dims=[1,1,0,0,0]`, occurrence=1) is *expected under the null hypothesis of perfect semantic equivariance*. E_bit consequently fires identically in controls and targets (zero discriminative power), and the label `TRACE_ALIGNMENT_OR_CAPTURE_CANDIDATE` points the wrong way: scanner transparency (3/3 PASS) and mutation calibration (4/4 PASS) disfavor capture faults, while the label vocabulary (`v11_2_child.py:904-911`) simply lacks a "numerical reduction-order" branch.

**C8. "E_threshold at Cummax only for 604001fa is suggestive but insufficient; possibly amplification" — CONFIRMED, and the repository's own microtrace goes further than 'possibly'.**
`cummax_microtrace.json` records `first_internal_byte_mismatch` at `down_affine_input`, sequence_index 1, for all three `604001fa` cases — the drift **pre-exists** the cummax computation (the block's input is already byte-different), and the artifact's own fields state `raw_cummax_structural_label_allowed=false` and `numerical_reduction_explanation_ruled_out=false`. `cummax_`/`diagonal_cummax_` are selection-and-rescale operations (`layers.py:418-446`) mathematically capable of discontinuously amplifying sub-tolerance input drift near ties. `604001fa` is also the only shape-heterogeneous audited task and carries the largest `(1,1,1,1,1)` tensor (614,400 elements; over-tolerance counts 2/7/5, max_abs ≤ 1.4130918657428992e-10), so higher near-tie density is an uncontrolled task-level covariate. On the question "can channel_layer/Cummax evidence say anything more about shortcuts?": with current data, **no** — its only defensible extra content is a *negative*: the untrained forward path is order-equivariant to ≤1.4e-10 with unchanged discrete outputs, which deprioritizes forward-path order-sensitivity as a shortcut locus and points the search at the training loop.

**C9. "AUDIT_DECODER_NOT_IN_FORWARD_SCOPE limits discrete claims" — CONFIRMED.**
With one completeness note: Phase 2 did include discrete-format readouts — v8's forward readout ran the production postprocessor (`predicted_grid_status`, `v8_child.py:771-777`, compared at `compare_forward_readouts:841-842`), and v11.2's `decoder_scope_certificate` computed `color_logit_argmax_equal` and mask-sign equality per case (`v11_2_child.py:821-832`). All were unchanged everywhere. So Codex missed no behavioral causal readout that could rescue a shortcut claim; the only behavioral data in Phase 2 are these null discrete checks at an untrained state, plus v4's `cycle_invariant_effects.md` reanalysis of frozen Phase 1 ledgers, which is self-labeled `EXPLORATORY_FROZEN_LEDGER_REANALYSIS` (5 tasks, median pairing delta > noise delta) and carries no new conditions.

**C10. "R5A-4B is a separate model line and does not retroactively isolate an R6A shortcut" — CONFIRMED.**
Verified against `vrm/twin_tasks/r5a4b_support_semantics_audit.py:46-70` (conditions `TRAINED_FULL`, `QUERY_ONLY`, `PAIRING_BROKEN`, `INPUTS_ZEROED`, `GEOMETRY_SCRAMBLED`, `ZEROED_OUTPUTS`, `FRESH`) and the behavioral metric block (~lines 495-560: `dense_cell_accuracy`, `dense_color_nll`, `exact_full_grid_correct`). These are behavioral condition contrasts on the neural VRM model, and the destructive conditions again remove legitimate evidence rather than preserving R while breaking a declared N.

**C11. Codex's blocker list (8 items) and classifications — CONFIRMED, one scoring nuance.**
All eight blockers reproduce independently. Nuance: `shortcut_isolation_gap_analysis.json` scores v8's condition A as `PARTIAL`; as-executed it should be `FAIL` — an invalid state action means the intervention actually applied did not preserve semantics, whatever the intent (task action alone passed).

**C12. Codex's minimal experiment spec (S = output-bag/histogram family on synthetic tasks with known R) — PARTIALLY_CONFIRMED as semantically valid and tiny-model executable, with two mandatory pins and one cheaper repo-native variant.**
Valid: the family is finite, N is semantically irrelevant by construction, C_preserve/C_break are well-defined, and behavioral outcomes are predeclared. Mandatory pins for CompressARC execution: (1) **color-set invariance** — `preprocessing.Task` builds `n_colors` from the union of observed colors (`preprocessing.py:80-89`) and the multitensor shapes depend on it (`preprocessing.py:91-93`), so C_preserve and C_break must use the identical color *set* (only frequencies/assignments may differ) or the two conditions instantiate different architectures; (2) **shape-frame invariance** — all conditions must hold per-example shapes and the `in_out_same_size`/`all_out_same_size` flags fixed, or the C2 confound re-enters. Executability: on CompressARC each condition costs one 1500-step per-task optimization (~36 min/run at Phase 1 rates — bounded but not free); on the `vrm/` tiny-model line the same design is cheaper and the R5A4B harness already computes the needed behavioral metrics. Cheaper first variant: see MINIMUM NEXT DECISION GATE below — a support-ORDER family test needs no synthetic task construction at all.

## Material findings absent from the Codex review

1. **Frame-confound stratification of Phase 1** (C2): S is shape-frame-preserving for exactly 3/6 Stratum A tasks; the symmetry preflight already proved the `604001fa` frame flip; per-task means and LOO keep the conclusion alive on frame-clean tasks. Evidence: `reports/r6a_phase2_symmetry_preflight_report.md:14-134`; dataset shape verification; `reports/r6a_phase1_decision_packet.md:12-64`.
2. **Exclusion result hiding in Phase 1** (C3): multiset-preserving S makes the positive delta already rule out pure output-marginal and pure per-example-memorization accounts on frame-clean tasks.
3. **Quantified v8 invalidation** (C4): 232-vs-30 permuted alias classes; 324-vs-0 accidental collision paths; channel-dimension collision `cummax_dim=shift_dim=decoding_dim=4` with n_examples=4; catastrophic (3.54) vs drift-scale (1e-10) magnitudes under one label.
4. **The v8 audit artifact is itself contaminated** (C4): `v8_action_behavior_audit.json` rests on a comma-shifted CSV re-parse; its named tensors are wrong even though its status conclusion is right. Any future citation should use the count evidence instead.
5. **v8/v10 magnitude identity for 604001fa** (C5): same drift, different boundary vocabulary; v8's state action was correct for that task.
6. **Reduction-order mechanism for E_bit** (C7): `layers.py:106-108,120-121` makes control-case bit drift expected; the v11.2 label vocabulary cannot express this.
7. **Microtrace pre-existence** (C8): the drift enters the cummax block already byte-different (`down_affine_input`, seq 1); the audit's own fields forbid the structural-cummax reading.
8. **Step-0 causal disconnect** (C6): Phase 2 audits a state that has never received pairing information; forward() consumes no raw grids at any step.
9. **Unused B2 machinery**: a shape-preserving pairing derangement (`select_derangement(shape_preserving=True)`, `r6a_common.py:71-118`, statuses `CONTENT_PAIRING_ELIGIBLE`) exists and was never run through the behavioral protocol (`reports/phase1_semantic_audit_report.md`: `FULL_vs_B2` count 0, verdict `EXECUTION_OR_PROTOCOL_BLOCKED`). The content-vs-frame decomposition of the Phase 1 delta is already implemented upstream and only needs runs.
10. **Metadata defect in v11.2**: `influences_later_discrete_decision` is declared `False` for `channel_layer` (`v11_2_child.py:386,763`) although its `z` output feeds `decode_affine` and everything downstream; harmless to the marker classifications (which do not consume the flag) but incorrect as provenance documentation.

## CLAUDE VERDICT

The Codex review's central answer is correct: **the program has not isolated a shortcut mechanism because no experiment combines A (semantic preservation), B (selective destruction of a declared finite nuisance family), and C (an end-to-end behavioral readout).** Independent analysis adds that the failure is structural, not incidental: Phase 1's only behavioral instrument destroys the relation it would need to preserve (and, on half its tasks, destroys the shape frame too), while every Phase 2 instrument audits an untrained forward pass that is causally upstream-disconnected from the pairing phenomenon — pairing information only ever enters the model through `train.take_step` gradients, which Phase 2 deliberately never runs. The byte- and threshold-level markers that Phase 2 produces are dominated by floating-point reduction-order effects that fire identically in controls, so no amount of further forward tracing can convert them into shortcut evidence.

What the current evidence *does* license, short of isolation: (i) the trained pipeline's query NLL depends on pairing-consistent cross-example structure beyond output marginals and per-example memorization, on frame-clean tasks (C2/C3); (ii) the untrained forward path is support-order equivariant to ≤1.4e-10 with unchanged discrete outputs under the audited regime (C6/C8); (iii) v8's protocol is invalid and its labels must not be cited (C4/C5). No narrower shortcut claim is provable from existing artifacts — in particular, "the shape-frame route carries part of the Phase 1 delta" is *plausible* from the `604001fa` confound but was never measured in a controlled comparison.

## AGREEMENTS WITH CODEX

- No existing experiment satisfies A/B/C simultaneously; no shortcut family is isolated (C1) — CONFIRMED.
- Phase 1's swap destroys legitimate R; it establishes sensitivity, not shortcut dependence (C3) — CONFIRMED.
- v8's AFFINE/NORMALIZE localization is invalid and superseded (C4/C5, conclusion) — CONFIRMED.
- v10/v11.2 execute no loss/backward/optimizer/adaptation and certify no production-decoder behavior (C6/C9) — CONFIRMED.
- E_bit is not a valid shortcut-localization marker; E_threshold at Cummax is insufficient for any carrier claim (C7/C8) — CONFIRMED.
- All six "WHAT MUST NOT BE DONE YET" assessments (no Cummax lock, no Channel-Layer lock, no VRM addition, no reasoning claim, no carrier claim, no further broad provenance scan) — CONFIRMED on independent evidence.
- The blocker taxonomy and the A/B/C gap analysis — CONFIRMED (one scoring nuance, C11).
- R5A-4B is a separate line whose destructive conditions do not isolate an R6A shortcut (C10) — CONFIRMED.

## CORRECTIONS TO CODEX

1. **Evidence base for the v8 invalidation (material).** The review cites `v8_action_behavior_audit.json` and its `head_weights`/`mask_weights` records as primary evidence. That artifact misparses v8's malformed CSV (comma-shifted columns; the real `has_real_example_axis` for `head_weights/0` is `false`). The invalidation should be cited from `state_action_certificate.json` (232 permuted classes) versus `semantic_state_action_certificate.json` (30), and `semantic_action_table.csv` (324 vs 0 accidental collisions). Same conclusion, different — and sound — evidence.
2. **Phase 1 claim scope (material).** "Pairing-sensitive query behavior" should be stratified: cleanly established on the three frame-preserving Stratum A tasks; conflated with shape-frame destruction on `604001fa`, `79fb03f4`, `aa4ec2a5` (evidence in the repository's own symmetry preflight). The Codex main report and claim matrix do not mention the S-condition frame confound.
3. **Control-case E_bit reading (interpretive).** "Baseline byte-instability or trace-alignment/capture marker" under-commits: the drift is mechanistically attributable to example-axis reduction order in `channel_layer`/`normalize`, and the passing transparency/calibration certificates disfavor the capture-fault branch of the v11.2 label.
4. **v8 A-score (minor).** `shortcut_isolation_gap_analysis.json` grades v8 condition A as PARTIAL; as-executed it is FAIL.
5. **Experiment spec pins (additive).** The proposed S/R/N design needs two explicit constraints to be executable on CompressARC (fixed color set; fixed shape frame), and there is a strictly cheaper first experiment using only existing machinery (below).

## REMAINING UNCERTAINTIES

- Why only `604001fa` crosses the tolerance threshold at Cummax is undecidable from current data: near-tie amplification of reduction-order drift (favored by the microtrace) versus some deeper example-coupling in the directional stack cannot be separated without a fixed-order numerical oracle outside the replay cone — which `selective_numerical_oracle.json` explicitly did not provide (`ORACLE_LIMITED_TO_REPLAY_CONE`).
- Whether the trained (post-1500-step) forward remains order-equivariant is untested; all equivariance evidence is step-0.
- Whether the production float32/CUDA stochastic regime behaves like the audited float64/CPU/zero-noise regime is untested.
- How much of the Phase 1 delta on frame-confounded tasks is frame versus content is unmeasured (the B1-vs-B2 decomposition exists in code but has never produced data).
- The frame-clean Phase 1 deltas could still contain undeclared nuisance families (e.g., palette-usage statistics that survive the multiset constraint); nothing measures this yet.

## MINIMUM NEXT DECISION GATE

The smallest missing causal comparison, buildable entirely from existing machinery, is:

**C_preserve = FULL versus C_break = ORDER_PERMUTED**, run through the frozen Phase 1 behavioral protocol: identical frozen initialization artifacts, identical seeds, 1500 `train.take_step` steps, evaluator-side query target-cell NLL plus decoded grids, judged against the existing F1/F2 replicate noise floor nu — on the three frame-clean Stratum A tasks (`3ed85e70`, `929ab4e9`, `981571dc`) plus Stratum B controls. Support order is semantically null by construction (the task-action certificates already prove the transform preserves every task field up to relabeling), so A and B hold exactly, and the readout supplies C. Two decisive outcomes: |delta_O| >> nu behaviorally isolates an order-linked shortcut family in the *trained* pipeline (something no current evidence addresses); delta_O ≈ nu behaviorally excludes the order family and licenses the second-smallest comparison — **B1 versus B2** (all-pairing swap versus shape-preserving swap, `r6a_common.py:71-118`) — which decomposes the established Phase 1 delta into content-pairing and shape-frame components. Only after one of these behavioral comparisons shows a family-specific effect does any mechanism-level (layer/trace) work have an interpretable target; until then, locking layers, adding VRM, or running further provenance scans remains unjustified, exactly as the Codex review concludes.
