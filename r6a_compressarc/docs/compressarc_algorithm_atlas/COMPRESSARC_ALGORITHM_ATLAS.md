# CompressARC / R6A Algorithm Atlas

## 0. Scope, source revision, and evidence standard

This atlas documents, from direct line-by-line source reading only, the algorithm executed by the CompressARC reference implementation vendored at `third_party/compressarc_reference/` and by its R6A Phase 1 wrapper under `r6a_compressarc/scripts/`. No model, `Task`, or Torch execution occurred; no tests, training, inference, or Git commands were run. All writes are confined to `r6a_compressarc/docs/compressarc_algorithm_atlas/`.

Source revision identifiers (read as plain text, no Git commands):
- `.git/HEAD` contains `ref: refs/heads/main` (branch pointer only).
- `r6a_compressarc/source_provenance.json` pins the upstream CompressARC source to commit `83a22218024d46273eb32b769a906340202ffb4d` of `https://github.com/iliao2345/CompressARC` ("added code for listing the puzzles that were solved"), official requirements `torch==2.5.1`, `numpy==2.2.2`.

Evidence standard: every equation, edge, or strong claim in this atlas and its companion files carries exactly one label from `CODE_EXACT`, `DERIVED_EQUIVALENT`, `SCHEMATIC`, `UNRESOLVED` (definitions per the task charter). Executable code wins over comments and docstrings. Where this document introduces symbols for readability, they are marked: *Notation introduced by this document only; not a variable name from source.* Companion files: `tensor_state_inventory.csv`, `forward_execution_graph.md`, `optimization_equations.md`, `source_evidence_index.csv`, `unresolved_or_ambiguous_points.md`, `reviewed_source_inventory.csv`.

## 1. Executive one-page algorithm summary

CompressARC solves one ARC task at a time by **per-task optimization of a randomly initialized decoder-only generative model** — there is no pretraining and no cross-task weight sharing (`train.py:134-152`; one fresh `ARCCompressor` and one fresh Adam per task). [DERIVED_EQUIVALENT, composition of CODE_EXACT facts; evidence E64]

1. **Task construction** (`preprocessing.Task`): raw ARC JSON becomes (a) an integer color-index tensor `task.problem` of shape `(n_examples, n_x, n_y, 2)` covering train inputs, train outputs, and test inputs — the hidden test output slot is skipped and remains background (`preprocessing.py:106-107, 115`); (b) rectangle masks `task.masks` `(n_examples, n_x, n_y, 2)`; (c) shape metadata including a *predicted* test output shape chosen by size flags (`preprocessing.py:44-62`). [CODE_EXACT]
2. **Initial state**: 18 multitensor leaves (5 axes: examples, colors, directions(8), x, y; validity rules give 18 combinations, 15 with an example axis). Per leaf: a trainable posterior mean `0.01·randn` and a trainable zero "local capacity adjustment" of shape `shape(dims, 4)`, plus a trainable 4-vector target capacity; plus all linear layer weights. **Everything is trainable and everything is registered with Adam** (`initializers.py`, `arc_compressor.py:34-91`). [CODE_EXACT]
3. **Inner loop** (1500 steps in R6A Phase 1; 2000 upstream): each `train.take_step` runs one stochastic forward pass, computes `loss = total_KL + 10·reconstruction_error`, backpropagates, and takes one Adam step (lr 0.01, betas (0.5, 0.9)) on the entire state (`train.py:37-120`). Reconstruction is a crop-position-marginalized cross-entropy over train inputs, train outputs, and the **test input** — never the hidden test output (`train.py:62-63`). [CODE_EXACT]
4. **Forward** (`ARCCompressor.forward`): takes **no data argument**. It samples per-leaf activations from the posteriors through an information-capacity channel (`channel_layer`), decodes them into a residual multitensor stream, and runs 4 identical blocks of share-up → softmax → cummax → shift → direction-share → nonlinear → share-down → normalize, then a color head and two mask heads. The only Task fields forward reads are `task.masks`, the two size flags, and `task.shapes` (out-of-bounds -1000 suppression) — never raw grids (`arc_compressor.py:94-158`). [CODE_EXACT]
5. **Decoding**: at every step the Logger slices the test-example output-mode logits, forms two candidate discrete grids (current and EMA) by per-cell argmax plus a mask-scored best-crop, scores them by certainty, and accumulates votes; the final prediction is the top-voted discrete grid across all steps (`solution_selection.py:54-162`). R6A Phase 1 additionally captures the softmax probabilities of the final 100 steps and evaluates NLL against the hidden solution **in the parent evaluator process only** (`phase1_condition_solver.py:146-155`; `run_phase1_multirepeat_pairing_audit.py:220-239`). [CODE_EXACT]

Task-specific information reaches the query prediction almost entirely **through gradient adaptation of the trainable state**; the forward pass itself sees only shape/mask metadata. [DERIVED_EQUIVALENT, E68]

## 2. Symbols and tensor-axis glossary

Axis system (CODE_EXACT, `multitensor_systems.py:9, 26-33, 53-68`): a leaf is identified by `dims = [e, c, d, x, y]` with axis lengths `[n_examples, n_colors, 8, n_x, n_y]`. A leaf tensor's physical axes are the included dims **in that fixed order**, plus one trailing channel axis. Validity (`dims_valid`, lines 34-51): `(x or y) → e`, and at least one of `c, d, x, y`. Iteration order is little-endian over bit index with examples as bit 0 (`lines 70-83`).

| Symbol (this doc) | Meaning | Source anchor |
|---|---|---|
| `E = n_examples` | train + test examples; axis 0 where present | `preprocessing.py:19` |
| `n_colors` | distinct observed colors excluding black | `preprocessing.py:89` |
| `n_x, n_y` | max height/width over all shapes incl. predicted test output | `preprocessing.py:77-78` |
| direction axis | size 8; indices are 2 direction-splits × 4 kinds (x, x+y, y, y−x) | `multitensor_systems.py:28`; `layers.py:353-402` |
| `ch(dims)` | residual channel width: 16 if no direction axis else 8 | `arc_compressor.py:31-32` |
| mode axis | trailing size-2 axis meaning input(0)/output(1) | `preprocessing.py:99, 112` |
| `T` | the `preprocessing.Task` object | *Notation introduced by this document only; not a variable name from source.* |
| `s_t` | tuple of all tensors in `model.weights_list` at step t | *Notation introduced by this document only; not a variable name from source.* |

Unresolved axis semantics: none at the leaf level; edge-case semantics of masked cells inside directional blocks are UNRESOLVED-04.

## 3. Raw ARC JSON -> preprocessing.Task

Required form (A), with every term mapped to source:

```text
T = Preprocess( {(x_i, y_i)}_{i=1..n_train}, {x_j}_{j=1..n_test} )        [SCHEMATIC wrapper; components below CODE_EXACT]
```

*Notation introduced by this document only; not a variable name from source.* Here `(x_i, y_i)` are the train input/output grids and `x_j` the test input grids of one challenge JSON entry.

Constructor sequence (`preprocessing.py:15-30`) [CODE_EXACT]:
1. `n_train`, `n_test`, `n_examples`; raw dict kept as `unprocessed_problem`.
2. `_collect_problem_shapes` (32-42): per-example `[in_shape, out_shape]`; test `out_shape = None` when the key is absent.
3. `_predict_solution_shapes` (44-62): flags `in_out_same_size` (train pairs), `all_in_same_size` (all inputs incl. test), `all_out_same_size` (present outputs). Test out-shape := test in-shape if `in_out_same_size`; elif `all_out_same_size` := `shapes[0][1]` (the **same list object** for every test example — latent aliasing, UNRESOLVED-06); else `[max_x, max_y]`.
4. `_construct_multitensor_system` (73-93): `n_x`, `n_y` = maxima over all shapes **including predicted test outputs**; `colors` = sorted union of observed colors ∪ {0}; `n_colors = len(colors) − 1`; builds `MultiTensorSystem(n_examples, n_colors, n_x, n_y, task)`.
5. `_compute_mask` (143-156): `masks[e,:,:,m] = outer(arange(n_x)<H, arange(n_y)<W)` with `(H,W) = shapes[e][m]` — the test-output rectangle uses the **predicted** shape.
6. `_create_problem_tensor` (95-115): one-hot canvas `(E, n_colors+1, n_x, n_y, 2)` filled for train in/out and test in; **test outputs skipped** (106-107); then `problem = argmax over color axis` → integer tensor `(E, n_x, n_y, 2)`; unfilled cells (padding and the hidden slot) argmax to index 0 = black.
7. `_create_solution_tensor` (125-141) runs **only** if a solution argument is passed. Presence of hidden query outputs: upstream `preprocess_tasks` passes solutions for training/evaluation splits (`preprocessing.py:159-174`) but they are used upstream only for accuracy plotting (`train.py:143-153`); the R6A Phase 1 solver passes `None` (`phase1_condition_solver.py:178`), so `task.solution is None` inside the solver process. [CODE_EXACT]

## 4. Task tensor inventory and exact shapes

(Full table: `tensor_state_inventory.csv`.) Key rows [all CODE_EXACT]:

- `task.problem`: `(E, n_x, n_y, 2)` int64 color **indices** into `task.colors`; consumed only by the loss (`train.py:74`); never read by forward.
- `task.masks`: `(E, n_x, n_y, 2)` default dtype; consumed by forward in cummax/shift (`arc_compressor.py:127, 131`) and in share-down masked aggregation (`layers.py:211-230`).
- `task.shapes`: list; consumed by loss crops (`train.py:75`), `postprocess_mask` (`layers.py:559-567`), decoder crop lengths (`solution_selection.py:140-141`).
- `task.colors`: sorted list with `colors[0] = 0` (black); index→color mapping used by the decoder (`solution_selection.py:156-159`) and Phase 1 probability canvas (`phase1_condition_solver.py:150-155`).
- Flags `in_out_same_size, all_in_same_size, all_out_same_size`: consumed by the loss branch (`train.py:68`), share-down branch (`layers.py:211, 221`), decoder crop branch (`solution_selection.py:112`).
- `task.solution`: `None` in the R6A solver; `(n_test, n_x, n_y)` int64 when provided upstream (plotting only).

## 5. Model state inventory at initialization

Required form (B):

```text
s_0 = { posterior means (18), local capacity adjustments (18), target capacities (18),
        decode weights, share-up/down weights, softmax weights, cummax weights, shift weights,
        direction-share weight tables, nonlinear weights, head weights, mask weights,
        (optimizer-visible mutable tensors = exactly the weights_list registrations) }      [CODE_EXACT]
```

*Notation introduced by this document only; not a variable name from source.* This state is not called a rule or `z` by this document.

Facts [CODE_EXACT unless noted]:
- Posterior per leaf: `mean = 0.01·torch.randn(shape(dims, 4))`, `local_capacity_adjustment = zeros(same shape)`; both `requires_grad=True`, both registered (`initializers.py:58-69`).
- Target capacity per leaf: `zeros([4])` — one 4-vector per leaf, not per element (`arc_compressor.py:51`; `initializers.py:24-30, 97-100`).
- Linear layers: `W = randn(n_in, n_out)/√n_in`, `b = randn(n_out)/√n_in`, both registered (`initializers.py:32-50`). Residual pairs: down `(ch → k)`, up `(k → ch)` (`initializers.py:52-56`), with per-block `k`: share_up 16, share_down 8, softmax 2 (up-width `2·(2^k_axes − 1)`, `arc_compressor.py:64-65`), cummax 4, shift 4, nonlinear 16.
- Head: `W (16, 2)` with **both output columns equal at initialization** by explicit duplication, `b (2,)` (`initializers.py:79-91`); the columns are independent parameters afterwards. Mask head: one `[W (16,2), b]` object shared by the x and y heads (`arc_compressor.py:72-74, 152-153`).
- Symmetrization rebinding: `symmetrize_xy` makes every `dims=[·,·,·,0,1]` leaf **alias** the `[·,·,·,1,0]` leaf's weight objects (`initializers.py:122-126`); `symmetrize_direction_sharing` aliases direction-pair maps to representative maps (`128-168`). Because registration happened at creation (`initializers.py:49`), the replaced originals remain in `weights_list` as **orphaned optimizer parameters** that no forward path uses. [DERIVED_EQUIVALENT, E19; enumeration UNRESOLVED-05]
- There is no non-trainable model buffer: every created tensor has `requires_grad=True` and is registered. `weights_list` is exactly the Adam parameter set (`train.py:137`).

## 6. Per-task optimization loop

Required form (C):

```text
s_{t+1} = OptimizerStep( s_t, grad_{s_t} L(T, s_t) )          [CODE_EXACT as call structure]
```

*Notation introduced by this document only; not a variable name from source.*

- One `take_step` = zero_grad → forward → loss → backward → `Adam.step()` → zero_grad → `Logger.log` (`train.py:49-120`). [CODE_EXACT]
- Gradient receivers: every `weights_list` tensor reachable in the graph — all posteriors, capacities, and used weights; orphaned duplicates receive none. [DERIVED_EQUIVALENT]
- Optimizer: `torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))` (`train.py:137`; `phase1_condition_solver.py:182`); internal Adam math is torch library semantics [SCHEMATIC; UNRESOLVED-01].
- Loop lengths: upstream `n_iterations = 2000` (`train.py:148`); R6A Phase 1 `--steps 1500` = `SMOKE_BUDGET` (`r6a_common.py:15`; `phase1_condition_solver.py:195`).
- Stochasticity: one fresh `randn` per leaf per forward (18 draws/step), consumed in leaf iteration order [CODE_EXACT + DERIVED_EQUIVALENT E63; values UNRESOLVED-02].

## 7. Exact forward-pass equations in execution order

Required form (D):

```text
(l_color, l_x, l_y, KL) = F(s_t, T)          [SCHEMATIC wrapper; expansion below]
```

*Notation introduced by this document only; not a variable name from source.* Source order (`arc_compressor.py:114-158`) [CODE_EXACT]; full per-edge detail in `forward_execution_graph.md`.

1. **decode_latents** (`layers.py:133-163`): for each leaf, `z, KL = channel_layer(tc, [mean, lca])`, then `x[dims] = z·W_dec + b_dec`.
2. **channel_layer** (`layers.py:71-131`) [CODE_EXACT], with `D = numel(mean)`, `abl` = mean over all axes except the trailing channel axis (includes the example axis when present):
   - `tc' = 10·tc`; `C_glob = exp(tc')·10000 + 0.5`; `out_scale = 1 − exp(−2·C_glob/D)`
   - `lc = tc' + lca − abl(lca)`; `C_loc = exp(lc)·10000 + 0.5`
   - `noise_std = exp(−C_loc/D)`; `signal_std = sqrt(1 − exp(−2·C_loc/D))` (stabilized by `where(arg>20 → 1)`)
   - `nm = (mean − abl(mean)) / sqrt(abl(nm² + 1e-8))`
   - **sampling**: `z = out_scale·(signal_std·nm + noise_std·ε)`, `ε ~ torch.randn(nm.shape)` — the only stochastic operation; the variable named `z` here is a source variable local to `channel_layer`.
   - `KL = 0.5·(noise_var + signal_var·nm² − 1) + C_loc/D` element-wise.
   - **Zero-noise path: none exists in upstream source.** [CODE_EXACT]
3. **Loop ×4** (`arc_compressor.py:118-145`):
   - `share_up`: down-affine each leaf; each leaf sums all lower leaves' down-projections (unsqueeze-broadcast); normalize; up-affine; residual add (`layers.py:166-199, 236-238`).
   - `softmax`: down to 2 channels; softmax over every non-empty subset of the leaf's non-example, non-channel axes; concatenate subsets on the channel axis; up-affine; residual add (`layers.py:283-307`). **The `pre_norm=True` argument is ineffective**: `add_residual` computes `normalize(x)` and immediately overwrites it, so the down-affine reads raw `x` (`layers.py:61-63`). [CODE_EXACT]
   - `cummax` (leaves `(1,1,1,1,1)`, `(1,0,1,1,1)` only): masked directional cumulative max, rescaled `(cummax(x−m)+m − min_)/(max_−min_)·2 − 1` with `m = 1e3·(1−mask)`; post-norm effective; residual add (`layers.py:418-452`).
   - `shift` (same leaves): masked one-step directional shift with zero padding (`layers.py:467-486`).
   - `direction_share` (direction-axis leaves): `x_d1 += coeff[(d2−d1) mod 8] · affine(normalize(x)_d2, W[d1][d2])`, coefficients `[1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]`; its own pre-norm **is** effective (`layers.py:488-524`).
   - `nonlinear`: `x += W_up(silu(W_down(x)))` (`layers.py:526-540`), pre-norm ineffective as above.
   - `share_down`: like share_up but aggregating higher leaves down; over x/y axes uses **masked mean** `sum(x·masks)/(sum(masks)+1e-4)` when `in_out_same_size or all_out_same_size`, else plain mean (`layers.py:200-234`).
   - `normalize`: `(x − mean)/sqrt(1e-8 + mean(x²))` over all non-channel axes (`layers.py:17-31`).
4. **Heads** (`arc_compressor.py:148-153`): `output = affine(x[(1,1,0,1,1)], head_W) + 100·head_b` → `(E, n_colors, n_x, n_y, 2)`; `x_mask = affine(x[(1,0,0,1,0)], mask_W, bias)` → `(E, n_x, 2)`; `y_mask` likewise from the y leaf with the **same** `mask_weights`.
5. **postprocess_mask** (`layers.py:542-570`): adds `−1000` to mask scores at positions ≥ per-example `max(in, out)` axis length (test uses the predicted out-shape).

Forward reads no raw grids; its Task inputs are exactly `task.masks`, the two size flags, and `task.shapes`/sizes. [CODE_EXACT, E24/E34/E36/E41]

## 8. Loss, KL, masking, and optimizer update equations

Full treatment in `optimization_equations.md`. Summary [CODE_EXACT]:

- `logits = cat(zeros_black, output)` on the color axis (`train.py:51`).
- `total_KL = Σ_leaf Σ_elements KL_leaf` (`train.py:54-56`).
- Reconstruction, per participating (example, mode) — train in, train out, test in; **test out skipped** (`train.py:62-63`): crop-position-marginalized cross-entropy — `score(o_x,o_y) = x_lp(o_x) − x_logZ + y_lp(o_y) − y_logZ − CE(crop)`, `logprob = logsumexp(c₂·score)/c₂`, with mask-slice log-probabilities `x_lp(o) = −Σc₁·xm[:o] + Σc₁·xm[o:o+H] − Σc₁·xm[o+H:]`, annealing `c₁ = 0.01^max(0,1−t/100)`, `c₂ = 0.1^max(0,1−t/100)` only when `grid_size_uncertain` (`train.py:23-35, 68-104`).
- `loss = total_KL + 10·reconstruction_error`; `backward()`; `Adam.step()` (`train.py:106-108`).
- Hidden query target: absent from Task (solver), loss, optimizer, forward, and decoder; present in the parent evaluator only (`run_phase1_multirepeat_pairing_audit.py:427-428, 450`), with a solver-side `open()` firewall (`phase1_condition_solver.py:24-37`). [CODE_EXACT]

## 9. Support/query information-flow map

Required form (F), two separate paths [labels per line]:

```text
Path 1 (gradient path) — CODE_EXACT composition:
Visible support grids (train in/out) and test input
  -> Task construction (problem tensor, masks, shapes, colors, flags)     preprocessing.py:15-156
  -> loss (crop-marginalized CE over participating grids + KL)            train.py:54-104
  -> gradients (loss.backward)                                            train.py:107
  -> adapted state (Adam step on ALL weights_list tensors)                train.py:108
  -> forward (reads adapted state + masks/shapes/flags only)              arc_compressor.py:94-158
  -> query prediction (test-slice logits -> voted decode)                 solution_selection.py:54-162
```

```text
Path 2 (direct forward dependencies) — CODE_EXACT:
Visible support grids -> direct forward reads: NONE of the raw grids.
Forward's only Task-derived inputs are:
  task.masks                (cummax/shift masking; share_down masked mean)   layers.py:334-343, 211-230
  in_out_same_size / all_out_same_size flags (share_down branch)             layers.py:211, 221
  task.shapes and n_examples/n_x/n_y (postprocess_mask -1000 suppression)    layers.py:558-567
  tensor dimensions n_examples/n_colors/n_x/n_y (leaf shapes)                multitensor_systems.py:53-68
```

Therefore raw support **content** (which color is where) influences the query logits **only** through gradient adaptation of `s_t`; raw support **geometry** (shapes) additionally influences forward directly through masks/shape metadata. [DERIVED_EQUIVALENT, E68]

## 10. Query decoding and evaluator path

Required form (E):

```text
ŷ_q = Decode( l_color_q, l_x_q, l_y_q, T )        [SCHEMATIC wrapper; expansion below is CODE_EXACT]
```

*Notation introduced by this document only; not a variable name from source.*

Training-time voting decoder (`solution_selection.py`):
1. Per step, `current_* = logits/x_mask/y_mask[n_train:, ..., mode=1]` (56-58) and EMA copies with decay 0.97 (60-62).
2. For each candidate (current, EMA): per-cell `colors = argmax` over the color axis (132); per-cell `uncertainty = logsumexp − amax` (133); crop via `_best_slice_point` — slice score `−Σmask[:o] + Σmask[o:o+L] − Σmask[o+L:]`, with `L` fixed to the predicted query shape when `in_out_same_size or all_out_same_size`, else searched over all lengths (106-128); indices mapped through `task.colors` (156-159).
3. Scores: `−10·mean(uncertainty)`, minus 10 if `step < 150`, minus 4 for the EMA candidate (73-77); accumulated per solution hash via `np.logaddexp` (81-82); top two tracked (90-104).
4. Final attempts = `solution_most_frequent`, `solution_second_most_frequent` — consumed by R6A as `attempt_1/attempt_2` (`phase1_condition_solver.py:137-143`).

R6A evaluator path: solver saves softmax probabilities of test query 0 for steps 1401-1500 onto a 30×30×10 canvas (`canonical_probability`, `phase1_condition_solver.py:146-155, 198-200`); the parent process computes `target_cell_nll_from_final100_mean_probability` over hidden-solution cells and exact/cell accuracies of attempt_1 (`run_phase1_multirepeat_pairing_audit.py:220-239`; `r6a_common.py:133-201`). [CODE_EXACT]

## 11. Does CompressARC contain an explicit task-level z?

### 11.1 Code fact

Objects that exist in source [CODE_EXACT]:
- `self.multiposteriors` — 18 per-leaf pairs `[mean, local_capacity_adjustment]` (`arc_compressor.py:48`; `initializers.py:58-69`).
- `self.target_capacities` — 18 per-leaf 4-vectors (`arc_compressor.py:51`).
- A **local variable named `z`** inside `layers.channel_layer` (line 124) and inside `decode_latents_` (line 157): the per-leaf sampled activation. It is transient (recomputed every forward), per-leaf, and never stored on the model.
- All layer weights (decode/share/softmax/cummax/shift/direction/nonlinear/head/mask), every one of them trained on the single task.

There is **no single object in source called a task latent, rule, or task-level z**; the only `z` names are per-leaf forward locals.

### 11.2 Mathematical description

*Notation introduced by this document only; not a variable name from source:* let `s_t^task ⊆ s_t` denote the subset of state whose values differ from `s_0` after adaptation. Because every tensor in `weights_list` receives Adam updates (§6), `s_t^task` is in principle **all** of `s_t`: the 36 posterior tensors (18 means + 18 capacity adjustments), the 18 target capacities, and every layer/head/mask weight jointly can carry task-specific information after 1500 steps. The posteriors are the only state with example-indexed axes (15 of 18 leaves), and the sampled per-leaf activations are deterministic functions of `(mean, lca, tc, ε)`; but nothing in source privileges them over the trained weights as "the" task representation. [DERIVED_EQUIVALENT]

### 11.3 Epistemic limit

Static source reading establishes what state exists, what updates it, and what reads it. It cannot establish that this state constitutes a rule, a program, reasoning, a shared mechanism, or a coherent latent z: those are claims about *what the adapted values encode and how they causally produce behavior*, which requires behavioral or causal experiments (interventions on state or data with end-to-end readouts). Such experiments are outside the scope of this atlas. Accordingly this atlas asserts only: task information is **distributed across the entire trainable state**, with no explicit task-level latent variable in source. [CODE_EXACT for absence of such an object; everything stronger is out of scope]

## 12. Full end-to-end pseudocode

[SCHEMATIC — faithful to call structure; line anchors in brackets]

```text
# Per task (upstream: train.py:134-152; R6A: phase1_condition_solver.run_condition)
task  = preprocessing.Task(name, problem_json, solution_or_None)      [preprocessing.py:15-30]
model = ARCCompressor(task)                                           [arc_compressor.py:34-91]
opt   = Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))           [train.py:137]
log   = Logger(task)                                                  [solution_selection.py:15-38]

for t in range(STEPS):            # 2000 upstream / 1500 R6A
    opt.zero_grad()                                                   [train.py:49]
    # forward: no data argument
    x, KLs, KLnames = decode_latents(model.target_capacities,
                                     model.decode_weights,
                                     model.multiposteriors)           [arc_compressor.py:114-116]
      # per leaf: z = out_scale*(signal_std*nm + noise_std*randn()); KL element-wise
    for layer in range(4):                                            [arc_compressor.py:118-145]
        x = share_up(x, W_su[layer])
        x = softmax(x, W_sm[layer])            # pre_norm arg dead
        x = cummax(x, W_cm[layer], task.masks) # only 2 leaves; post-norm
        x = shift(x,  W_sh[layer], task.masks) # only 2 leaves; post-norm
        x = direction_share(x, W_ds[layer])    # direction leaves; real pre-norm
        x = nonlinear(x, W_nl[layer])          # pre_norm arg dead
        x = share_down(x, W_sd[layer])         # masked mean if size flags
        x = normalize(x)
    out    = affine(x[(1,1,0,1,1)], head_W) + 100*head_b              [arc_compressor.py:148-151]
    x_mask = affine(x[(1,0,0,1,0)], mask_W, bias=True)                [arc_compressor.py:152]
    y_mask = affine(x[(1,0,0,0,1)], mask_W, bias=True)                [arc_compressor.py:153]
    x_mask, y_mask = postprocess_mask(task, x_mask, y_mask)           [arc_compressor.py:156]

    logits = cat(zeros_black, out, color_axis)                        [train.py:51]
    loss   = sum(KLs) + 10 * crop_marginalized_CE(logits, x_mask, y_mask,
                             task.problem, task.shapes, flags, t)     [train.py:54-106]
    loss.backward(); opt.step(); opt.zero_grad()                      [train.py:107-109]
    log.log(t, logits, x_mask, y_mask, ...)   # votes two decoded candidates
                                                                       [solution_selection.py:40-104]

prediction = (log.solution_most_frequent, log.solution_second_most_frequent)
# R6A only: save softmax(query logits) for steps 1401-1500; parent process
# evaluates NLL/accuracy against the hidden solution.                  [phase1_condition_solver.py:146-155;
#                                                                       run_phase1_multirepeat_pairing_audit.py:220-239]
```

## 13. R6A Phase 1 wrapper versus upstream CompressARC

Upstream semantics (`third_party/compressarc_reference/`) [CODE_EXACT]:
- `train.py.__main__`: builds all tasks of a split via `preprocess_tasks` (solutions loaded for training/evaluation splits and used for the accuracy plot only), 2000 steps per task, float32 + default device `cuda` set at import (`arc_compressor.py:10-11`).

R6A Phase 1 wrapper (`r6a_compressarc/scripts/`) [CODE_EXACT]:
1. **Condition construction**: `challenge_for_condition` returns a deep-cloned problem; `FULL`/`FULL_REPEAT` unchanged; `ALL_PAIRING_SWAP` reassigns `train[idx]['output'] = original_outputs[perm[idx]]` for a derangement chosen by `select_swap` to maximize padded Hamming distance (`r6a_common.py:121-130`; `run_phase1_multirepeat_pairing_audit.py:56-73`). Because `Task` recomputes shapes/flags/masks from whatever JSON it receives, a swap can change `in_out_same_size`, per-example mask rectangles, and the predicted query shape when output shapes are heterogeneous (mechanical consequence of `preprocessing.py:44-62, 143-156`).
2. **Frozen initialization**: per task/seed, one artifact stores model weights, Adam state, and RNG states created from the FULL problem (`run_phase1_multirepeat_pairing_audit.py:111-147`); the solver rebuilds `Task` from the per-condition challenge, constructs a fresh model, then copies the frozen weights in place, loads the optimizer state, and restores RNG (`phase1_condition_solver.py:176-184`). Weight shapes stay compatible across F/S because the swap preserves the multisets of shapes and colors.
3. **Budget and dtype**: 1500 steps (`SMOKE_BUDGET`, `r6a_common.py:15`); `torch.set_default_dtype(float32)` and explicit device selection inside the solver (`phase1_condition_solver.py:169-170`).
4. **Isolation**: solver-side `open()` firewall against `solutions`/`submission.json` (`phase1_condition_solver.py:24-37`); solutions read only in the parent evaluator (`run_phase1_multirepeat_pairing_audit.py:427-428, 450`).
5. **Extra outputs**: last-100-step query probability canvas (30×30×10, first test query only) and the two voted attempts; evaluator computes target-cell NLL of the mean canvas plus discrete accuracies (`phase1_condition_solver.py:146-155, 137-143`; `run_phase1_multirepeat_pairing_audit.py:220-239`).
6. **No algorithmic change**: the wrapper calls the upstream `train.take_step` and upstream model unmodified (`phase1_condition_solver.py:164-167, 196`).

## 14. Known task-dependent branches and shape/frame effects

All CODE_EXACT:
1. Predicted query output shape branch (`preprocessing.py:52-62`): in_out_same_size → test-input shape; all_out_same_size → first train output shape (shared list object); else `[max_x, max_y]`.
2. Loss size-certainty branch and annealing (`train.py:68-102`); uncertain partitions marginalize over all lengths (`79-87`).
3. share_down masked-mean vs plain-mean branch on `(in_out_same_size or all_out_same_size)` per aggregated axis (`layers.py:211, 221`).
4. cummax/shift leaf filter `{(1,1,1,1,1), (1,0,1,1,1)}` (`layers.py:448, 482`); direction_share leaf filter `dims[2]==1` (`layers.py:488-490`).
5. `channel_dim_fn`: 16 vs 8 by direction axis (`arc_compressor.py:31-32`).
6. postprocess_mask −1000 beyond per-example `max(in, out)` lengths, test rows using predicted shapes (`layers.py:558-567`).
7. Decoder crop-length branch (`solution_selection.py:112-115`).
8. softmax subset count `2^k − 1` with `k = dims[1]+dims[2]+dims[3]+dims[4]` (`arc_compressor.py:64`; `layers.py:295-300`).
9. `n_x/n_y` maxima include predicted test-output shapes (`preprocessing.py:77-78`) — so the *predicted* query shape can set canvas size for the whole model.

## 15. Exact claims / derived claims / schematic descriptions / unresolved points

- CODE_EXACT: 59 evidence rows (see `source_evidence_index.csv`, E01-E12, E14-E18, E20-E62 excluding rows labeled otherwise) — task construction, visibility rules, initialization, forward equations, loss terms, optimizer registration, decoder mechanics, wrapper mechanics.
- DERIVED_EQUIVALENT: 7 rows (E13 leaf count; E19 orphaned registrations; E63 RNG order; E64 per-task-optimization characterization; E65 epsilon placement; E66 softmax width; E68 information-flow composition).
- SCHEMATIC: 2 rows (E62 Adam internals; E67 pipeline banner) plus the wrapper forms in §§3, 7, 10, 12 marked inline.
- UNRESOLVED: 6 items (U1 Adam as-executed math; U2 RNG values; U3 kernel tie/order semantics; U4 fully-masked-slice algebra; U5 orphan enumeration; U6 shapes-aliasing mutability) — see `unresolved_or_ambiguous_points.md`.

Notable corrections to informal descriptions established by source: (a) `add_residual`'s pre-norm is dead code — softmax/cummax/shift/nonlinear down-projections read the raw residual stream (`layers.py:61-63`); (b) there is no explicit task-level latent — the whole model trains per task; (c) the "posterior/KL" machinery is a capacity-parameterized noisy channel with its own KL formula, not a textbook Gaussian VAE ELBO (equations in §7.2); (d) the decoder is a step-wise vote over discrete candidate grids, not a single final argmax (`solution_selection.py:81-104`).

## 16. Source evidence index

See `source_evidence_index.csv` (68 rows, every strong claim mapped to path, line range, function, and a sub-20-word excerpt). Companion detail files: `tensor_state_inventory.csv` (24 state rows with axis semantics and alias flags), `forward_execution_graph.md` (16 edges in verified call order), `optimization_equations.md` (loss/gradient/update), `unresolved_or_ambiguous_points.md` (6 items), `reviewed_source_inventory.csv` (12 rows; 10 source files read line-by-line plus 2 identifier files).
