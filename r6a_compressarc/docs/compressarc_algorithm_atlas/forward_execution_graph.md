# Forward Execution Graph (source call order)

Scope: one call of `ARCCompressor.forward()` (`third_party/compressarc_reference/arc_compressor.py:94-158`) followed by the two decoding paths that consume its outputs (training-time `solution_selection.Logger` voting decoder, and the R6A Phase 1 probability capture + evaluator). Axis notation: a leaf `dims=[e,c,d,x,y]` has physical axes = the included dims in that fixed order plus one trailing channel axis (`multitensor_systems.py:53-68`). `ch(dims) = 16 if dims[2]==0 else 8` (`arc_compressor.py:31-32`). Every claim below carries an evidence class; source-line citations are to executable code.

Verified order note: the listed template order "channel_layer -> decode_latents -> share -> softmax -> cummax -> shift -> direction -> nonlinear -> normalization" is **not** exactly the source order. Actual source order per forward call is:
`decode_latents` (which internally calls `channel_layer` then a per-leaf `affine`) then a 4-iteration loop of `share_up -> softmax -> cummax -> shift -> direction_share -> nonlinear -> share_down -> normalize`, then the three heads, then `postprocess_mask` (`arc_compressor.py:114-158`). Evidence class: CODE_EXACT.

Important structural fact used below: `add_residual`'s `pre_norm` branch computes `normalize(x)` and immediately overwrites it (`layers.py:61-63`), so for `softmax`, `cummax`, `shift`, and `nonlinear` the down-projection reads the **raw residual stream**, not a normalized copy. Only `direction_share` implements an effective pre-norm of its own (`layers.py:503`). Evidence class: CODE_EXACT.

---

## Edge 1 — posteriors/capacities -> channel_layer (inside decode_latents)

```text
caller/callee: layers.decode_latents -> decode_latents_ (multify per leaf) -> layers.channel_layer
input tensor(s): target_capacities[dims] shape [4]; multiposteriors[dims] = [mean, local_capacity_adjustment], each shape(dims, 4)
output tensor(s): z shape(dims, 4); KL shape(dims, 4) (element-wise)
axis semantics: included dims axes in order (examples, colors, directions, x, y) + channel axis of size 4
task-specific information source: adapted posterior/capacity values only (shapes fixed by n_examples/n_colors/n_x/n_y)
source path and lines: third_party/compressarc_reference/layers.py:71-131 (channel_layer), 152-163 (decode_latents_)
evidence class: CODE_EXACT
```

Executed once per valid leaf (18 leaves; iteration order little-endian over `[e,c,d,x,y]` bits, `multitensor_systems.py:70-83`). The single `torch.randn` at `layers.py:124` is the only stochastic draw in the forward pass (CODE_EXACT). Zero-noise path: **no zero-noise branch exists in upstream source**; the R6A Phase 2 harnesses substitute `layers.torch.randn` externally, which is outside this atlas's forward graph (CODE_EXACT for absence in upstream files read).

## Edge 2 — z -> decode affine -> residual stream x

```text
caller/callee: decode_latents_ -> layers.affine
input tensor(s): z shape(dims, 4); decode_weights[dims] = [W (4, ch(dims)), b (ch(dims),)]
output tensor(s): x[dims] shape(dims, ch(dims))
axis semantics: channel axis remapped 4 -> ch(dims); other axes unchanged
task-specific information source: adapted z and adapted decode weights
source path and lines: layers.py:158 (affine with use_bias=True), 33-47 (affine definition)
evidence class: CODE_EXACT
```

`KL_amounts`/`KL_names` (`str(dims)`) are collected here and returned to the loss (`layers.py:159-160`); they are not consumed further in forward (CODE_EXACT).

## Edge 3 — x -> share_up (layer loop, 4 iterations begin)

```text
caller/callee: ARCCompressor.forward -> layers.share_up -> share_direction(direction=1)
input tensor(s): full MultiTensor x; share_up_weights[layer][dims] = [[W_down (ch,16), b],[W_up (16,ch), b]]
output tensor(s): x[dims] + up(normalize(sum of lower-leaf down-projections)) per leaf
axis semantics: lower leaves are unsqueezed on missing axes then summed (broadcast add across leaves)
task-specific information source: adapted weights and stream; no Task field on the UP direction
source path and lines: layers.py:166-199 (down-affine 183, unsqueeze-sum 186-199), 236 (normalize), 237 (up-affine), 238 (residual add), 241-250 (share_up)
evidence class: CODE_EXACT
```

## Edge 4 — x -> softmax block

```text
caller/callee: forward -> layers.softmax (multify + add_residual)
input tensor(s): x[dims]; softmax_weights[layer][dims] = [[W_down (ch,2), b],[W_up (2*(2^k-1), ch), b]] with k = dims[1]+dims[2]+dims[3]+dims[4]
output tensor(s): x[dims] + W_up( concat over non-empty axis subsets of softmax(down(x)) )
axis semantics: softmax normalizes over each chosen subset of non-example, non-channel axes; example axis is excluded from subsets when present (layers.py:296-297); channel axis width becomes 2*(2^k-1) before up-projection
task-specific information source: adapted stream/weights only
source path and lines: layers.py:283-307; call at arc_compressor.py:123 (pre_norm=True is ineffective per layers.py:61-63; use_bias=False)
evidence class: CODE_EXACT (width formula DERIVED_EQUIVALENT, see source_evidence_index E66)
```

## Edge 5 — x -> cummax block (Task masks enter forward here)

```text
caller/callee: forward -> layers.cummax (multify + only_do_for_certain_shapes + add_residual + make_directional_layer)
input tensor(s): x[dims] for dims in {(1,1,1,1,1), (1,0,1,1,1)} ONLY; cummax_weights[layer][dims]; task.masks (n_examples, n_x, n_y, 2)
output tensor(s): x[dims] + W_up(normalize(directional cummax of down(x)))
axis semantics: direction axis (size 8) is split as 2 direction_splits x 4 direction_inds (x, x+y, y, y-x); channel is split into two halves (forward/backward); masks are reshaped to the leaf and applied as 1e3*(1-mask) offsets
task-specific information source: task.masks (raw-shape-derived) + adapted state; THIS IS A DIRECT Task-field read inside forward
source path and lines: layers.py:418-452 (cummax_ 418-423, diagonal 424-446, decorators 447-452), mask plumbing 323-343; call at arc_compressor.py:126-129 (post_norm=True effective per layers.py:65-66)
evidence class: CODE_EXACT
```

Cardinal formula (CODE_EXACT, `layers.py:418-423`): with `m = 1e3*(1-mask)`, `max_ = max(x-m, dim)+m+1e-3`, `min_ = min(x+m, dim)-m-1e-3`, output `= (cummax(x-m, dim)+m - min_)/(max_-min_)*2 - 1`.

## Edge 6 — x -> shift block

```text
caller/callee: forward -> layers.shift (same decorator stack as cummax)
input tensor(s): x[dims] for dims in {(1,1,1,1,1), (1,0,1,1,1)}; shift_weights[layer][dims]; task.masks
output tensor(s): x[dims] + W_up(normalize(directional one-step shift of down(x)))
axis semantics: zero-pad shift by one position along the direction-selected spatial axis; diagonal variant shifts both axes with pad values
task-specific information source: task.masks (multiplied in via directional layer masking at layers.py:334-343) + adapted state
source path and lines: layers.py:467-486; call at arc_compressor.py:130-133
evidence class: CODE_EXACT
```

## Edge 7 — x -> direction_share block

```text
caller/callee: forward -> layers.direction_share (multify + only for dims[2]==1 leaves)
input tensor(s): x[dims]; direction_share_weights[layer][dims][d1][d2] = [W (ch,ch), b]
output tensor(s): stack over d1 of ( x_d1 + sum_d2 coeff[(d2-d1)%8] * affine(z_d2, W[d1][d2]) )
axis semantics: unbind/stack along the direction axis; z = normalize(x) because pre_norm here IS effective (layers.py:503)
task-specific information source: adapted state; weight table is heavily aliased by symmetrize_direction_sharing (initializers.py:128-168)
source path and lines: layers.py:488-524 (coefficients [1,0.2,0.4,0.2,1,0.2,0.4,0.2] at 513); call at arc_compressor.py:136
evidence class: CODE_EXACT
```

## Edge 8 — x -> nonlinear block

```text
caller/callee: forward -> layers.nonlinear (multify + add_residual)
input tensor(s): x[dims]; nonlinear_weights[layer][dims] = [[W_down (ch,16), b],[W_up (16,ch), b]]
output tensor(s): x[dims] + W_up(silu(W_down(x[dims])))
axis semantics: channel remap ch -> 16 -> ch; silu element-wise
task-specific information source: adapted state only
source path and lines: layers.py:526-540; call at arc_compressor.py:139 (pre_norm=True ineffective)
evidence class: CODE_EXACT
```

## Edge 9 — x -> share_down (Task flags and masks can enter here)

```text
caller/callee: forward -> layers.share_down -> share_direction(direction=-1)
input tensor(s): full MultiTensor x; share_down_weights[layer][dims]
output tensor(s): x[dims] + up(normalize(sum over higher leaves of axis-aggregated down(x)))
axis semantics: aggregation over the extra axes of higher leaves; when (task.in_out_same_size or task.all_out_same_size) and the aggregated axis is x (dim==3) or y (dim==4), a MASKED mean sum(x*masks)/(sum(masks)+1e-4) is used with masks combined as 1-(1-m_in)(1-m_out); otherwise plain torch.mean
task-specific information source: task.in_out_same_size / task.all_out_same_size flags and task.masks — a second direct Task read inside forward
source path and lines: layers.py:200-234 (branch at 211 and 221), 252-261 (share_down); call at arc_compressor.py:142
evidence class: CODE_EXACT
```

## Edge 10 — x -> normalize (end of each loop iteration)

```text
caller/callee: forward -> layers.normalize (multify)
input tensor(s): x[dims]
output tensor(s): (x - mean)/sqrt(1e-8 + mean(x^2)) computed over all axes except channel, per channel index
axis semantics: reduction over every non-channel axis (INCLUDING the example axis when present)
task-specific information source: none beyond the stream
source path and lines: layers.py:17-31; call at arc_compressor.py:145
evidence class: CODE_EXACT
```

## Edge 11 — x -> color head

```text
caller/callee: forward -> layers.affine on leaf x[[1,1,0,1,1]]
input tensor(s): x[(1,1,0,1,1)] shape (n_examples, n_colors, n_x, n_y, 16); head_weights = [W (16,2), b (2,)]
output tensor(s): output = affine(x, head_weights, use_bias=False) + 100*head_weights[1]  -> shape (n_examples, n_colors, n_x, n_y, 2)
axis semantics: axis0 example; axis1 color (NO black channel yet); axis2 x; axis3 y; axis4 in/out mode
task-specific information source: adapted state only
source path and lines: arc_compressor.py:148-151
evidence class: CODE_EXACT
```

## Edge 12 — x -> x/y mask heads (shared weights)

```text
caller/callee: forward -> layers.affine on x[[1,0,0,1,0]] and x[[1,0,0,0,1]]
input tensor(s): x-leaf (n_examples, n_x, 16) and y-leaf (n_examples, n_y, 16); the SAME mask_weights object [W (16,2), b (2,)]
output tensor(s): x_mask (n_examples, n_x, 2); y_mask (n_examples, n_y, 2)
axis semantics: per-example per-position mask scores for in/out modes
task-specific information source: adapted state only
source path and lines: arc_compressor.py:152-153 (use_bias=True)
evidence class: CODE_EXACT
```

## Edge 13 — masks -> postprocess_mask (Task shapes enter forward here)

```text
caller/callee: forward -> layers.postprocess_mask
input tensor(s): x_mask, y_mask; task.n_examples, task.n_x, task.n_y, task.shapes
output tensor(s): x_mask/y_mask with -1000 added at positions >= max(in_shape, out_shape) per example per axis
axis semantics: hard out-of-bounds suppression; for the test example the out_shape is the PREDICTED shape from preprocessing
task-specific information source: task.shapes — third direct Task read inside forward
source path and lines: layers.py:542-570; call at arc_compressor.py:156
evidence class: CODE_EXACT
```

Forward returns `(output, x_mask, y_mask, KL_amounts, KL_names)` (`arc_compressor.py:158`). Forward reads **no raw grid data**: its only Task inputs are `task.masks` (Edges 5, 6, 9), the size flags (Edge 9), and `task.shapes`/sizes (Edge 13). Evidence class: CODE_EXACT (signature and call sites; see source_evidence_index E24/E34/E36/E41/E68).

---

## Edge 14 — logits -> black augmentation -> Logger (training-time decode path)

```text
caller/callee: train.take_step -> torch.cat -> Logger.log -> Logger._track_solution
input tensor(s): output (n_examples, n_colors, n_x, n_y, 2)
output tensor(s): logits (n_examples, n_colors+1, n_x, n_y, 2) with a zero black channel at color index 0; Logger.current_logits = logits[n_train:, :, :, :, 1]
axis semantics: color axis gains black at index 0, matching task.colors ordering (colors[0]=0)
task-specific information source: n_train slicing
source path and lines: train.py:51, 112-120; solution_selection.py:56-58
evidence class: CODE_EXACT
```

## Edge 15 — Logger -> discrete solution candidates -> voted prediction

```text
caller/callee: Logger._track_solution -> _postprocess_solution -> best_crop/_best_slice_point -> _update_most_frequent_solutions
input tensor(s): (current_logits, current_x_mask, current_y_mask) and (ema_logits, ema_x_mask, ema_y_mask), EMA decay 0.97
output tensor(s): per-step: two candidate discrete grids + scores; across steps: solution_most_frequent, solution_second_most_frequent
axis semantics: colors = argmax over color axis; uncertainty = logsumexp - amax per cell; crop = argmax slice score -sum(mask[:o])+sum(mask[o:o+L])-sum(mask[o+L:]); L fixed to the predicted query shape when in_out_same_size or all_out_same_size else searched over 1..n; cell indices mapped through task.colors
task-specific information source: task.shapes (crop length), task.colors (index->color), size flags (crop-length branch)
source path and lines: solution_selection.py:60-104, 106-128, 130-162; scores at 73-77; logaddexp accumulation at 81-82
evidence class: CODE_EXACT
```

## Edge 16 — R6A Phase 1 probability capture and evaluator decode (wrapper path)

```text
caller/callee: phase1_condition_solver.run_condition -> canonical_probability; parent process run_phase1_multirepeat_pairing_audit.evaluator_metrics
input tensor(s): Logger.current_logits at each of steps 1401-1500
output tensor(s): per-step softmax over the color axis of test query 0, written into a 30x30x10 canvas indexed by ARC color; evaluator: mean over the 100 snapshots, then NLL at hidden-solution cells; separately, discrete attempts = Logger's two voted solutions
axis semantics: canvas axes (row, col, arc_color); only test example index 0 is captured
task-specific information source: task.colors, task.n_x, task.n_y for canvas placement; hidden solution ONLY in the parent evaluator process
source path and lines: phase1_condition_solver.py:146-155, 195-200, 137-143; run_phase1_multirepeat_pairing_audit.py:220-239, 427-428, 450
evidence class: CODE_EXACT
```

---

## Output-size-affecting branches (consolidated)

1. Predicted query output shape: `in_out_same_size` -> test-input shape; else `all_out_same_size` -> first train output shape (shared list object); else `[max_x, max_y]` (`preprocessing.py:44-62`). CODE_EXACT.
2. `postprocess_mask` hard -1000 beyond per-example `max(in, out)` lengths (`layers.py:558-567`). CODE_EXACT.
3. Decoder crop length: fixed to the predicted shape when `in_out_same_size or all_out_same_size`, else searched over all lengths with mask scores (`solution_selection.py:112-115`). CODE_EXACT.
4. Loss-side size marginalization when `grid_size_uncertain` (`train.py:68-87`). CODE_EXACT.

## Where task information enters forward ONLY through adapted state

Everything except Edges 5/6/9/13: the posteriors, capacities, and all layer weights are trained per task, so after any optimizer step their values are task-informed; forward consumes them with no other data input (`arc_compressor.py:94`, signature `def forward(self)`). DERIVED_EQUIVALENT (composition of CODE_EXACT facts E24, E44, E48, E49; see source_evidence_index E68).
