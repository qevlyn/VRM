# Optimization Equations (loss, gradients, update)

Scope: exactly one call of `train.take_step(task, model, optimizer, train_step, train_history_logger)` (`third_party/compressarc_reference/train.py:37-120`), which is the entire inner loop body for both upstream training (`train.py:148-150`, 2000 iterations) and the R6A Phase 1 solver (`r6a_compressarc/scripts/phase1_condition_solver.py:195-196`, 1500 iterations). Every equation carries an evidence class.

Notation introduced by this document only; not a variable name from source: `s_t` = the tuple of all tensors in `model.weights_list`; `T` = the `preprocessing.Task` object; `t` = `train_step`; `E` = `task.n_examples`; `m` in `{0,1}` = in/out mode.

---

## 1. All loss terms

CODE_EXACT (`train.py:54-56, 104, 106`):

```text
L_t = total_KL + 10 * reconstruction_error
total_KL = sum over the 18 leaves of sum over elements of KL_leaf
reconstruction_error = - sum over (example, mode) pairs of logprob(example, mode)
```

There are exactly two loss terms. There are no weight-decay, auxiliary, or regularization terms in source (CODE_EXACT: full read of `take_step`; the only additions to `loss` are at `train.py:106`).

## 2. KL term

CODE_EXACT (`layers.py:71-131`; consumed at `train.py:54-56`). Per leaf with posterior `[mean, lca]`, capacity vector `tc` (shape `[4]`), and `D = numel(mean)`:

```text
tc'         = 10 * tc                                            (layers.py:93)
C_global    = exp(tc') * 10000 + 0.5                             (layers.py:88-91, 101)
out_scale   = 1 - exp(-2 * C_global / D)                         (layers.py:102)
lc          = tc' + lca - mean_over_all_but_last_axis(lca)       (layers.py:106-108)
C_local     = exp(lc) * 10000 + 0.5                              (layers.py:109)
noise_std   = exp(-C_local / D);  noise_var = noise_std^2        (layers.py:113-114)
signal_std  = sqrt(1 - exp(-2*C_local/D))  [where(arg>20 -> 1)]  (layers.py:115-116)
signal_var  = 1 - noise_var                                      (layers.py:117)
nm          = (mean - mean_abl(mean)) / sqrt(mean_abl(nm^2 + 1e-8))   (layers.py:120-121)
KL_leaf     = 0.5*(noise_var + signal_var*nm^2 - 1) + C_local/D  (layers.py:130)
```

`mean_abl` = mean over all axes except the trailing channel axis, INCLUDING the example axis when present (CODE_EXACT, `layers.py:84`). The `+1e-8` placement inside the mean is algebraically equal to adding `1e-8` after the mean (DERIVED_EQUIVALENT: mean of a constant is the constant).

The sampled activation that carries gradients into the network is `z = out_scale * (signal_std * nm + noise_std * eps)`, `eps ~ torch.randn` (CODE_EXACT, `layers.py:124-125`). There is **no zero-noise branch in upstream source**; `eps` is drawn every forward (CODE_EXACT).

## 3. Reconstruction term, cell participation, and masking

CODE_EXACT (`train.py:59-104`). The loop runs over `example in 0..E-1` and `mode in {0,1}` and **skips test outputs**:

```text
if example >= n_train and mode == 1: continue        (train.py:62-63)
```

Participating grids: every train input, every train output, every test INPUT. The hidden query output participates in **no** solver-side term (see section 6).

Per participating (example, mode) with true shape `(H, W) = task.shapes[example][mode]`:

```text
grid_size_uncertain = not (in_out_same_size
                           or (all_out_same_size and mode==1)
                           or (all_in_same_size  and mode==0))      (train.py:68; 'and' binds tighter than 'or')

c1 = 0.01 ** max(0, 1 - t/100)  if uncertain else 1                 (train.py:69-72)
c2 = 0.10 ** max(0, 1 - t/100)  if uncertain else 1                 (train.py:99-102)

x_lp(o)  = -sum(c1*xm[:o]) + sum(c1*xm[o:o+H]) - sum(c1*xm[o+H:])   (train.py:23-35, 76)
x_logZ   = logsumexp over offsets o of x_lp(o)                       (train.py:34)
           [if uncertain: logsumexp additionally over ALL lengths 1..n_x  (train.py:79-87)]
(y_lp, y_logZ analogously with W and ym)                             (train.py:77, 84-87)

CE(ox,oy) = cross_entropy( logits[example, :, ox:ox+H, oy:oy+W, mode],
                           task.problem[example, :H, :W, mode], reduction='sum' )   (train.py:94-96)

score(ox,oy) = x_lp(ox) - x_logZ + y_lp(oy) - y_logZ - CE(ox,oy)    (train.py:93-96)
logprob      = logsumexp over (ox,oy) of ( c2 * score ) / c2         (train.py:103)
reconstruction_error -= ... accumulates -logprob                     (train.py:104)
```

where `xm = x_mask[example, :, mode]`, `ym = y_mask[example, :, mode]`, and `logits` are the black-augmented logits (`train.py:51`: a zero channel is concatenated at color index 0, so the class axis has `n_colors+1` entries aligned with `task.colors`).

Masked-out cells: the color cross-entropy itself is computed over the full `H x W` crop with no cell mask; cells are excluded only by (a) the skip rule for test outputs and (b) crop boundaries `H, W`. Spatial masking acts through the mask-score offset weighting, not through per-cell exclusion (CODE_EXACT, `train.py:90-98`).

## 4. Flag effects on loss branches

CODE_EXACT:

- `in_out_same_size` true makes every (example, mode) size-certain (`train.py:68`), so `c1 = c2 = 1` and offset partitions use only the true length.
- `all_out_same_size` certifies only `mode==1`; `all_in_same_size` certifies only `mode==0` (same line; precedence as written).
- When uncertain, the log-partitions marginalize over all crop lengths (`train.py:79-87`) and both annealing coefficients decay their effect until `t >= 100`.
- These flags are computed in `preprocessing.py:48-50` and are data properties of the (possibly rewritten) challenge; the R6A `ALL_PAIRING_SWAP` condition can change them because it reassigns outputs across examples (`r6a_common.py:121-130`) — flag recomputation happens inside `Task` construction for whatever challenge JSON the solver receives.

## 5. Gradient set

Notation introduced by this document only; not a variable name from source:

```text
g_t = grad_{s_t} L_t
```

`loss.backward()` (`train.py:107`) populates `.grad` on exactly those `weights_list` tensors that participated in the graph: all 18 posterior means, all 18 local capacity adjustments, all 18 target capacity vectors, decode weights, share/softmax/cummax/shift/direction/nonlinear weights actually used by their leaves, head weights, and mask weights (CODE_EXACT for the call; participation DERIVED_EQUIVALENT from the forward graph). Tensors registered in `weights_list` but orphaned by `symmetrize_xy` / `symmetrize_direction_sharing` rebinding (`initializers.py:122-126, 128-168` after registration at `initializers.py:49`) receive no gradient because they appear in no computation (DERIVED_EQUIVALENT).

## 6. Hidden query target visibility (answered per stage)

- Task construction: absent — the test-output fill is skipped (`preprocessing.py:106-107`) and Phase 1 passes `solution=None` (`phase1_condition_solver.py:178`), so `task.solution is None` in the solver process (CODE_EXACT).
- Loss: absent — test outputs are skipped (`train.py:62-63`) (CODE_EXACT).
- Optimizer: absent — gradients derive only from `L_t` (CODE_EXACT `train.py:106-108`).
- Forward: absent — `forward()` reads no grid data at all (`arc_compressor.py:94-158`) (CODE_EXACT).
- Decoder (Logger): absent — voting uses logits, masks, shapes, colors only (`solution_selection.py:54-162`) (CODE_EXACT).
- Evaluator only: present — the parent process loads `arc-agi_*_solutions.json` and computes NLL/accuracy after the solver finishes (`run_phase1_multirepeat_pairing_audit.py:427-428, 450, 220-239`); the solver process additionally firewalls `open()` against solution paths (`phase1_condition_solver.py:24-37`) (CODE_EXACT).

## 7. Optimizer and registered state

CODE_EXACT:

```text
optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
```

(`train.py:137`; identically `phase1_condition_solver.py:182` with values from the frozen config, and `run_phase1_multirepeat_pairing_audit.py:132`.) Registered state = every tensor ever appended to `Initializer.weights_list` (`initializers.py:22, 29, 49, 68, 90`), including the orphaned pre-symmetrization duplicates.

Update (notation introduced by this document only):

```text
s_{t+1} = AdamStep(s_t, g_t; lr=0.01, betas=(0.5, 0.9))
```

The Adam recurrence itself (moment estimates, bias correction, epsilon) is torch library semantics, not repository source — SCHEMATIC. Repository-source facts: `optimizer.zero_grad()` before forward and after the step (`train.py:49, 109`), `optimizer.step()` once per `take_step` (`train.py:108`). Optimizer state existence and shape are library-managed; R6A's `optimizer_state_hash` (`phase1_condition_solver.py:60-72`) shows it contains `param_groups` plus per-parameter tensor state, without defining the math (CODE_EXACT about the hash function only).

## 8. What can change after one take_step

- Updated: every `weights_list` tensor with a non-None gradient — posterior means, local capacity adjustments, target capacities, and all reachable layer/head/mask weights (DERIVED_EQUIVALENT as in section 5). There is no frozen subset in source: **the entire model is trainable per task**.
- Read-only during optimization: `task.problem`, `task.masks`, `task.shapes`, `task.colors`, flags (never assigned after `Task.__init__`) (CODE_EXACT: no assignment sites outside `preprocessing.py:15-156`).
- Derived-only: forward activations, `KL_amounts`, `Logger` accumulators (`solution_selection.py:40-104`).

## 9. KL/capacity behavior in the update

`total_KL` back-propagates into `mean` (through `nm` and `KL_leaf`), into `lca` (through `C_local`), and into `tc` (through both `C_local` and, via `out_scale` and the reconstruction path, the sampled `z`) (DERIVED_EQUIVALENT from the equations of section 2; all quantities are differentiable subexpressions of `L_t`). No capacity scheduling or clamping exists outside these formulas (CODE_EXACT: no other writes to these tensors in files read).

## 10. Source-exact versus derived summary

- CODE_EXACT: sections 1, 2 (equations), 3, 4, 6, 7 (registration and hyperparameters), 8 (read-only list).
- DERIVED_EQUIVALENT: gradient-participation set; orphan no-gradient statement; KL gradient routing; the `+1e-8` equivalence.
- SCHEMATIC: the Adam recurrence internals.
- UNRESOLVED items related to this file are listed in `unresolved_or_ambiguous_points.md` (U1: as-executed Adam math across torch versions; U2: RNG stream values; U4: fully-masked-slice edge algebra).
