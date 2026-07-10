# Unresolved or Ambiguous Points

Static-inspection residue only. No runtime observation has been performed for any item.

---

UNRESOLVED-01

Question: What is the exact as-executed Adam update recurrence (moment handling, bias correction, epsilon, foreach/fused kernel selection) for `torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))`?
Why unresolved: The repository contains only the constructor call and `optimizer.step()` (`train.py:137, 108`; `phase1_condition_solver.py:182`). The update math lives in the torch library, and two different torch versions are relevant (official pin `torch==2.5.1` per `r6a_compressarc/source_provenance.json`; the Phase 1 report header records `torch 2.12.1+cu130` in the execution environment), whose optimizer internals may differ.
Source inspected: third_party/compressarc_reference/train.py:49-109; r6a_compressarc/scripts/phase1_condition_solver.py:182-183; r6a_compressarc/source_provenance.json.
What static inspection would resolve it: Reading the installed torch version's `torch/optim/adam.py` for each environment.
What runtime observation would resolve it: Hashing parameters before/after a single step against a hand-computed Adam step.
No runtime observation has been performed.

---

UNRESOLVED-02

Question: What are the concrete noise values `eps` consumed by `channel_layer` at each leaf and step under a given seed/device?
Why unresolved: The draw order is statically determined (one `torch.randn` per leaf, leaves in little-endian dims order; `layers.py:124`, `multitensor_systems.py:70-83`), but the values depend on the RNG engine state, device, and dtype at runtime, including CUDA generator behavior under the Phase 1 restored RNG states (`phase1_condition_solver.py:75-80`).
Source inspected: third_party/compressarc_reference/layers.py:124; third_party/compressarc_reference/multitensor_systems.py:70-83; r6a_compressarc/scripts/phase1_condition_solver.py:75-80.
What static inspection would resolve it: Nothing fully; the generator algorithm is library/device-defined.
What runtime observation would resolve it: Recording the draws (as the R6A Phase 2 noise-tape harness does in its own scope).
No runtime observation has been performed.

---

UNRESOLVED-03

Question: What are the exact tie-breaking and reduction-order semantics of `torch.cummax`, `torch.max`, and `torch.min` as used in `cummax_`/`diagonal_cummax_` under float32 CUDA execution?
Why unresolved: `layers.py:418-446` fixes the mathematical intent, but kernel-level ordering/tie behavior is library- and device-dependent and can change last-bit results and argmax selections between environments.
Source inspected: third_party/compressarc_reference/layers.py:418-452.
What static inspection would resolve it: Reading the specific torch kernel implementations for the executing device and version.
What runtime observation would resolve it: Bit-level comparison of outputs across devices/dtypes for fixed inputs.
No runtime observation has been performed.

---

UNRESOLVED-04

Question: What do `cummax_`/`shift_` outputs mean on fully-masked slices (rows/columns where `mask == 0` everywhere) and on out-of-bounds cells after the `(x - min_)/(max_ - min_)` rescale?
Why unresolved: With `m = 1e3*(1-mask)`, fully-masked slices make `max_` and `min_` degenerate combinations of the ±1e3 offsets and the ±1e-3 guards (`layers.py:418-423`); the sign and scale of the resulting cells is statically derivable in principle but is sensitive to the interaction with the later post-norm and residual add, and this document has not carried out that case analysis to a verified conclusion. The diagonal variant additionally divides by `max_-min_+1e-5` (`layers.py:446`) with a final `*masks` factor, which zeroes masked cells there but not in the cardinal variant.
Source inspected: third_party/compressarc_reference/layers.py:334-343, 418-446.
What static inspection would resolve it: A complete symbolic case analysis of masked/unmasked cell combinations for both variants.
What runtime observation would resolve it: Printing the block output for a task with heterogeneous shapes.
No runtime observation has been performed.

---

UNRESOLVED-05

Question: Which exact `weights_list` positions are orphaned by `symmetrize_xy` and `symmetrize_direction_sharing` (count and identity per leaf/layer)?
Why unresolved: The rebinding rules are explicit (`initializers.py:122-126, 128-168`) and registration precedes rebinding (`initializers.py:49`), so orphans certainly exist; but the full enumeration requires walking all 18 leaves x 4 layers x 8x8 direction pairs against the representative-map rules, which this document has not done exhaustively.
Source inspected: third_party/compressarc_reference/initializers.py:32-50, 122-168; third_party/compressarc_reference/arc_compressor.py:61-91.
What static inspection would resolve it: A mechanical enumeration of the rebinding map over all (dims, dir1, dir2) cells and all symmetrized weight families.
What runtime observation would resolve it: Comparing `id()` sets of multitensor-reachable tensors versus `weights_list` members (as the R6A v8/v10 harnesses did in their own scope).
No runtime observation has been performed.

---

UNRESOLVED-06

Question: Is the shared-list aliasing created by the `all_out_same_size` branch (`default_shape = self.shapes[0][1]` assigned to every test out-shape; `preprocessing.py:55-58`) ever observable, i.e., is any element of `task.shapes` mutated after `Task.__init__`?
Why unresolved: No assignment to `task.shapes` elements outside `preprocessing.py:15-62` was found in the files read, so the aliasing appears benign; but "no mutation anywhere" is a whole-program property, and R6A wrapper scripts pass `task` into library code paths this atlas did not exhaustively scan for in-place mutation of nested lists.
Source inspected: third_party/compressarc_reference/preprocessing.py:44-62; all files in reviewed_source_inventory.csv.
What static inspection would resolve it: A global write-site search for `shapes[` element assignment across every module importable in the solver process.
What runtime observation would resolve it: Identity and content checks on `task.shapes` before/after a solve.
No runtime observation has been performed.
