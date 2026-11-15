# R5A-4A Neural VRM Interface Audit

This audit was written before adding the R5A-4A runner. It inspects the
existing neural branch in `/mnt/d/VRM/vrm/model.py` and the existing ARC
collation/loss path in `/mnt/d/VRM/vrm/data.py`, `/mnt/d/VRM/vrm/metrics.py`,
and `/mnt/d/VRM/vrm/train.py`.

## Exact model class

- Class: `VRM` in `/mnt/d/VRM/vrm/model.py:178`.
- Config dataclass: `VRMConfig` in `/mnt/d/VRM/vrm/model.py:27`.
- Current default architecture string in the model docstring:
  `VRM V1.1 - Masked Query Spatial Evidence Rewrite`
  (`/mnt/d/VRM/vrm/model.py:178`).

## Forward signature

`VRM.forward` is keyword-only (`/mnt/d/VRM/vrm/model.py:382`):

```python
forward(
    *,
    support_evidence_colors,
    support_evidence_rows,
    support_evidence_cols,
    support_evidence_roles,
    support_evidence_pair_indices,
    support_evidence_mask,
    query_inputs,
    query_input_mask,
    disable_support_evidence=False,
)
```

No `task_id`, `puzzle_id`, filename, dataset index, or persistent task key is
accepted by this forward signature.

## Input tensors consumed by the model

The model consumes exactly the tensors returned by
`model_inputs_from_batch` in `/mnt/d/VRM/vrm/data.py:376`:

- `support_evidence_colors`
- `support_evidence_rows`
- `support_evidence_cols`
- `support_evidence_roles`
- `support_evidence_pair_indices`
- `support_evidence_mask`
- `query_inputs`
- `query_input_mask`

The collator also returns `task_ids`, `query_pair_indices`, and
`source_splits` for bookkeeping (`/mnt/d/VRM/vrm/data.py:370`), but
`model_inputs_from_batch` does not forward them to the model.

The held-out target tensors `query_targets`, `query_target_mask`,
`height_targets`, and `width_targets` are created by collation
(`/mnt/d/VRM/vrm/data.py:340`) and consumed by the loss after inference, not
by `VRM.forward`.

## Output tensors returned by the model

`VRM.forward` returns a dictionary at `/mnt/d/VRM/vrm/model.py:436` with:

- `color_logits`
- `color_log_probs`
- `logits`
- `color_probabilities`
- `height_logits`
- `width_logits`
- `final_canvas`
- `final_rewrite_state`
- `support_evidence_bank`
- `support_evidence_mask`
- `rule_tokens`
- `copy_gate`
- `copy_attention`
- `copy_distribution`
- `paint_distribution`
- `copy_entropy`
- `paint_entropy`

## How inputs enter computation

- Color tokens enter through `color_embedding`
  (`/mnt/d/VRM/vrm/model.py:194`, `/mnt/d/VRM/vrm/model.py:245`,
  `/mnt/d/VRM/vrm/model.py:275`).
- Rows enter through `row_embedding`
  (`/mnt/d/VRM/vrm/model.py:195`, `/mnt/d/VRM/vrm/model.py:246`,
  `/mnt/d/VRM/vrm/model.py:276`).
- Columns enter through `col_embedding`
  (`/mnt/d/VRM/vrm/model.py:196`, `/mnt/d/VRM/vrm/model.py:247`,
  `/mnt/d/VRM/vrm/model.py:277`).
- Roles enter through `role_embedding`
  (`/mnt/d/VRM/vrm/model.py:197`, `/mnt/d/VRM/vrm/model.py:278`).
- Support-pair indices enter through `support_pair_embedding`
  (`/mnt/d/VRM/vrm/model.py:198`, `/mnt/d/VRM/vrm/model.py:279`).
  These are visible support-slot/pair indices, not task identifiers.
- Query grids enter as flattened, padded color tokens through
  `query_inputs` and `query_input_mask`
  (`/mnt/d/VRM/vrm/model.py:242`, `/mnt/d/VRM/vrm/model.py:402`).
- Support masks enter through `support_evidence_mask`
  (`/mnt/d/VRM/vrm/model.py:290`, `/mnt/d/VRM/vrm/model.py:301`).
- Query masks enter through `query_input_mask`
  (`/mnt/d/VRM/vrm/model.py:396`, `/mnt/d/VRM/vrm/model.py:346`).

## Task-ID and persistent-state audit

No task ID, puzzle ID, dataset index, filename, cache key, or persistent
task embedding reaches `VRM.forward`.

Allowed bookkeeping-only metadata is present in the collation output:

- `task_ids` (`/mnt/d/VRM/vrm/data.py:370`)
- `query_pair_indices` (`/mnt/d/VRM/vrm/data.py:371`)
- `source_splits` (`/mnt/d/VRM/vrm/data.py:372`)
- `source_file` in `ArcTask`/`ArcEpisode`
  (`/mnt/d/VRM/vrm/data.py:84`, `/mnt/d/VRM/vrm/data.py:94`)

These fields are not included by `model_inputs_from_batch`.

The model state persists across optimizer steps in the usual PyTorch
parameter state. There is no per-task mutable memory, task lookup table,
or persistent task-specific embedding in `VRM`.

Audit status: `ID_FREE_AUDIT_PASS`.

## Output height and width decoding

Height and width use learned shape tokens:

- `height_shape_token` and `width_shape_token`
  (`/mnt/d/VRM/vrm/model.py:200`).
- The final hidden states at positions 0 and 1 are passed through
  `height_head` and `width_head`
  (`/mnt/d/VRM/vrm/model.py:230`, `/mnt/d/VRM/vrm/model.py:432`).
- Decoding is `argmax + 1`, matching target classes `height - 1` and
  `width - 1` created in collation
  (`/mnt/d/VRM/vrm/data.py:346`, `/mnt/d/VRM/vrm/train.py:478`).

## Color prediction decoding

Output-cell color distributions are produced by `copy_and_paint_log_probs`
(`/mnt/d/VRM/vrm/model.py:337`). The path combines:

- query-copy attention over `query_inputs` using `query_input_mask`;
- a learned paint distribution from `paint_head`;
- a learned scalar copy gate from `copy_gate_head`.

The final color distribution is logged as `color_log_probs`
(`/mnt/d/VRM/vrm/model.py:367`) and decoded with `argmax`
(`/mnt/d/VRM/vrm/train.py:473`).

## Existing loss

The existing loss is `factorized_shape_color_loss` in
`/mnt/d/VRM/vrm/metrics.py:25`.

It computes:

- color negative log-likelihood over the true target rectangle;
- height cross entropy;
- width cross entropy;
- total loss as
  `color_loss + shape_loss_weight * 0.5 * (height_loss + width_loss)`.

The existing default `shape_loss_weight` is `0.5`
(`/mnt/d/VRM/vrm/train.py:2348`).

## Existing train entrypoint

The existing neural train entrypoint is:

```bash
python -m vrm.train --mode v1_1_capacity50
```

It is implemented by `run_v1_1_capacity50`
(`/mnt/d/VRM/vrm/train.py:1035`) and dispatched by `main`
(`/mnt/d/VRM/vrm/train.py:2366`).

Existing optimizer conventions include AdamW, batch size 1, learning rate
`2e-4`, minimum learning rate `2e-5`, 100 warmup steps, gradient clipping at
`1.0`, beta1 `0.9`, beta2 `0.95`, and weight decay `0.01`
(`/mnt/d/VRM/vrm/train.py:1047`, `/mnt/d/VRM/vrm/train.py:2343`).

## Existing checkpoint format

The existing checkpoint writer is `save_checkpoint` in
`/mnt/d/VRM/vrm/train.py:797`.

The checkpoint payload contains:

- `model_state_dict`
- `optimizer_state_dict`
- `config`
- `step`
- `extra`

R5A-4A will use the same model/config state layout in a new additive artifact
directory, without writing to the older V1.1 checkpoint locations.
