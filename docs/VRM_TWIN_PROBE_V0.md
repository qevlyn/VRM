# VRM Twin Probe v0 - Correspondence Capacity Test

This probe trains a fresh VRM V1.1-shaped model from scratch on the already-audited synthetic marker twin suite at:

```text
/mnt/d/VRM/artifacts/twin_task_pilot_v1/
```

It is not ARC-2 training, not V1.2 latent refinement, and not a claim that VRM solves ARC.

## Question

Can the V1.1 evidence-bank and rewrite architecture use coherent support input-output correspondence to predict the correct query output, and does its prediction switch when the support correspondence is swapped to the opposite twin?

## Model Input Contract

The model receives only:

- support input grids;
- support output grids;
- query input grid;
- query input mask.

Candidate grids, candidate labels, target outputs, twin side, base IDs, rule/template metadata, and original pair-order metadata are not passed to `model.forward`. The target output is used only by the supervised direct-grid loss.

## Evaluation Conditions

- `NORMAL`: coherent support pairs and original target.
- `TWIN_SWAP_SUPPORTS`: same query/candidates/target, but support pairs from the opposite twin.
- `PAIR_ORDER_SHUFFLED`: coherent complete support pairs under a deterministic non-identity permutation.
- `SUPPORT_EVIDENCE_PATH_ZEROED`: supplementary/OOD; uses the existing V1.1 `disable_support_evidence` inference flag.

## Commands

Dry run:

```bash
python -m vrm.twin_tasks.run_vrm_probe \
  --mode dry_run \
  --seed 20260628 \
  --steps 25 \
  --batch-size 2 \
  --data /mnt/d/VRM/artifacts/twin_task_pilot_v1 \
  --out /mnt/d/VRM/artifacts/twin_vrm_probe_v0
```

Full run:

```bash
python -m vrm.twin_tasks.run_vrm_probe \
  --mode full \
  --seed 20260628 \
  --max-steps 5000 \
  --batch-size 4 \
  --eval-every 250 \
  --checkpoint-every 250 \
  --learning-rate 2e-4 \
  --warmup-steps 100 \
  --min-learning-rate 2e-5 \
  --grad-clip-norm 1.0 \
  --data /mnt/d/VRM/artifacts/twin_task_pilot_v1 \
  --out /mnt/d/VRM/artifacts/twin_vrm_probe_v0
```

## Interpretation

Normal accuracy alone does not establish correspondence learning. The strongest evidence is high normal candidate score, high paired twin joint normal accuracy, substantial correct mapping switch under `TWIN_SWAP_SUPPORTS`, and little collapse under `PAIR_ORDER_SHUFFLED`.

This is a single-seed capacity probe. Results do not transfer to ARC-2 without further evidence.
