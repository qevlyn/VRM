# R4A-1P - Forensic Closure and Frozen Length-Three Composition Probe

R4A-1P is an evaluation-only probe of the frozen R4A-0R V2 direct support-conditioned pointer model. It does not train, resume training, alter model architecture, write checkpoints, or select checkpoints.

The only approved checkpoints are:

- `/mnt/d/VRM/checkpoints/composition_r4a0r_v2/composition_r4a0r_seed_20260628_best_dev.pt`
- `/mnt/d/VRM/checkpoints/composition_r4a0r_v2/composition_r4a0r_seed_20260629_best_dev.pt`
- `/mnt/d/VRM/checkpoints/composition_r4a0r_v2/composition_r4a0r_seed_20260630_best_dev.pt`

The probe answers two narrow questions:

- Whether the old pointer-source metric is exact source-coordinate routing rather than same-color matching.
- Whether the frozen support-conditioned pointer model generalizes from single-step and two-step evidence to selected functionally distinct length-three transformations inside the same controlled grammar.

The catalog enumerates all 27 length-three words over `C`, `F`, and `T`, groups them by exact unrestricted integer affine signature, excludes functions equivalent to identity, single-step, or two-step functions, and keeps only canonical groups that are universally feasible across the frozen R4A-0R V2 train-visible and sealed visual query types.

Run dry-run:

```bash
python -m vrm.twin_tasks.run_composition_r4a1p \
  --mode dry_run \
  --seeds 20260628,20260629,20260630 \
  --batch-size 128 \
  --max-batches 4 \
  --data-out /mnt/d/VRM/artifacts/composition_r4a1p_v1
```

Run full evaluation:

```bash
python -m vrm.twin_tasks.run_composition_r4a1p \
  --mode full \
  --seeds 20260628,20260629,20260630 \
  --batch-size 128 \
  --data-out /mnt/d/VRM/artifacts/composition_r4a1p_v1
```

Passing this probe would only support the claim that the frozen direct support-conditioned pointer model generalized through length three in this controlled grammar. It would not prove explicit primitive execution, recovered intermediate states, a general hidden toolbox, ARC solving, or that richer variable-depth execution work is unnecessary.
