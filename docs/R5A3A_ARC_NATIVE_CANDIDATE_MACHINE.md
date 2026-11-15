# R5A-3A ARC-Native Structural Candidate Machine v0

R5A-3A is a new deterministic, no-training sibling architecture. It does not modify V2, Topology R2, R5A-2D-R1, or R5A-2D-R2.

The goal is not immediate ARC solving. The goal is a smallest ARC-native candidate generator that accepts raw variable-size ARC grids and emits bounded deterministic candidate output grids.

## Contract

- Grid height and width: `1..30`
- Palette values: `0..9`
- Support pair count: `1..10`
- Query count: `1..10`
- Input and output shapes may differ
- No crop, pad, resize, recolor, remap, or normalization before architecture input
- No training, neural model, checkpoint, task ID runtime input, LLM call, router, verifier, or test-time training

## Named Stages

1. A. scene hypotheses
2. B. object/region extraction
3. C. support-pair delta extraction
4. D. structural program enumeration
5. E. program execution
6. F. visible-support consistency scoring
7. G. top-K deduplicated candidate output grids

## Frozen v0 Limits

- `K = [1, 4, 16, 64, 128]`
- Maximum program depth: `2`
- Maximum scene hypotheses: `1`
- Maximum program enumeration count before ranking: `8` per scene hypothesis

This is a development measurement substrate only. A synthetic/property pass is mechanism validation, not ARC success. Candidate recall is not a final-answer solver and does not justify claims about reasoning.

Run:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.run_r5a3a_arc_native_candidate_machine
pytest -q
```
