# R5A-3B-P0 Relation Constraint Causal Pilot

R5A-3B-P0 is an isolated sibling experiment for a 24-task ARC-AGI-2 development pilot. It compares five candidate-generation and pruning conditions from the same raw visible support information:

- `OBJECT_ONLY`
- `RELATION_NO_PRUNE`
- `RELATION_SOFT_CONSTRAINT`
- `RELATION_HARD_CONSTRAINT`
- `MATCHED_RANDOM_PRUNE`

The pilot does not train a model, resume a model, use task ids as runtime inputs, inspect ARC test outputs, run a full 4357-fold audit, or modify V2, topology R2, R5A-2D, or R5A-3A sources or artifacts.

## Fixed Budgets

- `MAX_PARSE_HYPOTHESES = 4`
- `MAX_RAW_PROGRAMS_PER_PARSE = 8`
- `MAX_RAW_CANDIDATES_PER_FOLD = 32`
- `TOP_K = [1, 4, 16, 64, 128]`
- `MAX_PROGRAM_DEPTH = 2`

## Mechanism

Each query grid is parsed with up to four deduplicated hypotheses from modal background and color 0, using 4- and 8-connectivity. Object graphs expose node attributes and edge predicates for same color, same shape, same size, nearest, alignment, touching, containment, relative direction, and border relation. Nearest ties are represented and are not silently selected.

Constraints are acquired from visible support pairs only. A hard constraint requires universal support, unambiguous correspondence, contrastive visible evidence, and a mechanically detectable predicate. Otherwise the constraint remains soft.

## Artifacts

The runner writes artifacts under:

```text
/mnt/d/VRM/artifacts/r5a3b_relation_constraint_pilot/
```

The required artifact set includes the pilot manifest, selection spec, pre-hidden commitment, parse/object/constraint reports, raw and condition ledgers, random-prune report, runtime profile, recall metrics, causal comparison, representative case cards, determinism verification, final report, and pytest result.

## Run

Focused tests:

```bash
cd /mnt/d/VRM
pytest -q tests/test_r5a3b_relation_constraint_pilot.py
```

Pilot:

```bash
cd /mnt/d/VRM
python -m vrm.twin_tasks.run_r5a3b_relation_constraint_pilot
```

## Final Report Actions

The final report records exactly one narrow next action:

- `A`: stop at the pilot because no actionable relation/constraint signal appeared;
- `B`: run a frozen broader R5A-3B audit from the same pre-registered design;
- `C`: repair constraint mechanics before any broader audit;
- `D`: create a new version with an expanded grammar and a new pre-registration.
