# R6A Exact-Solve Readout - 10 Tasks

This folder contains a bounded behavioral readout for frozen R6A/CompressARC FULL-condition per-task adaptation.

This report measures exact decoded-query performance after a declared
per-task adaptation protocol. It does not establish reasoning, a shared
latent rule z, cross-task pretraining, shortcut absence, or ARC benchmark
performance beyond the fixed ten-task readout.

## Result

- Protocol-valid task results: 10/10
- Exact full-grid solves: 3/10
- Recovered existing FULL runs: 9
- Fresh frozen reruns: 1
- Fresh reruns passed G1-G5: True
- Protected source scope unchanged: True

## Solved Task IDs

- 0dfd9992
- 1f876c06
- 1c786137

## Nonexact Task IDs

- 3ed85e70
- 929ab4e9
- 981571dc
- 604001fa
- 79fb03f4
- aa4ec2a5
- 00576224

## Not Recoverable / Invalid Task IDs

- none

## Files

- `task_selection_manifest.json` records the deterministic tenth-task selection before its fresh run.
- `run_manifest.json` records the recovery and fresh-run rules plus the fixed seed before execution.
- `protected_scope_before_after.json` records protected source hashes before and after.
- `target_isolation_certificate.json` records target-isolation and G1-G5 checks.
- `exact_solve_table.csv` is the task-level scoreboard.
- `predictions/<task_id>.json` stores predicted grid, gold target grid, mismatch count, query NLL, decoded shape, and condition name FULL for each protocol-valid task.
