# R6A CompressARC Preflight Report

- official repository: https://github.com/iliao2345/CompressARC
- commit: 83a22218024d46273eb32b769a906340202ffb4d
- paper: ARC-AGI Without Pretraining (Isaac Liao, Albert Gu)
- arXiv: https://arxiv.org/abs/2512.06104
- OpenReview: https://openreview.net/forum?id=TbbMyr3E0x
- license: MIT License
- official requirements: matplotlib==3.10.0, numpy==2.2.2, torch==2.5.1, tqdm==4.66.6
- local compatibility note: separate /home/quyen/.venvs/compressarc could not be created because WSL Python lacks ensurepip; Python 3.14 could not install pinned matplotlib 3.10.0 without a compiler, so the existing CUDA VRM venv with matplotlib 3.11.0/tqdm 4.68.3 was used as a non-algorithmic compatibility runtime.

## Source Audit

Safe solver path: solve_task.py constructs preprocessing.Task(task_name, problems[task_name], None).

Unsafe/scoring paths:
- preprocessing.preprocess_tasks loads *_solutions.json for training/evaluation splits.
- train.py main uses preprocess_tasks and true solution hashes for plotting accuracy.
- list_solved_puzzles.py uses preprocess_tasks to compare against true solution hashes.
- scoring.py loads solutions for evaluator-side scoring.

## Preflight Gates

- OFFICIAL_SOURCE_IDENTIFIED: PASS
- RUNNABLE_COMMAND_IDENTIFIED: PASS
- QUERY_TARGET_FIREWALL_IMPLEMENTED: PASS
- QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL: QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL_PASSED
- PROTECTED_VRM_MANIFEST_CREATED: PASS

The objective in train.take_step is total_KL + 10 * reconstruction_error. total_KL is the latent/code-length/MDL-like term recorded per multitensor component. reconstruction_error is support/query-input reconstruction NLL-like error; hidden query outputs are skipped during loss.

Focused pytest: 10 passed in 0.33s

## Explicit Preflight Answers

1. Official paper/codebase found: ARC-AGI Without Pretraining by Isaac Liao and Albert Gu; repository https://github.com/iliao2345/CompressARC.
2. Commit used: 83a22218024d46273eb32b769a906340202ffb4d (added code for listing the puzzles that were solved).
3. Dataset files expected/read by the implementation: dataset/arc-agi_evaluation_challenges.json, dataset/arc-agi_evaluation_solutions.json, dataset/arc-agi_test_challenges.json, dataset/arc-agi_training_challenges.json, dataset/arc-agi_training_solutions.json, and dataset/sample_submission.json. The R6A solver wrapper reads only ARC-AGI-1 evaluation challenge JSON; the evaluator reads evaluation solutions after inference.
4. Source target access audit: solve_task.py uses Task(task_name, problems[task_name], None) and does not load solutions. preprocessing.preprocess_tasks, train.py main, list_solved_puzzles.py, and scoring.py can load solutions and are not used in the solver process.
5. Safe solver path: r6a_compressarc/scripts/solve_challenge_only.py, which imports official modules and constructs preprocessing.Task(args.task_id, problem, None) after installing a solver-side solution-file firewall.
6. Smallest official end-to-end command: README documents python analyze_example.py, then interactive split/task selection, for 1500 steps. R6A noninteractive equivalent: /home/quyen/.venvs/vrm/bin/python r6a_compressarc/scripts/solve_challenge_only.py --compressarc-root /mnt/d/VRM/third_party/compressarc_reference --challenge-json <challenge.json> --task-id <task> --steps 1500 --seed 0 --condition FULL --output-json <solver.json> --device cuda, followed by evaluator-only scoring.
7. Prior claims verified/unknown: verified locally only that the official reference code can run 1500-step ARC-AGI-1 evaluation smoke tasks and can exactly solve smoke task 903d1b4a in this run. No leaderboard or aggregate benchmark reproduction is claimed. Paper/README claims are provenance only.
8. Shape changes: source supports output-shape prediction through Task._predict_solution_shapes, mask terms, and Logger.best_crop; B1 permits output-shape reassignment, while B2 is restricted to same-shape output reassignment because multitensor dimensions depend on observed support/test shapes.
9. Multitensor assumptions: Task fixes n_examples, color set, n_x, and n_y from the challenge. Reassigning outputs can alter per-pair output-shape semantics; B2 exists to avoid invalidating each support slot output-shape assignment.
10. Objective terms: train.take_step computes loss = total_KL + 10 * reconstruction_error. total_KL is the latent/code-length/MDL-like term from decoded multitensor latents. reconstruction_error is support/train plus visible query-input reconstruction; hidden query output is skipped. Adam weight decay/explicit extra regularization is not used in the inspected optimizer path.

## Frozen ARC-AGI-1 Pilot Manifests

- Phase 0 smoke selected IDs: 903d1b4a, e66aafb8, 67b4a34d.
- Phase 1 selected IDs: 604001fa, 79fb03f4, 981571dc, 3ed85e70, aa4ec2a5, 929ab4e9.
- Pool label: UNTOUCHED_ARC_AGI_1_EVALUATION_POOL.
- Outcomes read before freeze: false in both manifests.
