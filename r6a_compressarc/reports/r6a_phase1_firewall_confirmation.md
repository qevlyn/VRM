# R6A Phase1 Firewall Confirmation

- firewall preflight passed: True
- solver working directory: /mnt/d/VRM
- solver input paths: ['/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json', '/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt']
- solver commands: `[["/home/quyen/.venvs/vrm/bin/python", "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py", "--compressarc-root", "/mnt/d/VRM/third_party/compressarc_reference", "--challenge-json", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json", "--initialization-artifact", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt", "--task-id", "903d1b4a", "--stratum", "FIREWALL_CPU_NEGATIVE", "--condition", "FIREWALL_CPU_PREFIX", "--steps", "20", "--output-json", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_solver.json", "--probability-npz", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_probability.npz", "--device", "cpu"], ["/home/quyen/.venvs/vrm/bin/python", "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py", "--compressarc-root", "/mnt/d/VRM/third_party/compressarc_reference", "--challenge-json", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json", "--initialization-artifact", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt", "--task-id", "903d1b4a", "--stratum", "FIREWALL_CPU_NEGATIVE", "--condition", "FIREWALL_CPU_PREFIX", "--steps", "20", "--output-json", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_solver.json", "--probability-npz", "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_probability.npz", "--device", "cpu"]]`
- solution JSON path absent from solver inputs: True
- solution JSON path absent from solver commands: True
- CPU dummy-target negative control byte-identical: True
- CPU dummy probability tensors byte-identical: True
- evaluator metrics differ: True

## Static Proof

```json
{
  "data_path_environment_variables": {
    "XDG_DATA_DIRS": "/usr/local/share:/usr/share:/var/lib/snapd/desktop"
  },
  "evaluator_only_solution_paths": [
    "/mnt/d/VRM/third_party/compressarc_reference/dataset/arc-agi_evaluation_solutions.json"
  ],
  "mounted_or_copied_files": [
    "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
    "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt"
  ],
  "solution_json_path_absent_from_data_path_env_vars": true,
  "solution_json_path_absent_from_solver_commands": true,
  "solution_json_path_absent_from_solver_inputs": true,
  "solver_commands": [
    [
      "/home/quyen/.venvs/vrm/bin/python",
      "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py",
      "--compressarc-root",
      "/mnt/d/VRM/third_party/compressarc_reference",
      "--challenge-json",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
      "--initialization-artifact",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt",
      "--task-id",
      "903d1b4a",
      "--stratum",
      "FIREWALL_CPU_NEGATIVE",
      "--condition",
      "FIREWALL_CPU_PREFIX",
      "--steps",
      "20",
      "--output-json",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_solver.json",
      "--probability-npz",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_probability.npz",
      "--device",
      "cpu"
    ],
    [
      "/home/quyen/.venvs/vrm/bin/python",
      "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py",
      "--compressarc-root",
      "/mnt/d/VRM/third_party/compressarc_reference",
      "--challenge-json",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
      "--initialization-artifact",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt",
      "--task-id",
      "903d1b4a",
      "--stratum",
      "FIREWALL_CPU_NEGATIVE",
      "--condition",
      "FIREWALL_CPU_PREFIX",
      "--steps",
      "20",
      "--output-json",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_solver.json",
      "--probability-npz",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_probability.npz",
      "--device",
      "cpu"
    ]
  ],
  "solver_input_paths": [
    "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
    "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt"
  ],
  "solver_working_directory": "/mnt/d/VRM"
}
```
