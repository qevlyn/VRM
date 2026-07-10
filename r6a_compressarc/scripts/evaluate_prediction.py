from __future__ import annotations

import argparse
import json
from pathlib import Path

from r6a_common import evaluate_prediction


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CompressARC predictions in evaluator-only process.")
    parser.add_argument("--challenge-json", required=True)
    parser.add_argument("--solution-json", required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    challenges = json.loads(Path(args.challenge_json).read_text(encoding="utf-8"))
    challenge_problem = challenges[args.task_id] if args.task_id in challenges else challenges
    solutions = json.loads(Path(args.solution_json).read_text(encoding="utf-8"))
    solution_grids = solutions[args.task_id] if args.task_id in solutions else solutions
    prediction = json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))
    prediction_grids = [row["attempt_1"] for row in prediction["predictions"]]
    payload = {
        "task_id": args.task_id,
        "condition": prediction.get("condition"),
        "evaluator_process_loaded_solution": True,
        "solver_process_loaded_solution": False,
        "metrics": evaluate_prediction(
            challenge_problem=challenge_problem,
            solution_grids=solution_grids,
            prediction_grids=prediction_grids,
        ),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

