from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.model_v1 import SequentialResidualInductorV1
from vrm2.inductor.schema_v1 import validate_terminated_program
from vrm2.inductor.sequential_decoder import materialize_program
from vrm2.inductor.types import QueryEpisode, SupportEpisode


def support_tensors_v1(
    episodes: Sequence[SupportEpisode], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [item.support_input_grids for item in episodes], dtype=torch.long, device=device
        ),
        torch.tensor(
            [item.support_output_grids for item in episodes], dtype=torch.long, device=device
        ),
    )


def target_tensors_v1(
    supports: Sequence[SupportEpisode], queries: Sequence[QueryEpisode]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if [item.episode_id for item in supports] != [item.episode_id for item in queries]:
        raise ValueError("V1 support/query IDs are not aligned")
    return (
        torch.tensor([item.target_program_length for item in supports], dtype=torch.long),
        torch.tensor([item.target_typed_program_ids for item in supports], dtype=torch.long),
        torch.tensor([item.query_input_grid for item in queries], dtype=torch.long),
        torch.tensor([item.query_target_grid for item in queries], dtype=torch.long),
    )


@torch.inference_mode()
def predict_programs_v1(
    model: SequentialResidualInductorV1,
    episodes: Sequence[SupportEpisode],
    batch_size: int,
    device: torch.device,
    *,
    teacher_forced: bool = False,
    support_permutation: Sequence[int] | None = None,
    replacement_inputs: torch.Tensor | None = None,
    replacement_outputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    model.eval()
    all_lengths = []
    all_programs = []
    all_confidences = []
    start = time.perf_counter()
    for offset in range(0, len(episodes), batch_size):
        batch = episodes[offset : offset + batch_size]
        inputs, outputs = support_tensors_v1(batch, device)
        if replacement_inputs is not None:
            inputs = replacement_inputs[offset : offset + len(batch)].to(device)
        if replacement_outputs is not None:
            outputs = replacement_outputs[offset : offset + len(batch)].to(device)
        if support_permutation is not None:
            inputs = inputs[:, support_permutation]
            outputs = outputs[:, support_permutation]
        if teacher_forced:
            targets = torch.tensor(
                [item.target_typed_program_ids for item in batch],
                dtype=torch.long,
                device=device,
            )
            logits = model(inputs, outputs, targets)
            lengths, programs, confidences = materialize_program(logits)
        else:
            _, lengths, programs, confidences = model.forward_autoregressive(inputs, outputs)
        all_lengths.append(lengths.cpu())
        all_programs.append(programs.cpu())
        all_confidences.append(confidences.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        torch.cat(all_lengths),
        torch.cat(all_programs),
        torch.cat(all_confidences),
        time.perf_counter() - start,
    )


@torch.inference_mode()
def execute_programs_v1(
    interpreter: torch.nn.Module,
    input_grids: torch.Tensor,
    lengths: torch.Tensor,
    programs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    if programs.dtype != torch.long or lengths.dtype != torch.long:
        raise TypeError("V1 execution boundary requires hard torch.long IDs")
    if not lengths.ge(0).logical_and(lengths.le(3)).all():
        raise ValueError("V1 derived lengths must be in 0..3")
    interpreter.eval()
    predictions = torch.empty_like(input_grids)
    start = time.perf_counter()
    zero = lengths.eq(0).nonzero(as_tuple=False).flatten()
    if len(zero):
        predictions[zero] = input_grids[zero]
    for length in (1, 2, 3):
        indices = lengths.eq(length).nonzero(as_tuple=False).flatten()
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            grids = input_grids[selected].to(device)
            typed = programs[selected, :length].to(device=device, dtype=torch.long)
            logits = interpreter(grids, typed)
            predictions[selected] = logits[:, -1].argmax(dim=1).cpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return predictions, time.perf_counter() - start


def program_metrics_v1(
    predicted_lengths: torch.Tensor,
    predicted_programs: torch.Tensor,
    target_lengths: torch.Tensor,
    target_programs: torch.Tensor,
) -> dict[str, Any]:
    instruction_exact = predicted_programs.eq(target_programs).all(dim=2)
    program_mask = predicted_lengths.eq(target_lengths) & instruction_exact.all(dim=1)
    predicted_null = predicted_programs[:, :, 0].eq(0)
    target_null = target_programs[:, :, 0].eq(0)
    termination = predicted_null.eq(target_null).all(dim=1)
    fields = {}
    for index, name in enumerate(("operator", "selector", "argument_type", "argument_value")):
        fields[name] = float(
            predicted_programs[:, :, index].eq(target_programs[:, :, index]).float().mean().item()
        )
    return {
        "program_exact": float(program_mask.float().mean().item()),
        "program_exact_mask": program_mask,
        "derived_length_exact": float(predicted_lengths.eq(target_lengths).float().mean().item()),
        "null_termination_exact": float(termination.float().mean().item()),
        "instruction_exact_by_step": {
            str(step + 1): float(instruction_exact[:, step].float().mean().item())
            for step in range(3)
        },
        "field_accuracy": fields,
    }


@torch.inference_mode()
def evaluate_model_v1(
    model: SequentialResidualInductorV1,
    interpreter: torch.nn.Module,
    supports: Sequence[SupportEpisode],
    queries: Sequence[QueryEpisode],
    batch_size: int,
    device: torch.device,
    *,
    include_loo: bool = True,
) -> dict[str, Any]:
    lengths, programs, _, induction_seconds = predict_programs_v1(
        model, supports, batch_size, device
    )
    tf_lengths, tf_programs, _, _ = predict_programs_v1(
        model, supports, batch_size, device, teacher_forced=True
    )
    target_lengths, target_programs, query_inputs, query_targets = target_tensors_v1(
        supports, queries
    )
    autoregressive = program_metrics_v1(lengths, programs, target_lengths, target_programs)
    teacher = program_metrics_v1(
        tf_lengths, tf_programs, target_lengths, target_programs
    )
    query_predictions, execution_seconds = execute_programs_v1(
        interpreter, query_inputs, lengths, programs, batch_size, device
    )
    oracle_predictions, _ = execute_programs_v1(
        interpreter, query_inputs, target_lengths, target_programs, batch_size, device
    )
    query_exact = query_predictions.eq(query_targets).flatten(1).all(dim=1)
    oracle_exact = oracle_predictions.eq(query_targets).flatten(1).all(dim=1)

    support_inputs, support_outputs = support_tensors_v1(supports, torch.device("cpu"))
    repeated_lengths = lengths.repeat_interleave(3)
    repeated_programs = programs.repeat_interleave(3, dim=0)
    support_predictions, _ = execute_programs_v1(
        interpreter,
        support_inputs.reshape(-1, 8, 8),
        repeated_lengths,
        repeated_programs,
        batch_size,
        device,
    )
    behavioral = support_predictions.eq(support_outputs.reshape(-1, 8, 8)).flatten(1).all(dim=1)
    behavioral = behavioral.reshape(len(supports), 3).all(dim=1)

    result = {
        key: value for key, value in autoregressive.items() if key != "program_exact_mask"
    }
    result.update(
        {
            "num_episodes": len(supports),
            "autoregressive_program_exact": autoregressive["program_exact"],
            "teacher_forced_program_exact": teacher["program_exact"],
            "teacher_forcing_gap": float(
                teacher["program_exact"] - autoregressive["program_exact"]
            ),
            "teacher_forced_derived_length_exact": teacher["derived_length_exact"],
            "teacher_forced_null_termination_exact": teacher["null_termination_exact"],
            "behavioral_program_exact": float(behavioral.float().mean().item()),
            "query_exact": float(query_exact.float().mean().item()),
            "oracle_query_exact": float(oracle_exact.float().mean().item()),
            "induction_gap": float(oracle_exact.float().mean().item() - query_exact.float().mean().item()),
            "induction_seconds": induction_seconds,
            "induction_ms_per_episode": induction_seconds * 1000.0 / len(supports),
            "execution_seconds": execution_seconds,
            "execution_ms_per_grid": execution_seconds * 1000.0 / len(supports),
        }
    )
    if include_loo:
        loo_program_correct = 0
        loo_execution_correct = 0
        loo_total = 0
        for omitted in range(3):
            keep = [index for index in range(3) if index != omitted]
            fold_supports = []
            for support in supports:
                fold_supports.append(
                    SupportEpisode(
                        episode_id=support.episode_id,
                        support_input_grids=tuple(support.support_input_grids[index] for index in keep),  # type: ignore[arg-type]
                        support_output_grids=tuple(support.support_output_grids[index] for index in keep),  # type: ignore[arg-type]
                        target_program_length=support.target_program_length,
                        target_typed_program_ids=support.target_typed_program_ids,
                    )
                )
            fold_lengths, fold_programs, _, _ = predict_programs_v1(
                model, fold_supports, batch_size, device
            )
            fold_metrics = program_metrics_v1(
                fold_lengths, fold_programs, target_lengths, target_programs
            )
            loo_program_correct += int(fold_metrics["program_exact_mask"].sum().item())
            held_predictions, _ = execute_programs_v1(
                interpreter,
                support_inputs[:, omitted],
                fold_lengths,
                fold_programs,
                batch_size,
                device,
            )
            loo_execution_correct += int(
                held_predictions.eq(support_outputs[:, omitted]).flatten(1).all(dim=1).sum().item()
            )
            loo_total += len(supports)
        result["loo_program_exact"] = loo_program_correct / loo_total
        result["loo_execution_exact"] = loo_execution_correct / loo_total
    return result


def write_predicted_programs_v1(
    path: Path,
    episodes: Sequence[SupportEpisode],
    lengths: torch.Tensor,
    programs: torch.Tensor,
    confidence: torch.Tensor,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    lines = []
    for index, episode in enumerate(episodes):
        length = int(lengths[index].item())
        program = programs[index].tolist()
        validate_terminated_program(length, program)
        lines.append(
            canonical_json_bytes(
                {
                    "episode_id": episode.episode_id,
                    "predicted_program_length": length,
                    "predicted_typed_program_ids": program,
                    "per_field_confidence": confidence[index].tolist(),
                    "inductor_checkpoint_hash": checkpoint_hash,
                }
            )
        )
    payload = b"\n".join(lines) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "num_programs": len(lines),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
    }
