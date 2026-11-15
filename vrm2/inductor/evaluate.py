from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.model import SupervisedTypedProgramInductorV0, materialize_hard_program
from vrm2.inductor.schema import validate_typed_program
from vrm2.inductor.types import QueryEpisode, SupportEpisode


def support_tensors(
    episodes: Sequence[SupportEpisode], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [item.support_input_grids for item in episodes], dtype=torch.long, device=device
    )
    outputs = torch.tensor(
        [item.support_output_grids for item in episodes], dtype=torch.long, device=device
    )
    return inputs, outputs


@torch.inference_mode()
def predict_programs(
    model: SupervisedTypedProgramInductorV0,
    episodes: Sequence[SupportEpisode],
    batch_size: int,
    device: torch.device,
    *,
    support_permutation: Sequence[int] | None = None,
    replacement_inputs: torch.Tensor | None = None,
    replacement_outputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], float]:
    model.eval()
    all_lengths = []
    all_programs = []
    length_confidences = []
    field_confidences = []
    start = time.perf_counter()
    for offset in range(0, len(episodes), batch_size):
        batch = episodes[offset : offset + batch_size]
        inputs, outputs = support_tensors(batch, device)
        if replacement_inputs is not None:
            inputs = replacement_inputs[offset : offset + len(batch)].to(device)
        if replacement_outputs is not None:
            outputs = replacement_outputs[offset : offset + len(batch)].to(device)
        if support_permutation is not None:
            inputs = inputs[:, support_permutation]
            outputs = outputs[:, support_permutation]
        logits = model(inputs, outputs)
        lengths, programs, confidence = materialize_hard_program(logits)
        all_lengths.append(lengths.cpu())
        all_programs.append(programs.cpu())
        length_confidences.append(confidence["length"].cpu())
        field_confidences.append(confidence["fields"].cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        torch.cat(all_lengths),
        torch.cat(all_programs),
        {
            "length": torch.cat(length_confidences),
            "fields": torch.cat(field_confidences),
        },
        elapsed,
    )


@torch.inference_mode()
def execute_programs(
    interpreter: torch.nn.Module,
    input_grids: torch.Tensor,
    lengths: torch.Tensor,
    programs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    interpreter.eval()
    predictions = torch.empty_like(input_grids)
    start = time.perf_counter()
    for length in (2, 3):
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


def target_tensors(
    supports: Sequence[SupportEpisode], queries: Sequence[QueryEpisode]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if [item.episode_id for item in supports] != [item.episode_id for item in queries]:
        raise ValueError("support/query episode IDs are not aligned")
    lengths = torch.tensor([item.target_program_length for item in supports], dtype=torch.long)
    programs = torch.tensor([item.target_typed_program_ids for item in supports], dtype=torch.long)
    query_inputs = torch.tensor([item.query_input_grid for item in queries], dtype=torch.long)
    query_targets = torch.tensor([item.query_target_grid for item in queries], dtype=torch.long)
    return lengths, programs, query_inputs, query_targets


def _program_metrics(
    predicted_lengths: torch.Tensor,
    predicted_programs: torch.Tensor,
    target_lengths: torch.Tensor,
    target_programs: torch.Tensor,
) -> dict[str, Any]:
    length_exact = predicted_lengths.eq(target_lengths)
    active = torch.arange(3).unsqueeze(0) < target_lengths.unsqueeze(1)
    instruction_exact = predicted_programs.eq(target_programs).all(dim=2)
    program_exact = length_exact & (instruction_exact | ~active).all(dim=1)
    field_names = ("operator", "selector", "argument_type", "argument_value")
    fields = {}
    for index, name in enumerate(field_names):
        fields[name] = float(
            predicted_programs[:, :, index].eq(target_programs[:, :, index])[active].float().mean().item()
        )
    return {
        "program_exact": float(program_exact.float().mean().item()),
        "length_exact": float(length_exact.float().mean().item()),
        "instruction_exact_by_step": {
            str(step + 1): float(instruction_exact[:, step][active[:, step]].float().mean().item())
            for step in range(3)
            if active[:, step].any()
        },
        "field_accuracy": fields,
        "program_exact_mask": program_exact,
    }


@torch.inference_mode()
def evaluate_model_on_episodes(
    model: SupervisedTypedProgramInductorV0,
    interpreter: torch.nn.Module,
    supports: Sequence[SupportEpisode],
    queries: Sequence[QueryEpisode],
    batch_size: int,
    device: torch.device,
    *,
    include_loo: bool = True,
) -> dict[str, Any]:
    predicted_lengths, predicted_programs, _, induction_seconds = predict_programs(
        model, supports, batch_size, device
    )
    target_lengths, target_programs, query_inputs, query_targets = target_tensors(supports, queries)
    metrics = _program_metrics(
        predicted_lengths, predicted_programs, target_lengths, target_programs
    )
    query_predictions, execution_seconds = execute_programs(
        interpreter, query_inputs, predicted_lengths, predicted_programs, batch_size, device
    )
    query_exact_mask = query_predictions.eq(query_targets).flatten(1).all(dim=1)
    oracle_predictions, _ = execute_programs(
        interpreter, query_inputs, target_lengths, target_programs, batch_size, device
    )
    oracle_exact_mask = oracle_predictions.eq(query_targets).flatten(1).all(dim=1)

    support_inputs, support_outputs = support_tensors(supports, torch.device("cpu"))
    repeated_lengths = predicted_lengths.repeat_interleave(3)
    repeated_programs = predicted_programs.repeat_interleave(3, dim=0)
    support_predictions, _ = execute_programs(
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
        key: value for key, value in metrics.items() if key != "program_exact_mask"
    }
    result.update(
        {
            "num_episodes": len(supports),
            "behavioral_program_exact": float(behavioral.float().mean().item()),
            "query_exact": float(query_exact_mask.float().mean().item()),
            "oracle_query_exact": float(oracle_exact_mask.float().mean().item()),
            "induction_gap": float(
                oracle_exact_mask.float().mean().item() - query_exact_mask.float().mean().item()
            ),
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
            inputs = support_inputs[:, keep].to(device)
            outputs = support_outputs[:, keep].to(device)
            logits = model(inputs, outputs)
            lengths, programs, _ = materialize_hard_program(logits)
            fold_metrics = _program_metrics(
                lengths.cpu(), programs.cpu(), target_lengths, target_programs
            )
            loo_program_correct += int(fold_metrics["program_exact_mask"].sum().item())
            held_predictions, _ = execute_programs(
                interpreter,
                support_inputs[:, omitted],
                lengths.cpu(),
                programs.cpu(),
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


def write_predicted_programs(
    path: Path,
    episodes: Sequence[SupportEpisode],
    lengths: torch.Tensor,
    programs: torch.Tensor,
    confidence: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    lines = []
    for index, episode in enumerate(episodes):
        program = programs[index].tolist()
        validate_typed_program(int(lengths[index].item()), program)
        record = {
            "episode_id": episode.episode_id,
            "predicted_program_length": int(lengths[index].item()),
            "predicted_typed_program_ids": program,
            "per_field_confidence": {
                "length": float(confidence["length"][index].item()),
                "fields": confidence["fields"][index].tolist(),
            },
            "inductor_checkpoint_hash": checkpoint_hash,
        }
        lines.append(canonical_json_bytes(record))
    payload = b"\n".join(lines) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "num_programs": len(lines),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
    }
