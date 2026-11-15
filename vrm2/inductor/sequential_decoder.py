from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class SequentialDecoderConfig:
    hidden_dim: int = 128
    instruction_embedding_dim: int = 16
    disable_residual_update: bool = False


def _masked_argmax(
    logits: torch.Tensor, valid_ids: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked[:, list(valid_ids)] = logits[:, list(valid_ids)]
    probabilities = masked.softmax(dim=-1)
    prediction = probabilities.argmax(dim=-1)
    confidence = probabilities.gather(1, prediction.unsqueeze(1)).squeeze(1)
    return prediction, confidence


def materialize_instruction(
    step_logits: Mapping[str, torch.Tensor], already_terminated: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply only frozen-schema masks and return one hard typed instruction."""

    batch = step_logits["operator"].shape[0]
    instruction = torch.zeros(batch, 4, dtype=torch.long, device=already_terminated.device)
    confidence = torch.ones(batch, 4, dtype=torch.float32, device=already_terminated.device)
    operator_probabilities = step_logits["operator"].softmax(dim=-1)
    operator = operator_probabilities.argmax(dim=-1)
    operator = torch.where(already_terminated, torch.zeros_like(operator), operator)
    active = operator.ne(0) & ~already_terminated
    instruction[active, 0] = operator[active]
    confidence[active, 0] = operator_probabilities.gather(1, operator.unsqueeze(1))[active, 0].float()

    for operator_id, selector_ids, argument_type_id, value_ids in (
        (1, (1, 2), 1, (1, 2, 3, 4)),
        (2, (1, 2), 2, (1, 2, 3, 4, 5, 6, 7, 8, 9)),
        (3, (3,), 3, (1, 2)),
    ):
        selected = active & operator.eq(operator_id)
        selectors, selector_confidence = _masked_argmax(step_logits["selector"], selector_ids)
        values, value_confidence = _masked_argmax(step_logits["argument_value"], value_ids)
        instruction[selected, 1] = selectors[selected]
        instruction[selected, 2] = argument_type_id
        instruction[selected, 3] = values[selected]
        confidence[selected, 1] = selector_confidence[selected].float()
        argument_type_probabilities = step_logits["argument_type"].softmax(dim=-1)
        confidence[selected, 2] = argument_type_probabilities[selected, argument_type_id].float()
        confidence[selected, 3] = value_confidence[selected].float()
    null = ~active
    confidence[null, 0] = operator_probabilities[null, 0].float()
    return instruction, confidence


def materialize_program(
    logits: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = logits["operator"].shape[0]
    terminated = torch.zeros(batch, dtype=torch.bool, device=logits["operator"].device)
    instructions = []
    confidences = []
    lengths = torch.full((batch,), 3, dtype=torch.long, device=terminated.device)
    for step in range(3):
        instruction, confidence = materialize_instruction(
            {name: values[:, step] for name, values in logits.items()}, terminated
        )
        newly_terminated = ~terminated & instruction[:, 0].eq(0)
        lengths[newly_terminated] = step
        terminated = terminated | newly_terminated
        instructions.append(instruction)
        confidences.append(confidence)
    return lengths, torch.stack(instructions, dim=1), torch.stack(confidences, dim=1)


class SequentialResidualProgramDecoder(nn.Module):
    """One recurrent decoder, residual update, and field-head set reused three times."""

    def __init__(self, config: SequentialDecoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        embedding = config.instruction_embedding_dim
        self.config = config
        self.residual_initializer = nn.Linear(hidden, hidden)
        self.begin_program_embedding = nn.Parameter(torch.empty(hidden))
        self.operator_embedding = nn.Embedding(4, embedding)
        self.selector_embedding = nn.Embedding(4, embedding)
        self.argument_type_embedding = nn.Embedding(4, embedding)
        self.argument_value_embedding = nn.Embedding(10, embedding)
        self.instruction_projection = nn.Linear(embedding * 4, hidden)
        self.decoder_input_projection = nn.Linear(hidden * 2, hidden)
        self.decoder_cell = nn.GRUCell(hidden, hidden)
        self.residual_update = nn.Linear(hidden * 3, hidden)
        self.residual_norm = nn.LayerNorm(hidden)
        self.operator_head = nn.Linear(hidden, 4)
        self.selector_head = nn.Linear(hidden, 4)
        self.argument_type_head = nn.Linear(hidden, 4)
        self.argument_value_head = nn.Linear(hidden, 10)
        nn.init.normal_(self.begin_program_embedding, std=0.02)

    def embed_typed_instruction(self, instruction: torch.Tensor) -> torch.Tensor:
        factors = (
            self.operator_embedding(instruction[:, 0]),
            self.selector_embedding(instruction[:, 1]),
            self.argument_type_embedding(instruction[:, 2]),
            self.argument_value_embedding(instruction[:, 3]),
        )
        return self.instruction_projection(torch.cat(factors, dim=-1))

    def _step(
        self, task: torch.Tensor, previous: torch.Tensor, residual: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        decoder_input = self.decoder_input_projection(torch.cat((task, previous), dim=-1))
        hidden = self.decoder_cell(decoder_input, residual)
        return {
            "operator": self.operator_head(hidden),
            "selector": self.selector_head(hidden),
            "argument_type": self.argument_type_head(hidden),
            "argument_value": self.argument_value_head(hidden),
        }, hidden

    def _updated_residual(
        self,
        initial: torch.Tensor,
        residual: torch.Tensor,
        instruction_embedding: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.disable_residual_update:
            return initial
        delta = torch.tanh(
            self.residual_update(torch.cat((residual, instruction_embedding, task), dim=-1))
        )
        return self.residual_norm(residual + delta)

    def teacher_forced(
        self, task: torch.Tensor, target_program: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if target_program.shape[1:] != (3, 4):
            raise ValueError("teacher-forced targets must have shape [batch, 3, 4]")
        initial = torch.tanh(self.residual_initializer(task))
        residual = initial
        previous = self.begin_program_embedding.unsqueeze(0).expand(task.shape[0], -1)
        collected: dict[str, list[torch.Tensor]] = {
            "operator": [], "selector": [], "argument_type": [], "argument_value": []
        }
        for step in range(3):
            step_logits, _ = self._step(task, previous, residual)
            for name, values in step_logits.items():
                collected[name].append(values)
            emitted = target_program[:, step]
            previous = self.embed_typed_instruction(emitted)
            residual = self._updated_residual(initial, residual, previous, task)
        return {name: torch.stack(values, dim=1) for name, values in collected.items()}

    def autoregressive(
        self, task: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = torch.tanh(self.residual_initializer(task))
        residual = initial
        previous = self.begin_program_embedding.unsqueeze(0).expand(task.shape[0], -1)
        terminated = torch.zeros(task.shape[0], dtype=torch.bool, device=task.device)
        lengths = torch.full((task.shape[0],), 3, dtype=torch.long, device=task.device)
        programs = []
        confidences = []
        collected: dict[str, list[torch.Tensor]] = {
            "operator": [], "selector": [], "argument_type": [], "argument_value": []
        }
        for step in range(3):
            step_logits, _ = self._step(task, previous, residual)
            for name, values in step_logits.items():
                collected[name].append(values)
            emitted, confidence = materialize_instruction(step_logits, terminated)
            newly_terminated = ~terminated & emitted[:, 0].eq(0)
            lengths[newly_terminated] = step
            terminated = terminated | newly_terminated
            previous = self.embed_typed_instruction(emitted)
            residual = self._updated_residual(initial, residual, previous, task)
            programs.append(emitted)
            confidences.append(confidence)
        return (
            {name: torch.stack(values, dim=1) for name, values in collected.items()},
            lengths,
            torch.stack(programs, dim=1),
            torch.stack(confidences, dim=1),
        )
