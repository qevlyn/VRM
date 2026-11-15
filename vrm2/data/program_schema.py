from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

TYPED_PROGRAM_SCHEMA_VERSION = "vrm2-typed-program-v1"


class OperatorId(IntEnum):
    NULL = 0
    TRANSLATE = 1
    RECOLOR = 2
    REFLECT = 3


class SelectorId(IntEnum):
    NULL = 0
    LEFTMOST_OBJECT = 1
    TOPMOST_OBJECT = 2
    ALL_CONTENT = 3


class ArgumentTypeId(IntEnum):
    NULL = 0
    CARDINAL_DIRECTION = 1
    TARGET_COLOR = 2
    AXIS = 3


class DirectionValue(IntEnum):
    NULL = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class AxisValue(IntEnum):
    NULL = 0
    ROW = 1
    COLUMN = 2


@dataclass(frozen=True)
class ProgramInstruction:
    operator_id: int
    selector_id: int
    argument_type: int
    argument_value: int

    def as_ids(self) -> tuple[int, int, int, int]:
        return (self.operator_id, self.selector_id, self.argument_type, self.argument_value)


_DIRECTION_IDS = {
    "up": DirectionValue.UP,
    "down": DirectionValue.DOWN,
    "left": DirectionValue.LEFT,
    "right": DirectionValue.RIGHT,
}


def factorize_instruction(name: str, parameters: Mapping[str, Any]) -> ProgramInstruction:
    if name == "translate_leftmost":
        return _translate(SelectorId.LEFTMOST_OBJECT, parameters)
    if name == "translate_topmost":
        return _translate(SelectorId.TOPMOST_OBJECT, parameters)
    if name == "recolor_leftmost":
        return _recolor(SelectorId.LEFTMOST_OBJECT, parameters)
    if name == "recolor_topmost":
        return _recolor(SelectorId.TOPMOST_OBJECT, parameters)
    if name == "reflect_rows":
        return ProgramInstruction(OperatorId.REFLECT, SelectorId.ALL_CONTENT, ArgumentTypeId.AXIS, AxisValue.ROW)
    if name == "reflect_columns":
        return ProgramInstruction(
            OperatorId.REFLECT,
            SelectorId.ALL_CONTENT,
            ArgumentTypeId.AXIS,
            AxisValue.COLUMN,
        )
    raise ValueError(f"unknown generator operation: {name}")


def _translate(selector: SelectorId, parameters: Mapping[str, Any]) -> ProgramInstruction:
    direction = parameters.get("direction")
    if direction not in _DIRECTION_IDS:
        raise ValueError(f"invalid translation direction: {direction!r}")
    return ProgramInstruction(
        OperatorId.TRANSLATE,
        selector,
        ArgumentTypeId.CARDINAL_DIRECTION,
        _DIRECTION_IDS[str(direction)],
    )


def _recolor(selector: SelectorId, parameters: Mapping[str, Any]) -> ProgramInstruction:
    color = parameters.get("color")
    if not isinstance(color, int) or not 1 <= color <= 9:
        raise ValueError(f"invalid target color: {color!r}")
    return ProgramInstruction(OperatorId.RECOLOR, selector, ArgumentTypeId.TARGET_COLOR, color)
