from __future__ import annotations

from typing import Sequence


def validate_terminated_program(
    program_length: int, program: Sequence[Sequence[int]]
) -> None:
    if program_length not in (0, 1, 2, 3):
        raise ValueError("derived program length must be in 0..3")
    if len(program) != 3 or any(len(item) != 4 for item in program):
        raise ValueError("terminated programs require three four-factor positions")
    for step, instruction in enumerate(program):
        operator, selector, argument_type, value = (int(item) for item in instruction)
        if step >= program_length:
            if (operator, selector, argument_type, value) != (0, 0, 0, 0):
                raise ValueError("NULL termination must force a canonical NULL suffix")
        elif operator == 1:
            if selector not in (1, 2) or argument_type != 1 or value not in (1, 2, 3, 4):
                raise ValueError("invalid TRANSLATE instruction")
        elif operator == 2:
            if selector not in (1, 2) or argument_type != 2 or value not in range(1, 10):
                raise ValueError("invalid RECOLOR instruction")
        elif operator == 3:
            if selector != 3 or argument_type != 3 or value not in (1, 2):
                raise ValueError("invalid REFLECT instruction")
        else:
            raise ValueError("active position cannot contain NULL or an unknown operator")
