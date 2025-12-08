from typing import Callable, TypeVar

from numpy import uint16

from data.state.block_state import BlockState
from data.vocabs import BLOCK_TO_ID


def apply_to(targets: list[str]):
    def decorator(
        func: Callable[[uint16], BlockState],
    ) -> Callable[[uint16], BlockState]:
        for target in targets:
            id = BLOCK_TO_ID[target]
            ID_TO_STATE[id] = func
        return func

    return decorator


T = TypeVar('T')


def state_parser(values: list[T], offset: int = 0) -> tuple[Callable[[uint16], T], int]:
    import math

    n = math.ceil(math.log2(len(values)))
    shift = 16 - n - offset
    mask = (1 << n) - 1

    def parser(state: uint16) -> T:
        return values[(state >> shift) & mask]

    return parser, offset + n


ID_TO_STATE: dict[int, Callable[[uint16], BlockState]] = dict()


def _default_state(_: uint16) -> BlockState:
    return BlockState()


def get_state(block_id: int, state: uint16) -> BlockState:
    return ID_TO_STATE.get(block_id, _default_state)(state)
