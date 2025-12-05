from typing import Callable, List, Tuple, TypeVar

import numpy as np

from data.state.block_state import BlockState
from data.vocabs import BLOCK_TO_ID


def apply_to(targets: List[str]):
    def decorator(func: Callable[[np.uint16], BlockState]) -> Callable[[np.uint16], BlockState]:
        for target in targets:
            id = BLOCK_TO_ID[target]
            ID_TO_STATE[id] = func
        return func
    return decorator


T = TypeVar('T')

def state_parser(values: List[T], offset: int = 0) -> Tuple[Callable[[np.uint16], T], int]:
    import math
    n = math.ceil(math.log2(len(values)))
    shift = 16 - n - offset
    mask = (1 << n) - 1
    def parser(state: np.uint16) -> T:
        return values[(state >> shift) & mask]
    return parser, offset + n


ID_TO_STATE = dict()


def _default_state(state: np.uint16) -> BlockState:
    return BlockState()

def get_state(block_id: int, state: np.uint16) -> BlockState:
    return ID_TO_STATE.get(block_id, _default_state)(state)
