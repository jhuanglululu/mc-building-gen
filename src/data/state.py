from data.vocabs import BLOCK_TO_ID
import numpy as np
from typing import Callable
from numpy.typing import NDArray
from numpy import int64, uint16

BlockIds = NDArray[int64]
RawStates = NDArray[uint16]
Outputs = dict[str, NDArray[int64]]

STATE_PARSERS: list[Callable[[BlockIds, RawStates, Outputs], None]] = list()
ALL_STATE_FIELDS: set[str] = set()


def _register_parser(filters: list[str], fields: list[tuple[str, int]]) -> None:
    filter_ids = np.array([BLOCK_TO_ID[filter] for filter in filters], dtype=np.int64)

    ALL_STATE_FIELDS.update(name for name, _ in fields)

    def parser(ids: BlockIds, raw_states: RawStates, outputs: Outputs) -> None:
        mask = np.isin(ids, filter_ids)
        if not mask.any():
            return

        s = raw_states[mask]
        offset = 0
        for field in fields:
            shift = 16 - offset - field[1]
            field_mask = (1 << field[1]) - 1
            outputs[field[0]][mask] = ((s >> shift) & field_mask) + 1
            offset += field[1]

    STATE_PARSERS.append(parser)


_register_parser(
    filters=[
        'oak_stairs',
        'birch_stairs',
        'spruce_stairs',
        'stone_stairs',
        'cobblestone_stairs',
        'stone_brick_stairs',
    ],
    fields=[
        ('facing', 2),
        ('half', 1),
        ('shape', 3),
        ('waterlogged', 1),
    ],
)


def parse_states(ids: BlockIds, raw_states: RawStates) -> Outputs:
    shape = ids.shape
    outputs: Outputs = {
        name: np.zeros(shape, dtype=np.int64) for name in ALL_STATE_FIELDS
    }

    for parser in STATE_PARSERS:
        parser(ids, raw_states, outputs)

    return outputs
