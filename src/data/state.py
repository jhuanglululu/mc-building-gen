import numpy as np
from typing import Callable
from numpy.typing import NDArray
from numpy import int64, uint16

BLOCK_TO_ID: dict[str, int] = dict()
ID_TO_BLOCK: dict[int, str] = dict()


def load_blocks(json_path: str):
    import json
    from pathlib import Path

    blocks = json.loads(Path(json_path).read_text())

    for id, block in enumerate(blocks['blocks']):
        BLOCK_TO_ID[block] = id
        ID_TO_BLOCK[id] = block

    for schema in blocks['schemas'].values():
        _register_parser(
            filters=schema['blocks'],
            fields=[(f['name'], f['bits']) for f in schema['fields']],
        )


BlockIds = NDArray[int64]
RawStates = NDArray[uint16]
Outputs = dict[str, NDArray[int64]]

STATE_PARSERS: list[Callable[[BlockIds, RawStates, Outputs], None]] = list()
ALL_STATE_FIELDS: set[str] = set()


def _register_parser(filters: list[str], fields: list[tuple[str, int]]):
    filter_ids = np.array([BLOCK_TO_ID[filter] for filter in filters], dtype=np.int64)

    ALL_STATE_FIELDS.update(name for name, _ in fields)

    def parser(ids: BlockIds, raw_states: RawStates, outputs: Outputs):
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


def parse_states(ids: BlockIds, raw_states: RawStates) -> Outputs:
    shape = ids.shape
    outputs: Outputs = {name: np.zeros(shape, dtype=int64) for name in ALL_STATE_FIELDS}

    for parser in STATE_PARSERS:
        parser(ids, raw_states, outputs)

    return outputs
