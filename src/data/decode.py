from pathlib import Path
from typing import Tuple

import numpy as np

from data.state import BlockState, get_state


def unpack_block(value: np.uint32) -> Tuple[int, BlockState]:
    block = int(value >> 16)
    state = np.uint16(value & 0xFFFF)
    return block, get_state(block, state)


def read_structure(path: str | Path):
    with open(path, 'rb') as f:
        x = int.from_bytes(f.read(4), 'little')
        y = int.from_bytes(f.read(4), 'little')
        z = int.from_bytes(f.read(4), 'little')

        desc_len = int.from_bytes(f.read(4), 'little')
        description = f.read(desc_len).decode('utf-8')

        block_count = x * y * z
        data = np.frombuffer(f.read(block_count * 4), dtype=np.uint32)
        data = data.reshape((x, y, z))

    return {'size': (x, y, z), 'description': description, 'blocks': data}
