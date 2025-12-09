from numpy import uint32
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Structure:
    size: tuple[int, int, int]
    description: str
    blocks: NDArray[uint32]


def read_structure(path: str | Path) -> Structure:
    with open(path, "rb") as f:
        x = int.from_bytes(f.read(4), "little")
        y = int.from_bytes(f.read(4), "little")
        z = int.from_bytes(f.read(4), "little")

        desc_len = int.from_bytes(f.read(4), "little")
        description = f.read(desc_len).decode("utf-8")

        block_count = x * y * z
        data = np.frombuffer(f.read(block_count * 4), dtype=np.uint32)
        data = data.reshape((x, y, z))

    return Structure(size=(x, y, z), description=description, blocks=data)
