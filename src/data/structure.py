from numpy import uint32
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class Structure:
    size: tuple[int, int, int]
    description: str
    blocks: NDArray[uint32]
