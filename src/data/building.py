from data.structure import read_structure
from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
from torch.utils.data import Dataset

from data.state import parse_states


@dataclass
class BuildingData:
    description: str
    grid_size: tuple[int, int, int]
    block_ids: torch.Tensor
    states: dict[str, torch.Tensor]


class BuildingDataset(Dataset[BuildingData]):
    def __init__(self, data_dir: str):
        self.chunk_size: int = 16
        self.buildings: list[Path] = list(Path(data_dir).glob("*.bin"))

    def __len__(self):
        return len(self.buildings)

    @override
    def __getitem__(self, index: int) -> BuildingData:
        structure = read_structure(self.buildings[index])
        blocks = structure.blocks

        ids = (blocks >> 16).astype("int64")
        raw_states = (blocks & 0xFFFF).astype("uint16")

        states = parse_states(ids, raw_states)

        x, y, z = structure.size
        grid_size = (
            x // self.chunk_size,
            y // self.chunk_size,
            z // self.chunk_size,
        )

        return BuildingData(
            description=structure.description,
            grid_size=grid_size,
            block_ids=torch.from_numpy(ids),
            states={k: torch.from_numpy(v) for k, v in states.items()},
        )
