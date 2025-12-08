from dataclasses import dataclass
from pathlib import Path
from typing import override
import numpy as np
import torch
from torch.utils.data import Dataset
from data.decode import read_structure
from data.state.helpers import get_state


@dataclass
class BuildingData:
    description: str
    grid_size: tuple[int, int, int]
    block_ids: torch.Tensor
    facing: torch.Tensor
    half: torch.Tensor
    shape: torch.Tensor
    waterlogged: torch.Tensor


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

        shape = blocks.shape
        flat_ids = ids.flatten()
        flat_states = raw_states.flatten()

        facing = np.zeros(len(flat_ids), dtype=np.int64)
        half = np.zeros(len(flat_ids), dtype=np.int64)
        shape_attr = np.zeros(len(flat_ids), dtype=np.int64)
        waterlogged = np.zeros(len(flat_ids), dtype=np.int64)

        for i, (block_id, state) in enumerate(zip(flat_ids, flat_states)):
            parsed = get_state(int(block_id), np.uint16(state))
            facing[i] = parsed.facing.value
            half[i] = parsed.half.value
            shape_attr[i] = parsed.shape.value
            waterlogged[i] = parsed.waterlogged.value

        facing = facing.reshape(shape)
        half = half.reshape(shape)
        shape_attr = shape_attr.reshape(shape)
        waterlogged = waterlogged.reshape(shape)

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
            facing=torch.from_numpy(facing),
            half=torch.from_numpy(half),
            shape=torch.from_numpy(shape_attr),
            waterlogged=torch.from_numpy(waterlogged),
        )
