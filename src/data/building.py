from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.decode import read_structure


class BuildingDataset(Dataset):
    def __init__(self, data_dir: str, chunk_size: int = 16):
        self.chunk_size = chunk_size
        self.buildings = list(Path(data_dir).glob("*.bin"))

    def __len__(self):
        return len(self.buildings)

    def __getitem__(self, idx):
        structure = read_structure(self.buildings[idx])
        blocks = structure["blocks"]

        block_ids = (blocks >> 16).astype("int32")
        states = (blocks & 0xffff).astype("int32")

        # Grid dimensions in chunks
        x, y, z = structure["size"]
        grid_size = (
            x // self.chunk_size,
            y // self.chunk_size,
            z // self.chunk_size,
        )

        return {
            "block_ids": torch.from_numpy(block_ids),
            "states": torch.from_numpy(states),
            "description": structure["description"],
            "grid_size": grid_size,
        }
