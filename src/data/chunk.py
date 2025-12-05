from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.decode import read_structure


class ChunkDataset(Dataset):
    def __init__(self, data_dir: str, chunk_size: int = 16):
        self.chunk_size = chunk_size
        self.chunks = []  # (path, x0, y0, z0)

        for path in Path(data_dir).glob('*.bin'):
            structure = read_structure(path)
            x, y, z = structure['size']

            for x0 in range(0, x, chunk_size):
                for y0 in range(0, y, chunk_size):
                    for z0 in range(0, z, chunk_size):
                        self.chunks.append((path, x0, y0, z0))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        path, x0, y0, z0 = self.chunks[idx]
        structure = read_structure(path)
        blocks = structure['blocks']

        chunk = self._extract_chunk(blocks, x0, y0, z0)

        block_ids = (chunk >> 16).astype('int32')
        states = (chunk & 0xffff).astype('int32')

        return {
            'block_ids': torch.from_numpy(block_ids),
            'states': torch.from_numpy(states),
        }

    def _extract_chunk(self, blocks, x0, y0, z0):
        cs = self.chunk_size
        x, y, z = blocks.shape

        x_end = min(x0 + cs, x)
        y_end = min(y0 + cs, y)
        z_end = min(z0 + cs, z)

        chunk = blocks[x0:x_end, y0:y_end, z0:z_end]

        if chunk.shape != (cs, cs, cs):
            padded = np.zeros((cs, cs, cs), dtype=np.uint32)
            padded[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
            chunk = padded

        return chunk
