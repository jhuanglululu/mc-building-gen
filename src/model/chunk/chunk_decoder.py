from typing import override
from torch import Tensor
from torch.nn import Sequential, ConvTranspose3d, BatchNorm3d, SiLU, Linear, Module


class ChunkDecoder(Module):
    def __init__(self, d_latent: int, d_embed: int):
        super().__init__()

        self.fc: Sequential = Sequential(
            Linear(d_latent, 512),
            SiLU(),
        )

        self.convs: Sequential = Sequential(
            ConvTranspose3d(512, 512, 2, stride=2, padding=0),
            BatchNorm3d(512),
            SiLU(),
            ConvTranspose3d(512, 256, 4, stride=2, padding=1),
            BatchNorm3d(256),
            SiLU(),
            ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            BatchNorm3d(128),
            SiLU(),
            ConvTranspose3d(128, d_embed, 4, stride=2, padding=1),
        )

    @override
    def forward(self, z: Tensor) -> Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 512, 1, 1, 1)
        x = self.convs(x)
        return x
