from typing import override
from torch import Tensor
from torch.nn import Sequential, Conv3d, BatchNorm3d, SiLU, Linear, Module


class ChunkEncoder(Module):
    def __init__(self, d_embed: int, d_latent: int = 512):
        super().__init__()

        self.convs: Sequential = Sequential(
            Conv3d(d_embed, 128, 4, stride=2, padding=1),
            BatchNorm3d(128),
            SiLU(),
            Conv3d(128, 256, 4, stride=2, padding=1),
            BatchNorm3d(256),
            SiLU(),
            Conv3d(256, 512, 4, stride=2, padding=1),
            BatchNorm3d(512),
            SiLU(),
            Conv3d(512, 512, 2, stride=2, padding=0),
            BatchNorm3d(512),
            SiLU(),
        )

        self.fc_mu: Linear = Linear(512, d_latent)
        self.fc_logvar: Linear = Linear(512, d_latent)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        return self.fc_mu(x), self.fc_logvar(x)
