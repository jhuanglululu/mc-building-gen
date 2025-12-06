import torch.nn as nn


class ChunkDecoder(nn.Module):
    def __init__(self, d_latent: int, d_block: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_latent, 512),
            nn.SiLU(),
        )

        self.convs = nn.Sequential(
            nn.ConvTranspose3d(512, 512, 2, stride=2, padding=0),
            nn.BatchNorm3d(512),
            nn.SiLU(),
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.SiLU(),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            nn.ConvTranspose3d(128, d_block, 4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 1, 1, 1)
        x = self.convs(x)
        return x
