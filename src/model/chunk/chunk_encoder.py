import torch.nn as nn


class ChunkEncoder(nn.Module):
    def __init__(self, d_embed: int, d_latent: int = 512):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(d_embed, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.SiLU(),
            nn.Conv3d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.SiLU(),
            nn.Conv3d(512, 512, 2, stride=2, padding=0),
            nn.BatchNorm3d(512),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(512, d_latent)
        self.fc_logvar = nn.Linear(512, d_latent)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        return self.fc_mu(x), self.fc_logvar(x)
