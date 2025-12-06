import torch.nn as nn


class ChunkEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_block: int, d_latent: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_block)

        self.convs = nn.Sequential(
            nn.Conv3d(d_block, 128, 4, stride=2, padding=1),
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
        x = self.embed(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
