import torch.nn as nn


class ChunkEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, latent_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # 16 → 8 → 4 → 2 → 1
        self.convs = nn.Sequential(
            nn.Conv3d(embed_dim, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(256, 512, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(512, 512, 2, stride=2, padding=0),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        # x: (B, 16, 16, 16) int
        x = self.embed(x)                    # (B, 16, 16, 16, embed_dim)
        x = x.permute(0, 4, 1, 2, 3)         # (B, embed_dim, 16, 16, 16)
        x = self.convs(x)                    # (B, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)            # (B, 512)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
