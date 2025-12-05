import torch.nn as nn
import torch.nn.functional as F


class ChunkDecoder(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int = 512):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512)

        # 1 → 2 → 4 → 8 → 16
        self.convs = nn.Sequential(
            nn.ConvTranspose3d(512, 512, 2, stride=2, padding=0),  # 1→2
            nn.SiLU(),
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),  # 2→4
            nn.SiLU(),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 4→8
            nn.SiLU(),
            nn.ConvTranspose3d(128, vocab_size, 4, stride=2, padding=1),  # 8→16
        )

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z)                       # (B, 512)
        x = x.view(x.size(0), 512, 1, 1, 1)  # (B, 512, 1, 1, 1)
        x = self.convs(x)                    # (B, vocab_size, 16, 16, 16)
        return x
