import torch
import torch.nn as nn

from model.chunk.chunk_decoder import ChunkDecoder
from model.chunk.chunk_encoder import ChunkEncoder


class ChunkVAE(nn.Module):
    def __init__(self, vocab_size: int, d_latent: int, d_embed: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.encoder = ChunkEncoder(d_embed, d_latent)
        self.decoder = ChunkDecoder(d_latent, d_embed)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        embedded = self.embed(x).permute(0, 4, 1, 2, 3)

        mu, logvar = self.encoder(embedded)
        z = self.reparameterize(mu, logvar)

        decoded = self.decoder(z)

        decoded = decoded.permute(0, 2, 3, 4, 1)
        logits = torch.matmul(decoded, self.embed.weight.T)

        return logits, mu, logvar

    def encode(self, x):
        embedded = self.embed(x).permute(0, 4, 1, 2, 3)
        mu, _ = self.encoder(embedded)
        return mu

    def decode(self, z):
        return self.decoder(z)
