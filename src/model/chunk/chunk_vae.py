import torch
import torch.nn as nn

from model.chunk.chunk_decoder import ChunkDecoder
from model.chunk.chunk_encoder import ChunkEncoder


class ChunkVAE(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, latent_dim: int = 512):
        super().__init__()
        self.encoder = ChunkEncoder(vocab_size, embed_dim, latent_dim)
        self.decoder = ChunkDecoder(vocab_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu  # For inference, just use mean

    def decode(self, z):
        return self.decoder(z)
