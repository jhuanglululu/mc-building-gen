import torch
import torch.nn as nn

from model.chunk.chunk_decoder import ChunkDecoder
from model.chunk.chunk_encoder import ChunkEncoder


class ChunkVAE(nn.Module):
    def __init__(self, vocab_size: int, d_latent: int, d_block: int):
        super().__init__()
        self.encoder = ChunkEncoder(vocab_size, d_block, d_latent)
        self.decoder = ChunkDecoder(d_latent, d_block)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        embeddings = self.decoder(z)
        return embeddings, mu, logvar

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)
