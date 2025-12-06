import torch
import torch.nn as nn

from model.chunk.chunk_decoder import ChunkDecoder
from model.chunk.chunk_encoder import ChunkEncoder


class ChunkVAE(nn.Module):
    def __init__(self, vocab_size: int, d_chunk: int, d_block: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_block)
        self.encoder = ChunkEncoder(d_block, d_chunk)
        self.decoder = ChunkDecoder(d_chunk, d_block)

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
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)
