from typing import override
import torch
from torch import Tensor
from torch.nn import Module, Embedding

from model.chunk.chunk_decoder import ChunkDecoder
from model.chunk.chunk_encoder import ChunkEncoder


class ChunkVae(Module):
    def __init__(self, vocab_size: int, chunk_size: int, d_latent: int, d_embed: int):
        from math import log2

        super().__init__()

        n_layers = int(log2(chunk_size))

        self.embed: Embedding = Embedding(vocab_size, d_embed)
        self.encoder: ChunkEncoder = ChunkEncoder(n_layers, d_embed, d_latent)
        self.decoder: ChunkDecoder = ChunkDecoder(n_layers, d_latent, d_embed)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embedded = self.embed(x).permute(0, 4, 1, 2, 3)

        mu, logvar = self.encoder(embedded)
        z = self.reparameterize(mu, logvar)

        decoded = self.decoder(z)

        decoded = decoded.permute(0, 2, 3, 4, 1)
        logits = torch.matmul(decoded, self.embed.weight.T)

        return logits, mu, logvar

    def encode(self, x: Tensor) -> Tensor:
        embedded = self.embed(x).permute(0, 4, 1, 2, 3)
        mu, _ = self.encoder(embedded)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
