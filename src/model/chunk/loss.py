import torch
import torch.nn.functional as F
from torch import Tensor


def vae_loss(
    logits: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    kl_weight: float = 0.001,
) -> Tensor:
    logits = logits.reshape(-1, logits.size(-1))
    target = target.reshape(-1)
    recon_loss = F.cross_entropy(logits, target)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_weight * kl_loss
