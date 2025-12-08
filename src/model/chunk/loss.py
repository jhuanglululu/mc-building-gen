import torch
import torch.nn.functional as F


def vae_loss(logits, target, mu, logvar, kl_weight=0.001):
    logits = logits.reshape(-1, logits.size(-1))
    target = target.reshape(-1)
    recon_loss = F.cross_entropy(logits, target)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_weight * kl_loss
