import os
import signal
import time
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.logging import RichHandler
import logging

from config import DataConfig, VaeSaveConfig, VaeParamConfig, VaeTrainConfig
from data import ChunkDataset
from model import ChunkVae, vae_loss


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger('train_vae')
console = Console()


class GracefulExit:
    def __init__(self):
        self.should_stop: bool = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, _signum: int, _frame: Any):
        if self.should_stop:
            log.warning('Force quit')
            raise SystemExit(1)
        log.warning('Stopping after this epoch... (Ctrl+C again to force quit)')
        self.should_stop = True


def save_model(
    model: ChunkVae,
    save_cfg: VaeSaveConfig,
    param_cfg: VaeParamConfig,
    train_cfg: VaeTrainConfig,
    epoch: int,
):
    import json

    os.makedirs(save_cfg.save_dir, exist_ok=True)
    timestamp = time.strftime('%m-%d-%H-%M-%S', time.localtime())

    model_path = os.path.join(save_cfg.save_dir, f'model-{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    log.info(f'Saved model to {model_path}')

    config_path = os.path.join(save_cfg.save_dir, f'config-{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(
            {
                'epoch': epoch,
                'param': {'d_latent': param_cfg.d_latent, 'd_embed': param_cfg.d_embed},
                'train': {
                    'batch_size': train_cfg.batch_size,
                    'learning_rate': train_cfg.learning_rate,
                    'epochs': train_cfg.epochs,
                    'kl_weight': train_cfg.kl_weight,
                },
            },
            f,
            indent=2,
        )


def train_vae(
    device: str,
    data_cfg: DataConfig,
    save_cfg: VaeSaveConfig,
    param_cfg: VaeParamConfig,
    train_cfg: VaeTrainConfig,
):
    from data import BLOCK_TO_ID

    graceful = GracefulExit()

    dataset = ChunkDataset(data_cfg.data_dir, chunk_size=16)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    log.info(f'Dataset: {len(dataset)} chunks')

    model = ChunkVae(
        vocab_size=len(BLOCK_TO_ID),
        d_latent=param_cfg.d_latent,
        d_embed=param_cfg.d_embed,
    ).to(device)
    log.info(f'Model: {sum(p.numel() for p in model.parameters()):,} parameters')

    optimizer = Adam(model.parameters(), lr=train_cfg.learning_rate)

    model.train()
    for epoch in range(train_cfg.epochs):
        total_loss: float = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f'Epoch {epoch + 1}/{train_cfg.epochs}', total=len(dataloader)
            )

            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits, mu, logvar = model(batch)
                loss = vae_loss(logits, batch, mu, logvar, train_cfg.kl_weight)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress.advance(task)

        avg_loss = total_loss / len(dataloader)
        log.info(f'Epoch {epoch + 1}/{train_cfg.epochs} | Loss: {avg_loss:.4f}')

        if graceful.should_stop:
            log.info('Early stop requested')
            save_model(model, save_cfg, param_cfg, train_cfg, epoch + 1)
            return

    save_model(model, save_cfg, param_cfg, train_cfg, train_cfg.epochs)
