import os
import signal
import time
import warnings
from datetime import datetime, timedelta
from typing import Any

warnings.filterwarnings('ignore', message="'pin_memory' argument is set as true")

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import logging

from config import DataConfig, VaeSaveConfig, VaeParamConfig, VaeTrainConfig
from data import ChunkDataset
from model import ChunkVae, vae_loss


log = logging.getLogger('training')


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
    log.info(
        f'Saved model to [bright_green]{model_path}[/]',
        extra={
            'highlighter': None,
            'markup': True,
        },
    )

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

    dataset = ChunkDataset(data_cfg.data_dir, chunk_size=param_cfg.chunk_size)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_worker,
        pin_memory=True,
    )
    log.info(f'Dataset: {len(dataset)} chunks')

    model = ChunkVae(
        vocab_size=len(BLOCK_TO_ID),
        chunk_size=param_cfg.chunk_size,
        d_latent=param_cfg.d_latent,
        d_embed=param_cfg.d_embed,
    ).to(device)
    log.info(f'Model: {sum(p.numel() for p in model.parameters()):,} parameters')

    optimizer = Adam(model.parameters(), lr=train_cfg.learning_rate)

    def format_duration(seconds: float) -> str:
        return f'{int(seconds // 3600)}h {int(seconds // 60 % 24)}m {seconds % 60:.2f}s'

    model.train()
    train_start = time.time()

    for epoch in range(train_cfg.epochs):
        epoch_start = time.time()
        total_loss: float = 0.0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(batch)
            loss = vae_loss(logits, batch, mu, logvar, train_cfg.kl_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - train_start
        epochs_left = train_cfg.epochs - (epoch + 1)
        avg_epoch_time = total_time / (epoch + 1)
        eta_seconds = avg_epoch_time * epochs_left
        eta_finish = datetime.now() + timedelta(seconds=eta_seconds)

        avg_loss = total_loss / len(dataloader)
        log.info(
            f'Epoch: {f"{epoch + 1}/{train_cfg.epochs}":<12} '
            + f'Loss:  {avg_loss:.4f}\n'
            + f'Time:  {format_duration(epoch_time):<12} '
            + f'Total: {format_duration(total_time)}\n'
            + f'ETA:   {format_duration(eta_seconds):<12} '
            + f'{eta_finish.strftime("%H:%M:%S")}',
        )

        if graceful.should_stop:
            log.info('Early stop requested')
            save_model(model, save_cfg, param_cfg, train_cfg, epoch + 1)
            return

    save_model(model, save_cfg, param_cfg, train_cfg, train_cfg.epochs)
