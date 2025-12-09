import torch
from config import (
    VaeTrainConfig,
    VaeSaveConfig,
    VaeParamConfig,
    DataConfig,
)


def train(config_root: str, config_name: str):
    data_cfg = DataConfig.read(config_root, config_name)  # ty:ignore
    param_cfg = VaeParamConfig.read(config_root, config_name)  # ty:ignore
    train_cfg = VaeTrainConfig.read(config_root, config_name)  # ty:ignore
    ckpt_cfg = VaeSaveConfig.read(config_root, config_name)  # ty:ignore

    check_device()


def check_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    torch.device(device)
    print(f'Using device {device}')
