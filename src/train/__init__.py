from typing import no_type_check as _no_type_check


@_no_type_check
def train(config_root: str, config_name: str):
    from config import VaeTrainConfig, VaeSaveConfig, VaeParamConfig, DataConfig
    from .vae import train_vae

    data_cfg: DataConfig = DataConfig.read(config_root, config_name)
    save_cfg: VaeSaveConfig = VaeSaveConfig.read(config_root, config_name)
    param_cfg: VaeParamConfig = VaeParamConfig.read(config_root, config_name)
    train_cfg: VaeTrainConfig = VaeTrainConfig.read(config_root, config_name)

    device = check_device()

    train_vae(device, data_cfg, save_cfg, param_cfg, train_cfg)


def check_device():
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    torch.device(device)
    print(f'Using device {device}')
    return device
