import logging
from typing import no_type_check as _no_type_check

log = logging.getLogger('training')


@_no_type_check
def train(config_root: str, config_name: str, device: str):
    from config import VaeTrainConfig, VaeSaveConfig, VaeParamConfig, DataConfig
    from .vae import train_vae

    data_cfg: DataConfig = DataConfig.read(config_root, config_name)
    save_cfg: VaeSaveConfig = VaeSaveConfig.read(config_root, config_name)
    param_cfg: VaeParamConfig = VaeParamConfig.read(config_root, config_name)
    train_cfg: VaeTrainConfig = VaeTrainConfig.read(config_root, config_name)

    train_vae(device, data_cfg, save_cfg, param_cfg, train_cfg)
