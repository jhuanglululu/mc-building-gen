from configparser import ConfigParser
from dataclasses import dataclass, fields
import os
from typing import Any, TypeVar, Callable

T = TypeVar('T')


def _config_loader(category: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        def read(root: str, name: str) -> T:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                raise ValueError(f'File "{path}" does not exist')

            parser = ConfigParser()
            parser.read(path)

            kwargs: dict[str, Any] = dict()
            if category not in parser:
                raise ValueError(f'Category {category} not found in {name}')

            for f in fields(cls):  # pyright:ignore[reportArgumentType]
                if f.name not in parser[category]:
                    raise ValueError(f'Field {f.name} not found in {name}/{category}')

                if f.type is int:
                    kwargs[f.name] = parser.getint(category, f.name)
                elif f.type is float:
                    kwargs[f.name] = parser.getfloat(category, f.name)
                elif f.type is bool:
                    kwargs[f.name] = parser.getboolean(category, f.name)
                else:
                    kwargs[f.name] = parser.get(category, f.name)
            return cls(**kwargs)

        setattr(cls, 'read', read)
        return cls

    return decorator


@_config_loader('data')
@dataclass
class DataConfig:
    data_dir: str


@_config_loader('vae')
@dataclass
class VaeSaveConfig:
    save_dir: str


@_config_loader('vae')
@dataclass
class VaeParamConfig:
    chunk_size: int
    d_latent: int
    d_embed: int


@_config_loader('vae_training')
@dataclass
class VaeTrainConfig:
    num_worker: int
    batch_size: int
    learning_rate: float
    epochs: int
    kl_weight: float
