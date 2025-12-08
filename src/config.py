from configparser import ConfigParser
from dataclasses import dataclass, fields
import os
from typing import Any, TypeVar, Callable

T = TypeVar("T")


def _config_loader(folder: str, category: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        def read(root: str, name: str) -> T:
            path = os.path.join(root, folder, name)
            if not os.path.exists(path):
                raise ValueError(f"File \"{path}\" does not exist")

            parser = ConfigParser()
            _ = parser.read(path)

            kwargs: dict[str, Any] = dict()  # pyright:ignore[reportExplicitAny]
            if category not in parser:
                raise ValueError(f"Category {category} not found in {folder}/{name}")

            for f in fields(cls):  # pyright:ignore[reportArgumentType]
                if f.name not in parser[category]:
                    raise ValueError(
                        f"Field {f.name} not found in {folder}/{name}/{category}"
                    )

                if f.type is int:
                    kwargs[f.name] = parser.getint(category, f.name)
                elif f.type is float:
                    kwargs[f.name] = parser.getfloat(category, f.name)
                elif f.type is bool:
                    kwargs[f.name] = parser.getboolean(category, f.name)
                else:
                    kwargs[f.name] = parser.get(category, f.name)
            return cls(**kwargs)

        setattr(cls, "read", read)
        return cls

    return decorator


@_config_loader("data", "data")
@dataclass
class DataConfig:
    data_dir: str


@_config_loader("checkpoint", "vae")
@dataclass
class VAECheckpointConfig:
    save_dir: str


@_config_loader("parameter", "vae")
@dataclass
class VAEParameterConfig:
    d_latent: int
    d_embed: int


@_config_loader("training", "vae")
@dataclass
class VAETraningConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    kl_weight: float
