from typing import override
from torch import Tensor
from torch.nn import Sequential, ConvTranspose3d, BatchNorm3d, SiLU, Linear, Module


def _dyn_decoder_layers(
    n_layers: int, d_latent: int, d_embed: int
) -> tuple[Sequential, int]:
    channels: list[int] = [min(d_embed * (2**i), d_latent) for i in range(n_layers)]
    channels.reverse()

    layers = [
        ConvTranspose3d(channels[0], channels[0], 2, stride=2, padding=0),
        BatchNorm3d(channels[0]),
        SiLU(),
    ]

    for i in range(n_layers - 2):
        layers.extend(
            [
                ConvTranspose3d(channels[i], channels[i + 1], 4, stride=2, padding=1),
                BatchNorm3d(channels[i + 1]),
                SiLU(),
            ]
        )

    layers.append(ConvTranspose3d(channels[-2], channels[-1], 4, stride=2, padding=1))

    return Sequential(*layers), channels[0]


class ChunkDecoder(Module):
    def __init__(self, n_layers: int, d_latent: int, d_embed: int):
        super().__init__()

        self.convs: Sequential
        self.convs, first_channel = _dyn_decoder_layers(n_layers, d_latent, d_embed)

        # hardcoded layers
        # self.convs: Sequential = Sequential(
        #     ConvTranspose3d(512, 512, 2, stride=2, padding=0),
        #     BatchNorm3d(512),
        #     SiLU(),
        #     ConvTranspose3d(512, 256, 4, stride=2, padding=1),
        #     BatchNorm3d(256),
        #     SiLU(),
        #     ConvTranspose3d(256, 128, 4, stride=2, padding=1),
        #     BatchNorm3d(128),
        #     SiLU(),
        #     ConvTranspose3d(128, d_embed, 4, stride=2, padding=1),
        # )

        # this layer actually comes before self.convs
        # but we need the first channel from _dyn_decoder_layers
        self.fc: Sequential = Sequential(
            Linear(d_latent, first_channel),
            SiLU(),
        )

        self.first_channel: int = first_channel

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), self.first_channel, 1, 1, 1)
        x = self.convs(x)
        return x
