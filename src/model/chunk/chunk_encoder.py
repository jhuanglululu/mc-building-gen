from typing import override
from torch import Tensor
from torch.nn import Sequential, Conv3d, BatchNorm3d, SiLU, Linear, Module


def _dyn_encoder_layers(
    n_layers: int, d_embed: int, d_latent: int
) -> tuple[Sequential, int]:
    channels = [min(d_embed * (2**i), d_latent) for i in range(n_layers)]
    layers: list[Module] = list()

    for i in range(n_layers - 1):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        layers.extend(
            [
                Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
                BatchNorm3d(out_ch),
                SiLU(),
            ]
        )

    layers.extend(
        [
            Conv3d(channels[-1], channels[-1], 2, stride=2, padding=0),
            BatchNorm3d(channels[-1]),
            SiLU(),
        ]
    )

    return Sequential(*layers), channels[-1]


class ChunkEncoder(Module):
    def __init__(self, n_layers: int, d_embed: int, d_latent: int):
        super().__init__()

        self.convs: Sequential
        self.convs, final_chan = _dyn_encoder_layers(n_layers, d_embed, d_latent)

        # hardcoded layers
        # self.convs: Sequential = Sequential(
        #     Conv3d(d_embed, 128, 4, stride=2, padding=1),
        #     BatchNorm3d(128),
        #     SiLU(),
        #     Conv3d(128, 256, 4, stride=2, padding=1),
        #     BatchNorm3d(256),
        #     SiLU(),
        #     Conv3d(256, 512, 4, stride=2, padding=1),
        #     BatchNorm3d(512),
        #     SiLU(),
        #     Conv3d(512, 512, 2, stride=2, padding=0),
        #     BatchNorm3d(512),
        #     SiLU(),
        # )

        self.fc_mu: Linear = Linear(final_chan, d_latent)
        self.fc_logvar: Linear = Linear(final_chan, d_latent)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        return self.fc_mu(x), self.fc_logvar(x)
