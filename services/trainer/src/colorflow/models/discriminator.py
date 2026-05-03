from torch import nn


class Discriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, block_cfg, input_channels, num_filters=64, n_down=3):
        super().__init__()
        self.block_cfg = block_cfg
        layers = [self.get_layers(input_channels, num_filters, norm=False)]
        layers += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                stride=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]
        layers += [
            self.get_layers(
                num_filters * 2**n_down, 1, stride=1, norm=False, activation=False
            )
        ]
        self.model = nn.Sequential(*layers)

    def get_layers(
        self,
        ni,
        nf,
        kernel_size=None,
        stride=None,
        padding=None,
        norm=True,
        activation=True,
    ):
        ks = kernel_size if kernel_size is not None else self.block_cfg.kernel_size
        st = stride if stride is not None else self.block_cfg.stride
        pad = padding if padding is not None else self.block_cfg.padding

        layers = [nn.Conv2d(ni, nf, ks, st, pad, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if activation:
            layers += [nn.LeakyReLU(self.block_cfg.leaky_relu_slope, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
