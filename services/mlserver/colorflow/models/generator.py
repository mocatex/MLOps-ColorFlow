import torch
from torch import nn


class UnetBlock(nn.Module):
    def __init__(
        self,
        nf,
        ni,
        block_cfg,
        submodule=None,
        input_channels=None,
        dropout=False,
        innermost=False,
        outermost=False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_channels is None:
            input_channels = nf

        ks = block_cfg.kernel_size
        st = block_cfg.stride
        pad = block_cfg.padding

        downconv = nn.Conv2d(
            input_channels, ni, kernel_size=ks, stride=st, padding=pad, bias=False
        )
        downrelu = nn.LeakyReLU(block_cfg.leaky_relu_slope, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(
                ni * 2, nf, kernel_size=ks, stride=st, padding=pad
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                ni, nf, kernel_size=ks, stride=st, padding=pad, bias=False
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                ni * 2, nf, kernel_size=ks, stride=st, padding=pad, bias=False
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(block_cfg.dropout)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(
        self,
        block_cfg,
        input_channels=1,
        output_channels=2,
        n_down=8,
        num_filters=64,
    ):
        super().__init__()
        unet_block = UnetBlock(
            num_filters * 8, num_filters * 8, block_cfg, innermost=True
        )
        for _ in range(n_down - 5):
            unet_block = UnetBlock(
                num_filters * 8,
                num_filters * 8,
                block_cfg,
                submodule=unet_block,
                dropout=True,
            )
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(
                out_filters // 2, out_filters, block_cfg, submodule=unet_block
            )
            out_filters //= 2
        self.model = UnetBlock(
            output_channels,
            out_filters,
            block_cfg,
            input_channels=input_channels,
            submodule=unet_block,
            outermost=True,
        )

    def forward(self, x):
        return self.model(x)


def build_backbone_unet(
    device, input_channels=1, output_channels=2, size=256, layers_to_cut=-2
):
    from fastai.vision.learner import create_body
    from fastai.vision.models.unet import DynamicUnet
    from torchvision.models import ResNet18_Weights, resnet18

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    body = create_body(backbone, n_in=input_channels, pretrained=True, cut=layers_to_cut)
    return DynamicUnet(body, output_channels, (size, size)).to(device)