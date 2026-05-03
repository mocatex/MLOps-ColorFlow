import torch
from torch import nn, optim

from colorflow.models.discriminator import Discriminator
from colorflow.models.generator import Unet
from colorflow.models.losses import GANLoss
from colorflow.utils import init_model


class MainModel(nn.Module):
    """Combined generator + discriminator with the pix2pix training step."""

    def __init__(self, model_cfg, training_cfg, device, generator=None):
        super().__init__()
        self.device = device
        self.lambda_l1 = model_cfg.loss.lambda_l1

        if generator is None:
            self.generator = init_model(
                Unet(
                    block_cfg=model_cfg.block,
                    input_channels=model_cfg.generator.input_channels,
                    output_channels=model_cfg.generator.output_channels,
                    n_down=model_cfg.generator.n_down,
                    num_filters=model_cfg.generator.num_filters,
                ),
                self.device,
            )
        else:
            self.generator = generator.to(self.device)

        self.discriminator = init_model(
            Discriminator(
                block_cfg=model_cfg.block,
                input_channels=model_cfg.discriminator.input_channels,
                num_filters=model_cfg.discriminator.num_filters,
                n_down=model_cfg.discriminator.n_down,
            ),
            self.device,
        )
        self.GANloss = GANLoss(gan_mode=model_cfg.loss.gan_mode).to(self.device)
        self.L1loss = nn.L1Loss()

        gan = training_cfg.gan
        self.gen_optim = optim.AdamW(
            self.generator.parameters(),
            lr=gan.gen_lr,
            betas=(gan.beta1, gan.beta2),
            weight_decay=gan.weight_decay,
        )
        self.disc_optim = optim.AdamW(
            self.discriminator.parameters(),
            lr=gan.disc_lr,
            betas=(gan.beta1, gan.beta2),
            weight_decay=gan.weight_decay,
        )

    @staticmethod
    def requires_grad(model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def prepare_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.gen_output = self.generator(self.L)

    def disc_backward(self):
        gen_image = torch.cat([self.L, self.gen_output], dim=1)
        gen_image_preds = self.discriminator(gen_image.detach())
        self.disc_loss_gen = self.GANloss(gen_image_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.discriminator(real_image)
        self.disc_loss_real = self.GANloss(real_preds, True)
        self.disc_loss = (self.disc_loss_gen + self.disc_loss_real) * 0.5
        self.disc_loss.backward()

    def gen_backward(self):
        gen_image = torch.cat([self.L, self.gen_output], dim=1)
        gen_image_preds = self.discriminator(gen_image)
        self.loss_G_GAN = self.GANloss(gen_image_preds, True)
        self.loss_G_L1 = self.L1loss(self.gen_output, self.ab) * self.lambda_l1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.discriminator.train()
        self.requires_grad(self.discriminator, True)
        self.disc_optim.zero_grad()
        self.disc_backward()
        self.disc_optim.step()

        self.generator.train()
        self.requires_grad(self.discriminator, False)
        self.gen_optim.zero_grad()
        self.gen_backward()
        self.gen_optim.step()
