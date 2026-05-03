import hydra
from omegaconf import DictConfig

from colorflow.inference import (
    colorize,
    load_generator,
    load_generator_from_checkpoint,
    load_l_channel,
    save_rgb,
)
from colorflow.utils import resolve_device


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.device)
    generator = (
        load_generator_from_checkpoint(cfg.checkpoint_path, device)
        if cfg.use_embedded_config
        else load_generator(cfg, device)
    )
    l_channel = load_l_channel(cfg.input_image, image_size=cfg.data.image_size_1)
    rgb = colorize(generator, l_channel, device)
    save_rgb(rgb, cfg.output_image)
    print(f"Saved colorized image to {cfg.output_image}")


if __name__ == "__main__":
    main()
