import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
from monai.networks.nets.efficientnet import get_efficientnet_image_size

from datamodule import KvasirSEGDataset
from network_module import Net

L.seed_everything(42, workers=True)

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    logger = loggers.TensorBoardLogger("logs/", name=str(cfg.run_name))

    model = instantiate(cfg.model.object)
    if cfg.img_size == "derived":
        img_size = get_efficientnet_image_size(model.model_name)
    else:
        img_size = cfg.img_size

    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=img_size)
    
    net = Net(
        model=model,
        criterion=instantiate(cfg.criterion),
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
    )

    trainer = instantiate(cfg.trainer, logger=logger)

    # if efficientnetb5, b6, or b7, use binsearch to find the largest batch size
    if cfg.model.object.model_name in ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(net, dataset, mode="binsearch")
    
    trainer.fit(net, dataset)
    trainer.test(net, dataset)


if __name__ == "__main__":
    main()
