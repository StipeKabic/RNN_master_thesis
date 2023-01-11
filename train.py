import hydra as hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


def train(cfg: DictConfig) -> None:
    datamodule:pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    logger =  hydra.utils.instantiate(cfg.logger)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint(cfg.paths.data_dir + "/final_model.ckpt")


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
