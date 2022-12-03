import hydra as hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


def train(cfg: DictConfig) -> None:
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)

@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()
