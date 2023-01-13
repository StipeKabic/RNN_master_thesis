import hydra as hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    logger =  hydra.utils.instantiate(cfg.logger)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint(f"{cfg.paths.log_dir}{cfg.run_name}/final_model_{cfg.datamodule.source}.ckpt")

def main():
    train()


if __name__ == "__main__":
    main()
