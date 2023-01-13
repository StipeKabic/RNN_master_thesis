import json
from typing import Dict
from pathlib import Path

import hydra as hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


ROOT = Path("models/logs")
INSTRUMENT = Path("drums")
CHECKPOINT = Path(f"final_model_{INSTRUMENT}.ckpt")
RESULT= Path("test_results.json")
CONFIG = Path("config")

CHECKPOINT_PATH = ROOT/INSTRUMENT/CHECKPOINT
CONFIG_PATH = ROOT/INSTRUMENT/CONFIG
RESULT_PATH = ROOT/INSTRUMENT/RESULT

if not CHECKPOINT_PATH.parent == CONFIG_PATH.parent == RESULT_PATH.parent:
    raise ValueError("Paths to checkpoint, config and result must be on same level")

CONFIG_NAME = "config.yaml"

@hydra.main(version_base="1.2", config_path=CONFIG_PATH.as_posix(), config_name=CONFIG_NAME)
def test(cfg: DictConfig) -> None:
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=CHECKPOINT_PATH.as_posix())

    metrics: Dict[str, float] = {key: float(value) for key, value in trainer.callback_metrics.items()}
    with open(RESULT_PATH, "w") as f:
        json.dump(metrics, f)


def main():
    test()


if __name__ == "__main__":
    main()
