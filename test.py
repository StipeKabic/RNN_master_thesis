import json
from typing import Dict
from pathlib import Path

import hydra as hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


ROOT_BASE: Path = Path("models/logs")
INSTRUMENT: str = "drums"
ROOT: Path = ROOT_BASE/INSTRUMENT

CHECKPOINT_PATH: str = (ROOT/f"final_model_{INSTRUMENT}.ckpt").as_posix()
CONFIG_PATH: str = (ROOT/"config").as_posix()
RESULT_PATH: Path = ROOT/"test_results.json"


@hydra.main(version_base="1.2", config_path=CONFIG_PATH, config_name="config.yaml")
def test(cfg: DictConfig) -> None:
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=CHECKPOINT_PATH)

    metrics: Dict[str, float] = {key: float(value) for key, value in trainer.callback_metrics.items()}
    with RESULT_PATH.open("w") as f:
        json.dump(metrics, f)


def main():
    test()


if __name__ == "__main__":
    main()
