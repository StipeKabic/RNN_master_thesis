from src.data.time_series_dataset.ts_dataset import TSDataModule
from src.models.architectures.SimpleRNN import SimpleRNN
from src.models.LightningRNN import LightningRNN

import pytorch_lightning as pl
from torch import nn

N_STEPS = 50
BATCH_SIZE = 8


def train():
    datamodule: pl.LightningDataModule = TSDataModule(n_steps=N_STEPS, batch_size=BATCH_SIZE)
    network: nn.Module = SimpleRNN(
        input_size=1,
        output_size=20,
        sequence_length=N_STEPS,
        batch_size=BATCH_SIZE
    )
    loss: nn.Module = nn.MSELoss()
    model: pl.LightningModule = LightningRNN(network=network, loss=loss)

    trainer: pl.Trainer = pl.Trainer(max_epochs=15, check_val_every_n_epoch=1)
    trainer.fit(model=model, datamodule=datamodule)


def main():
    train()


if __name__ == "__main__":
    main()