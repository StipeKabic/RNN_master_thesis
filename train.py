import pytorch_lightning as pl
from torch import nn

from src.data.musdb.MUSDBDataset import MUSDBDataModule
from src.models.LightningRNN import LightningRNN
from src.models.architectures.BLSTM import BLSTM
from src.models.spectrograms.spectrograms import Spectrogram, InverseSpectrogram

N_FFT = 2048

def train():
    datamodule: pl.LightningDataModule = MUSDBDataModule(
        batch_size=2,
        sample_rate=44100,
        chunk_length=2,
        source='drums'
    )
    network: nn.Module = BLSTM(
        nb_bins = N_FFT//2 + 1,
        nb_channels=4,
        hidden_size=128,
        nb_layers=3
    )

    loss: nn.Module = nn.L1Loss()
    spectrogram: nn.Module = Spectrogram(n_fft=N_FFT)
    inverse_spectrogram: nn.Module = InverseSpectrogram(n_fft=N_FFT)

    model: pl.LightningModule = LightningRNN(
        network=network,
        loss=loss,
        spectrogram=spectrogram,
        inverse_spectrogram=inverse_spectrogram
    )

    trainer: pl.Trainer = pl.Trainer(max_epochs=15, check_val_every_n_epoch=1)
    trainer.fit(model=model, datamodule=datamodule)


def main():
    train()

if __name__ == "__main__":
    main()
