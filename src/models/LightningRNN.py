from torch import Tensor, nn, optim
import pytorch_lightning as pl


class LightningRNN(pl.LightningModule):
    def __init__(
            self,
            network: nn.Module,
            loss: nn.Module,
            spectrogram: nn.Module,
            inverse_spectrogram: nn.Module
            ):
        super(LightningRNN, self).__init__()
        self.network = network
        self.loss = loss
        self.spectrogram = spectrogram
        self.inverse_spectrogram = inverse_spectrogram

        self.pipeline = nn.Sequential(
            self.spectrogram,
            self.network,
            self.inverse_spectrogram
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectrogram(x)
        x = self.network(x)
        x = self.inverse_spectrogram(x)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.log('validation_loss', loss)
        return loss
