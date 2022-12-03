from torch import Tensor, nn, optim
import pytorch_lightning as pl


class LightningRNN(pl.LightningModule):
    def __init__(
            self,
            network: nn.Module,
            loss: nn.Module,
            spectrogram: nn.Module,
            inverse_spectrogram: nn.Module,
            learning_rate: float
            ):
        super(LightningRNN, self).__init__()
        self.loss = loss
        self.learning_rate = learning_rate

        self.pipeline = nn.Sequential(
            spectrogram,
            network,
            inverse_spectrogram
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pipeline(x)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

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
