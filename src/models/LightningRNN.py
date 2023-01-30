import pytorch_lightning as pl
from torch import Tensor, nn, optim

from src.models.evaluation.metrics import calculate_metrics


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
        self.loss: nn.Module = loss
        self.learning_rate: float = learning_rate

        self.pipeline: nn.Module = nn.Sequential(
            spectrogram,
            network,
            inverse_spectrogram
        )

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.pipeline(x)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Tensor, batch_idx: int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.log('training_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.log('validation_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        metrics = calculate_metrics(target, output)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.log('test_loss', loss, prog_bar=True)

        metrics = calculate_metrics(target, output)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, prog_bar=True)

        return loss