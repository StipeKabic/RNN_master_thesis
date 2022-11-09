from torch import Tensor, nn, optim
import pytorch_lightning as pl


class LightningRNN(pl.LightningModule):
    def __init__(
            self,
            network: nn.Module,
            loss: nn.Module
            ):
        super(LightningRNN, self).__init__()
        self.network = network
        self.loss = loss

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

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
