import pytorch_lightning as pl
from torchaudio.datasets import MUSDB_HQ
from torch.utils.data import DataLoader

ROOT = "../../../data"


class MUSDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        """
        Datamodule for time series dataset
        Args:
            batch_size: batch size used for training
        """
        super().__init__()
        self.batch_size = batch_size

        self.train = MUSDB_HQ(root=ROOT, subset="train", split="train")
        self.valid = MUSDB_HQ(root=ROOT, subset="train", split="validation")
        self.test = MUSDB_HQ(root=ROOT, subset="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, self.batch_size)


def main():
    dataset = MUSDB_HQ(
        root="../../../data",
        subset="train",
        download=True
    )

    print(dataset.__getitem__(0))


if __name__ == "__main__":
    main()
