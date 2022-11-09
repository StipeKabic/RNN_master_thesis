from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

DatasetMapping = Dict[str, Dict[str, np.ndarray]]


def generate_time_series(batch_size: int, n_steps: int) -> np.ndarray:
    """
    generates a batch of time series
    Args:
        batch_size: number of time series
        n_steps: length of time series

   Returns: numpy array containing the time series
    """
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


def generate_train_test(n_steps: int) -> DatasetMapping:
    series = generate_time_series(10000, n_steps + 1)
    x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    x_test, y_test = series[9000:, :n_steps], series[9000:, -1]

    return {
        "train": {
            "x": x_train,
            "y": y_train
        },
        "valid": {
            "x": x_valid,
            "y": y_valid
        },
        "test": {
            "x": x_test,
            "y": y_test
        },
    }

    
class TSDataset(Dataset):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray
    ):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class TSDataModule(pl.LightningDataModule):
    def __init__(self, n_steps: int, batch_size: int):
        super().__init__()
        self.n_steps = n_steps
        self.batch_size = batch_size

        datasets = generate_train_test(n_steps)

        self.train = TSDataset(**datasets["train"])
        self.valid = TSDataset(**datasets["valid"])
        self.test = TSDataset(**datasets["test"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, self.batch_size)


def main():
    dataset_manager = TSDataModule(n_steps=50, batch_size=4)
    train = dataset_manager.train_dataloader()
    for item in train:
        print(item[0].shape, item[1].shape)
        break


if __name__ == "__main__":
    main()
