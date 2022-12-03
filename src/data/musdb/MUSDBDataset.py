import json
import os
from typing import Optional, List, Tuple

import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import MUSDB_HQ
from tqdm import tqdm

ROOT = "data"


class MUSDBDataset(Dataset):
    def __init__(self,
                 chunk_length: int,
                 sample_rate: int,
                 source: str,
                 subset: str,
                 split: Optional[str]):
        """
        Dataset class for MUSDB
        Args:
            chunk_length: length of audio chunks in seconds
            sample_rate: sample rate of audio in Hertz
            source: instrument to separate, i.e. 'bass'
            subset: subset of dataset ('train' or 'test')
            split: None if subset is 'test', otherwise 'train' or 'validation'
        """
        super(MUSDBDataset, self).__init__()

        self.sample_rate: int = sample_rate
        self.chunk_length: int = chunk_length

        self.sources: List[str] = ["mixture", source]

        if subset in ("train", "validation"):
            assert split is not None

        self.musdb: Dataset = MUSDB_HQ(root=ROOT,
                                       subset=subset,
                                       split=split,
                                       sources=self.sources)

        with open(os.path.join(ROOT, "musdb18hq/lengths.json")) as f:
            lengths: List[int] = json.load(f)

        self.lengths: List[int] = [
            int(length // (self.chunk_length * self.sample_rate))
            for length in lengths
        ]

        self.chunks: List[Tuple[int, int]] = [
            (i, chunk_length * self.sample_rate * chunk)
            for i, length in enumerate(self.lengths)
            for chunk in range(length)
        ]

        self.current_song: int = -1
        self.current_waveform: Optional[Tensor] = None

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        song, start = self.chunks[index]

        if song != self.current_song:
            waveform, _, _, _ = self.musdb[song]
            self.current_waveform = waveform
            self.current_song += 1

        chunked_waveform = self.current_waveform[:, :, start:start + self.sample_rate * self.chunk_length]

        mix, target = chunked_waveform[0, :, :].squeeze(0), chunked_waveform[1, :, :].squeeze(0)

        return mix, target

    def __len__(self) -> int:
        return len(self.chunks)


class MUSDBDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 sample_rate: int,
                 chunk_length: int,
                 source: str
                 ):
        """
        Datamodule for time series dataset
        Args:
            batch_size: batch size used for training
            sample_rate: audio sample rate in Hertz
            chunk_length: length of audio chunks in seconds
            source: instrument to separate
        """
        super().__init__()
        self.batch_size = batch_size
        self.sources = ["mixture", source]

        self.train = MUSDBDataset(
            chunk_length=chunk_length,
            sample_rate=sample_rate,
            source=source,
            subset='train',
            split='train'
        )
        self.valid = MUSDBDataset(
            chunk_length=chunk_length,
            sample_rate=sample_rate,
            source=source,
            subset='train',
            split='validation'
        )
        self.test = MUSDBDataset(
            chunk_length=chunk_length,
            sample_rate=sample_rate,
            source=source,
            subset='test',
            split=None
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, self.batch_size)


def main():
    datamodule = MUSDBDataModule(
        chunk_length=3,
        sample_rate=44100,
        batch_size=4,
        source='drums'
    )

    for mix, target in tqdm(datamodule.train_dataloader()):
        print(mix.shape, target.shape)
        break
        pass


if __name__ == "__main__":
    main()
