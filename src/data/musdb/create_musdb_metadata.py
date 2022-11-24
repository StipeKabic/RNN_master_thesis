import json
import os

from torch.utils.data import DataLoader
from torchaudio.datasets import MUSDB_HQ
from tqdm import tqdm


def main():

    ROOT = "../../../data"

    chunk_length = 3
    source = 'bass'
    subset = 'train'
    split = 'train'
    sources = ['mixture']

    musdb = DataLoader(MUSDB_HQ(root=ROOT, subset=subset, split=split, sources=sources))

    lengths = [int(length) for _, sampling_rate, length, name in tqdm(musdb)]

    with open(os.path.join(ROOT, "musdb18hq/lengths.json"), "w") as f:
        json.dump(lengths, f)


if __name__ == "__main__":
    main()
