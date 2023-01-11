import json
import os

from torch.utils.data import DataLoader
from torchaudio.datasets import MUSDB_HQ
from tqdm import tqdm


def main():

    ROOT = "../../../data"

    chunk_length = 3
    source = 'vocals'
    subset = 'train'
    sources = ['mixture']

    for split in {'train', 'validation'}:
        musdb = DataLoader(MUSDB_HQ(root=ROOT, subset=subset, split=split, sources=sources))

        lengths = [int(length) for _, sampling_rate, length, name in tqdm(musdb)]

        with open(os.path.join(ROOT, f"musdb18hq/lengths_{split}.json"), "w") as f:
            json.dump(lengths, f)

    subset = 'test'
    split = None

    musdb = DataLoader(MUSDB_HQ(root=ROOT, subset=subset, split=split, sources=sources))

    lengths = [int(length) for _, sampling_rate, length, name in tqdm(musdb)]

    with open(os.path.join(ROOT, f"musdb18hq/lengths_{subset}.json"), "w") as f:
        json.dump(lengths, f)


if __name__ == "__main__":
    main()
