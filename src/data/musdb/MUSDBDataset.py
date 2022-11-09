import torchaudio


def main():
    dataset = torchaudio.datasets.MUSDB_HQ(
        root="../../../data",
        subset="train",
        download=True
    )

    print(dataset.__getitem__(0))


if __name__ == "__main__":
    main()