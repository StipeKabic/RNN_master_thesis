import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LSTM


class BLSTM(nn.Module):
    def __init__(
            self,
            nb_bins: int = 4096,
            nb_channels: int = 2,
            hidden_size: int = 512,
            nb_layers: int = 3,
    ):
        super(BLSTM, self).__init__()

        self.nb_output_bins = nb_bins
        self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        self.bn1 = BatchNorm1d(hidden_size)

        lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: nb_samples, nb_channels, nb_bins, nb_frames

        Returns:
            y: nb_samples, nb_channels, nb_bins, nb_frames
        """

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x

        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)

        lstm_out = self.lstm(x)

        x = torch.cat([x, lstm_out[0]], -1)

        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)

        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x = F.relu(x) * mix

        return x.permute(1, 2, 3, 0)


def main():
    nb_samples = 4
    nb_channels = 2
    nb_bins = 1024
    nb_frames = 200

    shape = (nb_samples, nb_channels, nb_bins, nb_frames)

    tensor = torch.rand(shape)
    print(tensor.shape)

    model = BLSTM(
        nb_bins=nb_bins,
        nb_channels=nb_channels,
        hidden_size=512,
        nb_layers=3
    )
    print(model)

    output = model(tensor)
    print(output.shape)

    return


if __name__ == "__main__":
    main()
