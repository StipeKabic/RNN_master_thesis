import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LSTM


class SeparationBLSTM(nn.Module):
    def __init__(
            self,
            nb_bins: int,
            nb_channels: int,
            hidden_size: int,
            nb_layers: int ,
            dropout: float
    ):
        """
        BLSTM model for music source separation
        Args:
            nb_bins: number of spectrogram bins
            nb_channels: number of audio channels
            hidden_size: rnn feature space dimension
            nb_layers: number of BLSTM layers
        """
        super(SeparationBLSTM, self).__init__()
        lstm_hidden_size = hidden_size // 2
        fc2_hiddensize = hidden_size * 2

        self.nb_output_bins = nb_bins
        self.nb_bins = self.nb_output_bins
        self.hidden_size = hidden_size

        self.fc1 = nn.Sequential(
            Linear(self.nb_bins * nb_channels, hidden_size, bias=False),
            BatchNorm1d(hidden_size)
        )

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if nb_layers > 1 else 0
        )

        self.fc2 = nn.Sequential(
            Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False),
            BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False
            )
        )

        print(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: nb_samples, nb_channels, nb_bins, nb_frames
        Returns:
            y: nb_samples, nb_channels, nb_bins, nb_frames
        """

        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        mix = x

        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)

        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)

        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.fc3(x)

        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x = F.relu(x) * mix

        return x.permute(1, 2, 3, 0)
