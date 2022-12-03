from torch import stft, istft, nn


class Spectrogram(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, x):
        x = x.reshape(4,-1)
        x = stft(x, n_fft=self.n_fft, hop_length=490)
        x = x.permute(0,3,1,2)
        _, _, freq, time = x.shape
        x = x.reshape(2, -1, freq, time)
        return x


class InverseSpectrogram(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, x):
        batch, _, freq, time = x.shape
        x = x.reshape(batch*2, freq, time, 2)
        x = istft(x, n_fft=self.n_fft, hop_length=490)
        x = x.reshape(batch, 2, -1)
        return x