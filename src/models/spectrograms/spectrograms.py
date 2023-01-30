from torch import stft, istft, nn
import numpy as np
import torch
import einops

class Spectrogram(nn.Module):
    def __init__(self, n_fft: int, batch_size: int):
        super().__init__()
        self.n_fft = n_fft
        self.batch_size = batch_size

    def forward(self, x):
        b, c, t = x.shape
        x = einops.rearrange(x, 'b c t -> (b c) t')
        x = stft(x, n_fft=self.n_fft, hop_length=490)
        x = einops.rearrange(x, '(b c1) f t (c2) -> b (c1 c2) f t', b=self.batch_size)
        return x


class InverseSpectrogram(nn.Module):
    def __init__(self, n_fft: int, batch_size: int):
        super().__init__()
        self.n_fft = n_fft
        self.batch_size = batch_size

    def forward(self, x):
        x = einops.rearrange(x, 'b (c1 c2) f t -> (b c1) f t c2', b=self.batch_size, c2=2)
        x = istft(x, n_fft=self.n_fft, hop_length=490)
        x = einops.rearrange(x, '(b c) t -> b c t', b=self.batch_size)
        return x


def main():
    n_fft = 1024
    batch_size = 4
    x = np.random.rand(batch_size, 2, 44100)
    x = torch.tensor(x)
    spec = Spectrogram(n_fft, batch_size=batch_size)
    ispec = InverseSpectrogram(n_fft, batch_size=batch_size)
    y = ispec(spec(x))

    print(torch.norm(y - x))


if __name__ == "__main__":
    main()