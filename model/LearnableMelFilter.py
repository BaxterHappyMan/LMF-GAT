import torch
from torch import nn
import torchaudio
import torch.nn.functional as F


class LearnableMelFilter(nn.Module):
    def __init__(self, n_mel, sample_rate, n_fft, f_min=1, f_max=None, mfcc=False, init_mel=True):
        super(LearnableMelFilter, self).__init__()
        self.n_mel = n_mel
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.freq = torch.linspace(0, self.sample_rate//2, self.n_fft//2 + 1)
        if init_mel:
            fb = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mel,
                sample_rate=self.sample_rate,
                norm='slaney'
            ).T
            self.register_buffer('base_fb', fb)
            self.logit_weights = nn.Parameter(torch.rand_like(fb)*0.01) # 可学习参数
        else:
            # 非负约束
            self.weights = nn.Parameter(torch.rand(self.n_mel, self.n_fft//2+1) * self.freq[None: ]/self.f_max)

    def forward(self, spectrogram):
        if hasattr(self, 'base_fb'):
            # 保持平滑
            weights = self.base_fb * F.softplus(self.logit_weights)
        else:
            weights = F.softplus(self.weights)
        # mel_spec = torch.einsum("ft,mf->mt", spectrogram, weights)
        mel_spec = torch.einsum("tf,mf->tm", spectrogram, weights)
        return mel_spec