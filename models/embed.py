# models/embed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, learned=False):
        super(PositionalEmbedding, self).__init__()
        if learned:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, mean=0, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model).float()
            pe.requires_grad = False

            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model)).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=5):  # 增大 kernel_size
        super(TokenEmbedding, self).__init__()
        padding = (kernel_size - 1) // 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(TemporalEmbedding, self).__init__()

        self.freq = freq
        self.d_model = d_model
        self.embed_type = embed_type

        if embed_type == 'fixed':
            self.time_embed = nn.Embedding(24 * 60 * 60 // self._parse_freq(freq), d_model)
        elif embed_type == 'learned':
            self.time_embed = nn.Linear(6, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def _parse_freq(self, freq):
        if freq == '5min':
            return 5 * 60
        freq_map = {'y': 365 * 24 * 60 * 60, 'm': 30 * 24 * 60 * 60, 'w': 7 * 24 * 60 * 60,
                    'd': 24 * 60 * 60, 'b': 24 * 60 * 60, 'h': 60 * 60, 't': 60}
        if freq[-1:] in freq_map:
            return freq_map[freq[-1:]]
        raise ValueError(f"Unsupported frequency: {freq}")

    def forward(self, x):
        if self.embed_type == 'fixed':
            x = x.long()
            x = self.time_embed(x)
        elif self.embed_type == 'learned':
            x = self.time_embed(x.float())
        return self.dropout(x)

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super(TimeFeatureEmbedding, self).__init__()

        freq_simple = self._parse_freq(freq)

        freq_map = {
            'h': 4,
            't': 6,
            's': 6,
            'm': 1,
            'a': 1,
            'w': 2,
            'd': 3,
            'b': 3,
            '5min': 6
        }
        d_inp = freq_map[freq_simple]
        self.embed = nn.Sequential(
            nn.Linear(d_inp, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)

    def _parse_freq(self, freq):
        if freq.endswith('min'):
            return 't'
        elif freq.endswith('h'):
            return 'h'
        elif freq.endswith('d'):
            return 'd'
        elif freq.endswith('w'):
            return 'w'
        elif freq.endswith('m'):
            return 'm'
        elif freq.endswith('s'):
            return 's'
        elif freq in ['b', 'a']:
            return freq
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

    def forward(self, x):
        x = self.embed(x.float())
        return self.dropout(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, learned_pos=False):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, kernel_size=5)  # 传递 kernel_size
        self.dropout = nn.Dropout(p=dropout)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq, dropout=dropout) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, freq=freq, dropout=dropout)
        self.position_embedding = PositionalEmbedding(d_model=d_model, learned=learned_pos)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x_value = self.value_embedding(x)
            if x_mark.size(1) != x.size(1):
                target_len = x.size(1)
                x_mark = x_mark[:, :target_len, :] if x_mark.size(1) > target_len else F.pad(x_mark, (0, 0, 0, target_len - x_mark.size(1)))
            x_temporal = self.temporal_embedding(x_mark)
            x_pos = self.position_embedding(x)
            x = x_value + x_temporal + x_pos
        return self.dropout(x)
