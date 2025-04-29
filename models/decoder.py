# models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn import AttentionLayerWin, FlashAttentionLayer, FullAttention, AttentionLayerCrossWin
from utils.masking import TriangularCausalMask

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 分别打印 x_mask 和 cross_mask 的形状
        x_mask_shape = "None" if x_mask is None else (
            x_mask.mask.shape if isinstance(x_mask, TriangularCausalMask) else x_mask.shape)
        cross_mask_shape = "None" if cross_mask is None else (
            cross_mask.mask.shape if isinstance(cross_mask, TriangularCausalMask) else cross_mask.shape)
        # print(f"Decoder input shape: {x.shape}, self_attn_mask: {x_mask_shape}, cross_attn_mask: {cross_mask_shape}")

        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, window_size=4, d_model=512, time_dim=6):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.window_size = window_size
        self.d_model = d_model
        self.time_projection = nn.Linear(time_dim, d_model) if d_model else None

    def forward(self, x, cross, x_mask=None, cross_mask=None, pred_len=None, x_mark=None):
        outputs = x.clone()
        current_len = outputs.size(1)
        target_len = current_len + pred_len
        target_len = ((target_len + self.window_size - 1) // self.window_size) * self.window_size
        extra_steps = (target_len - current_len) // self.window_size

        current_outputs = outputs.clone()
        final_outputs = outputs.clone()
        for t in range(extra_steps):
            # 利用 x_mark 增强时间感知
            if x_mark is not None and self.time_projection is not None:
                target_seq_len = current_len + (t + 1) * self.window_size
                time_emb = self.time_projection(x_mark[:, :target_seq_len, :])
                # 确保 time_emb 的序列长度与 current_outputs 匹配
                if time_emb.size(1) != current_outputs.size(1):
                    if time_emb.size(1) < current_outputs.size(1):
                        padding = torch.zeros(time_emb.size(0), current_outputs.size(1) - time_emb.size(1),
                                              time_emb.size(2), device=time_emb.device)
                        time_emb = torch.cat([time_emb, padding], dim=1)
                    else:
                        time_emb = time_emb[:, :current_outputs.size(1), :]
                current_outputs = current_outputs + time_emb

            for layer in self.layers:
                current_outputs = layer(current_outputs, cross, x_mask=x_mask, cross_mask=cross_mask)

            last_output = current_outputs[:, -self.window_size:, :].clone()
            # 添加残差连接，减少误差累积
            if t > 0:
                last_output = last_output + final_outputs[:, -self.window_size:, :].clone()
            final_outputs = torch.cat([final_outputs, last_output], dim=1)
            current_outputs = final_outputs[:, :current_len + (t + 1) * self.window_size, :].clone()
            torch.cuda.empty_cache()

        outputs = final_outputs
        if self.norm is not None:
            outputs = self.norm(outputs)
        return outputs

class DecoderLayerWithFlashAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size, num_windows, d_ff=None,
                 dropout=0.1, activation="relu", block_size=64):
        super(DecoderLayerWithFlashAttention, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = AttentionLayerWin(
            None, d_model, n_heads, window_size=window_size, dropout=dropout,
            block_size=block_size
        )
        self.cross_attention = AttentionLayerCrossWin(
            None, d_model, n_heads, num_windows=num_windows, dropout=dropout,
            block_size=block_size
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_new, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x.clone() + self.dropout(x_new)
        x = self.norm1(x)

        x_cross, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x.clone() + self.dropout(x_cross)
        x = self.norm2(x)

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x.clone() + y

        return self.norm3(x)