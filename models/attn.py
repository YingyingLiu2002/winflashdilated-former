import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from flash_attn import flash_attn_func

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        chunk_size = 64
        K_sample_list = []
        for i in range(0, L_Q, chunk_size):
            end = min(i + chunk_size, L_Q)
            Q_chunk = Q[:, :, i:end, :]
            K_expand_chunk = K.unsqueeze(-3).expand(B, H, end - i, L_K, E)
            index_sample = torch.randint(L_K, (end - i, sample_k), device=Q.device)
            K_sample_chunk = K_expand_chunk[:, :, torch.arange(end - i).unsqueeze(1), index_sample, :]
            K_sample_list.append(K_sample_chunk)

        K_sample = torch.cat(K_sample_list, dim=2) if K_sample_list else K_expand_chunk
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, layer_num=0):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.layer_num = layer_num

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if self.inner_attention is None:
            out = flash_attn_func(
                queries, keys, values,
                dropout_p=0.1 if self.training else 0.0,
                causal=False,
                window_size=(-1, -1),
                softmax_scale=None
            )
            out = out.view(B, L, -1)
            return self.out_projection(out), None
        else:
            out, attn = self.inner_attention(queries, keys, values, attn_mask)
            if self.mix:
                out = out.transpose(2, 1).contiguous()
            out = out.view(B, L, -1)
            return self.out_projection(out), attn

# attn.py
class AttentionLayerWin(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False, layer_num=0,
                 window_size=8, output_attention=False, dropout=0.1, block_size=64, window_sizes=None):
        super(AttentionLayerWin, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.mix = mix
        self.layer_num = layer_num
        self.window_size = window_size
        self.output_attn = output_attention
        self.dropout = dropout
        self.block_size = block_size
        self.window_sizes = window_sizes  # 多个窗口大小

        # 如果 window_sizes 不为 None，启用多尺度窗口
        if self.window_sizes is not None:
            self.multi_scale = True
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"AttentionLayerWin using multi-scale windows: {self.window_sizes}")
            # 为每个 window_size 创建独立的投影层（可选）
            self.scale_weights = nn.Parameter(torch.ones(len(self.window_sizes)))  # 每个尺度的权重
        else:
            self.multi_scale = False

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if self.multi_scale:
            # 多尺度窗口注意力
            outputs = []
            for window_size in self.window_sizes:
                # 确保序列长度可以被 window_size 整除
                if L % window_size != 0 or S % window_size != 0:
                    # 填充序列长度
                    pad_L = (window_size - L % window_size) % window_size
                    pad_S = (window_size - S % window_size) % window_size
                    queries_padded = F.pad(queries, (0, 0, 0, 0, 0, pad_L))  # (B, L+pad_L, H, D)
                    keys_padded = F.pad(keys, (0, 0, 0, 0, 0, pad_S))
                    values_padded = F.pad(values, (0, 0, 0, 0, 0, pad_S))
                    L_padded = L + pad_L
                    S_padded = S + pad_S
                else:
                    queries_padded = queries
                    keys_padded = keys
                    values_padded = values
                    L_padded = L
                    S_padded = S

                # 按 window_size 分割
                queries_win = queries_padded.view(B * (L_padded // window_size), window_size, H, -1)
                keys_win = keys_padded.view(B * (S_padded // window_size), window_size, H, -1)
                values_win = values_padded.view(B * (S_padded // window_size), window_size, H, -1)

                if self.inner_attention is None:
                    out = flash_attn_func(
                        queries_win, keys_win, values_win,
                        dropout_p=self.dropout if self.training else 0.0,
                        causal=False,
                        window_size=(-1, -1),
                        softmax_scale=None,
                    )
                else:
                    out, _ = self.inner_attention(queries_win, keys_win, values_win, attn_mask)

                # 恢复原始序列长度
                out = out.view(B, L_padded, H, -1)
                if L_padded != L:
                    out = out[:, :L, :, :]  # 截断填充部分

                outputs.append(out)

            # 融合多尺度输出
            scale_weights = F.softmax(self.scale_weights, dim=0)  # 归一化权重
            out = torch.zeros_like(outputs[0])
            for i, output in enumerate(outputs):
                out += scale_weights[i] * output  # 加权平均

        else:
            # 单尺度窗口（原有逻辑）
            if L % self.window_size != 0 or S % self.window_size != 0:
                raise ValueError(f"Sequence length {L} or {S} not divisible by window_size {self.window_size}")

            queries = queries.view(B * (L // self.window_size), self.window_size, H, -1)
            keys = keys.view(B * (S // self.window_size), self.window_size, H, -1)
            values = values.view(B * (S // self.window_size), self.window_size, H, -1)

            if self.inner_attention is None:
                out = flash_attn_func(
                    queries, keys, values,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                    window_size=(-1, -1),
                    softmax_scale=None,
                )
            else:
                out, attn = self.inner_attention(queries, keys, values, attn_mask)

        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"Warning: AttentionLayerWin output contains NaN or Inf, sample: {out[0, :5]}")

        if self.output_attn and self.inner_attention is not None and not self.multi_scale:
            attn = self._output_attn(L, attn)
        else:
            attn = None

        out = out.view(B, L, H, -1)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

    def _output_attn(self, L, attn):
        num_window = L // self.window_size
        for k in range(num_window):
            if k == 0:
                p2d = (0, ((num_window - (k + 1)) * self.window_size))
                attn_tmp = F.pad(attn[:self.window_size, :, :, :], p2d)
            else:
                p2d = (k * self.window_size, (num_window - (k + 1)) * self.window_size)
                attn_tmp = torch.cat((attn_tmp, F.pad(attn[k * self.window_size:(k + 1) * self.window_size, :, :, :], p2d)), dim=2)
        return attn_tmp

# attn.py
class AttentionLayerCrossWin(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False, layer_num=0,
                 num_windows=4, output_attention=False, dropout=0.1, block_size=64, window_sizes=None):
        super(AttentionLayerCrossWin, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.mix = mix
        self.layer_num = layer_num
        self.num_windows = num_windows
        self.output_attn = output_attention
        self.dropout = dropout
        self.block_size = block_size
        self.window_sizes = window_sizes  # 多个窗口大小

        # 如果 window_sizes 不为 None，启用多尺度窗口
        if self.window_sizes is not None:
            self.multi_scale = True
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"AttentionLayerCrossWin using multi-scale windows: {self.window_sizes}")
            self.scale_weights = nn.Parameter(torch.ones(len(self.window_sizes)))  # 每个尺度的权重
        else:
            self.multi_scale = False

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if self.multi_scale:
            # 多尺度窗口注意力
            outputs = []
            for window_size in self.window_sizes:
                # 确保序列长度可以被 window_size 整除
                if L % window_size != 0 or S % window_size != 0:
                    pad_L = (window_size - L % window_size) % window_size
                    pad_S = (window_size - S % window_size) % window_size
                    queries_padded = F.pad(queries, (0, 0, 0, 0, 0, pad_L))
                    keys_padded = F.pad(keys, (0, 0, 0, 0, 0, pad_S))
                    values_padded = F.pad(values, (0, 0, 0, 0, 0, pad_S))
                    L_padded = L + pad_L
                    S_padded = S + pad_S
                else:
                    queries_padded = queries
                    keys_padded = keys
                    values_padded = values
                    L_padded = L
                    S_padded = S

                # 按 window_size 分割
                queries_win = queries_padded.view(B * (L_padded // window_size), L_padded // window_size, H, -1)
                keys_win = keys_padded.view(B * (S_padded // window_size), S_padded // window_size, H, -1)
                values_win = values_padded.view(B * (S_padded // window_size), S_padded // window_size, H, -1)

                if self.inner_attention is None:
                    out = flash_attn_func(
                        queries_win, keys_win, values_win,
                        dropout_p=self.dropout if self.training else 0.0,
                        causal=False,
                        window_size=(-1, -1),
                        softmax_scale=None,
                    )
                else:
                    out, _ = self.inner_attention(queries_win, keys_win, values_win, attn_mask)

                # 恢复原始序列长度
                out = out.view(B, L_padded, H, -1)
                if L_padded != L:
                    out = out[:, :L, :, :]

                outputs.append(out)

            # 融合多尺度输出
            scale_weights = F.softmax(self.scale_weights, dim=0)
            out = torch.zeros_like(outputs[0])
            for i, output in enumerate(outputs):
                out += scale_weights[i] * output

        else:
            # 单尺度窗口（原有逻辑）
            if L % self.num_windows != 0 or S % self.num_windows != 0:
                raise ValueError(f"Sequence length {L} or {S} not divisible by num_windows {self.num_windows}")

            queries = queries.view(B * self.num_windows, L // self.num_windows, H, -1)
            keys = keys.view(B * self.num_windows, S // self.num_windows, H, -1)
            values = values.view(B * self.num_windows, S // self.num_windows, H, -1)

            if self.inner_attention is None:
                out = flash_attn_func(
                    queries, keys, values,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                    window_size=(-1, -1),
                    softmax_scale=None,
                )
            else:
                out, attn = self.inner_attention(queries, keys, values, attn_mask)

        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"Warning: AttentionLayerCrossWin output contains NaN or Inf, sample: {out[0, :5]}")

        if self.output_attn and self.inner_attention is not None and not self.multi_scale:
            attn = self._output_attn(L, attn)
        else:
            attn = None

        out = out.view(B, L, H, -1)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

    def _output_attn(self, L, attn):
        window_size = L // self.num_windows
        for k in range(self.num_windows):
            if k == 0:
                p2d = (0, ((self.num_windows - (k + 1)) * window_size))
                attn_tmp = F.pad(attn[:window_size, :, :, :], p2d)
            else:
                p2d = (k * window_size, (self.num_windows - (k + 1)) * window_size)
                attn_tmp = torch.cat((attn_tmp, F.pad(attn[k * window_size:(k + 1) * window_size, :, :, :], p2d)), dim=2)
        return attn_tmp

# FlashAttentionLayer 已经支持 block_size，无需修改
class FlashAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1, causal=False, block_size=64):
        super(FlashAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = dropout
        self.causal = causal

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = flash_attn_func(
            queries, keys, values,
            dropout_p=self.dropout if self.training else 0.0,
            causal=self.causal,
            window_size=(-1, -1),
            softmax_scale=None
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), None