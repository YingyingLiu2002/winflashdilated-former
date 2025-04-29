import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer, DecoderLayerWithFlashAttention
from models.attn import FullAttention, ProbAttention, AttentionLayer, AttentionLayerWin, AttentionLayerCrossWin
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers-1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, pred_len=self.pred_len)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                        d_model, d_ff, dropout=dropout, activation=activation
                    ) for l in range(el)
                ],
                [ConvLayer(d_model) for l in range(el-1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, pred_len=self.pred_len)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

class FWinFlash(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), window_size=24, num_windows=4, dwindow_size=0,
                 block_size=64, dilation=2, learned_pos=False,
                 window_sizes=None, num_windows_list=None, time_dim=6):  # 添加 time_dim 参数
        super(FWinFlash, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        if dwindow_size == 0:
            dwindow_size = window_size

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, learned_pos=learned_pos)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout, learned_pos=learned_pos)

        # 编码器
        encoder_layers = nn.ModuleList()
        for l in range(e_layers):
            encoder_layers.append(EncoderLayer(
                AttentionLayerWin(None, d_model, n_heads, window_size=window_size, dropout=dropout,
                                 block_size=block_size, window_sizes=window_sizes),
                d_model, d_ff, dropout=dropout, activation=activation, dilation=dilation
            ))

        self.encoder = Encoder(
            encoder_layers,
            [ConvLayer(d_model, dilation=dilation, kernel_size=3) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
            dilation=dilation
        )

        # 解码器
        self.decoder = Decoder(
            [
                DecoderLayerWithFlashAttention(
                    d_model, n_heads, window_size=dwindow_size, num_windows=num_windows,
                    d_ff=d_ff, dropout=dropout, activation=activation,
                    block_size=block_size
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            window_size=dwindow_size,
            d_model=d_model,
            time_dim=time_dim  # 传递 time_dim
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                               pred_len=self.pred_len, x_mark=x_mark_dec)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]

class FWinFlashLite(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), window_size=24, num_windows=4, dwindow_size=0,
                 block_size=64, dilation=2, learned_pos=False,
                 window_sizes=None, num_windows_list=None, time_dim=6):  # 添加 time_dim 参数
        super(FWinFlashLite, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        if dwindow_size == 0:
            dwindow_size = window_size

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, learned_pos=learned_pos)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout, learned_pos=learned_pos)

        # 编码器
        encoder_layers = nn.ModuleList()
        for l in range(e_layers):
            encoder_layers.append(EncoderLayer(
                AttentionLayerWin(None, d_model, n_heads, window_size=window_size, dropout=dropout,
                                 block_size=block_size, window_sizes=window_sizes),
                d_model, d_ff, dropout=dropout, activation=activation, dilation=dilation
            ))

        self.encoder = Encoder(
            encoder_layers,
            [ConvLayer(d_model, dilation=dilation, kernel_size=3) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
            dilation=dilation
        )

        # 解码器
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayerWin(None, d_model, n_heads, window_size=dwindow_size, dropout=dropout,
                                     window_sizes=window_sizes),
                    AttentionLayerCrossWin(None, d_model, n_heads, num_windows=num_windows, dropout=dropout,
                                          window_sizes=window_sizes),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            window_size=dwindow_size,
            d_model=d_model,
            time_dim=time_dim  # 传递 time_dim
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, pred_len=self.pred_len,
                               x_mark=x_mark_dec)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]