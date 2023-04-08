import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from .DLinear import series_decomp


class Model(nn.Module):
    """
    DInformer :
    In order to prove that Dlinear performs well in a single variable, it only improves performance because
    its decomposition operation makes the task easier, so we make a decomposition with Informer.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention

        # decomposition
        self.Decompsition = series_decomp(configs.avg_kernel_size)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)      # 2 layers
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for _ in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)    # 1 layer
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc:          EncodeInput——{batch, seq_len, channel}

        x_mark_enc:     Encoder_TimeFeature——{batch, seq_len, time_freq}
                        [time_freq] is include [0:month, 1:day, 2:weekday, 3:hour, 4:minute]

        x_dec:          DecodeInput——{batch, label_len+pred_len, channel}
                        The label len is the overlap of the encoder input and the decoder input
                        {:, -pred_len, :} is zero

        x_mark_dec：     Decoder_TimeFeature——{batch, label+pred_len, time_freq}
        """
        seasonal_init, trend_init = self.Decompsition(x_enc)
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)     # return {bath, seq_len, d_model}
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)     # return {bath, label_len+pred_len}, d_model}
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :]+trend_output, attns
        else:
            return dec_out[:, -self.pred_len:, :]+trend_output  # [B, L, D]
