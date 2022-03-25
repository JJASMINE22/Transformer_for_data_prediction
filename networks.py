# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：networks.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
import config as cfg
from torch import nn
from CustomLayers import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self,
                 source_size: int,
                 embedding_size: int,
                 multihead_num: int,
                 num_layers: int,
                 drop_rate: float
                 ):
        super(Encoder, self).__init__()
        self.source_size = source_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.enc_layers = nn.ModuleList([MultiHeadAttention(source_size=self.source_size,
                                                            embedding_size=self.embedding_size,
                                                            multihead_num=self.multihead_num,
                                                            drop_rate=self.drop_rate)
                                         for i in range(self.num_layers)])

    def forward(self, x):

        enc_outputs, attentions = [], []
        for i in range(self.num_layers):
            x, attention = self.enc_layers[i]([x, x, x])
            enc_outputs.append(x)
            attentions.append(attention)

        return enc_outputs, attentions


class DecoderLayer(nn.Module):
    """
    Eliminate FeedForward mechanism
    """
    def __init__(self,
                 source_size,
                 embedding_size,
                 multihead_num,
                 drop_rate):
        super(DecoderLayer, self).__init__()
        self.source_size = source_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.drop_rate = drop_rate

        self.attn1 = MultiHeadAttention(source_size=self.source_size,
                                        embedding_size=self.embedding_size,
                                        multihead_num=self.multihead_num,
                                        drop_rate=self.drop_rate)
        self.attn2 = MultiHeadAttention(source_size=self.source_size,
                                        embedding_size=self.embedding_size,
                                        multihead_num=self.multihead_num,
                                        drop_rate=self.drop_rate)

    def forward(self, x, enc_output, mask=None):
        x, _ = self.attn1([x, x, x], mask)
        x, attention = self.attn2([x, enc_output])

        return x, attention


class Decoder(nn.Module):
    def __init__(self,
                 source_size: int,
                 embedding_size: int,
                 multihead_num: int,
                 num_layers: int,
                 drop_rate: float
                 ):
        super(Decoder, self).__init__()
        self.source_size = source_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.dec_layers = nn.ModuleList([DecoderLayer(source_size=self.source_size,
                                                      embedding_size=self.embedding_size,
                                                      multihead_num=self.multihead_num,
                                                      drop_rate=self.drop_rate)
                                         for i in range(self.num_layers)])

    def forward(self, x, enc_outputs, mask):

        attentions = []
        for i in range(self.num_layers):
            x, attention = self.dec_layers[i](x, enc_outputs[i], mask)
            attentions.append(attention)

        return x, attentions


class CreateModel(nn.Module):
    """
    Unlike the standard Transformer
    Padding_mask, Positional_encoding and other mechanisms are eliminated
    Use linear transformation instead of embedding operation
    """
    def __init__(self,
                 target_size: int,
                 source_size: int,
                 embedding_size: int,
                 multihead_num: int,
                 num_layers: int,
                 drop_rate: float
                 ):
        super(CreateModel, self).__init__()
        self.target_size = target_size
        self.source_size = source_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.num_layers = num_layers
        self.drop_rate = drop_rate

        self.linear_enc = nn.Linear(in_features=self.target_size,
                                    out_features=self.embedding_size)
        self.linear_dec = nn.Linear(in_features=self.target_size,
                                    out_features=self.embedding_size)
        self.linear = nn.Linear(in_features=self.embedding_size,
                                out_features=self.target_size)

        self.encoder = Encoder(source_size=self.source_size,
                               embedding_size=self.embedding_size,
                               multihead_num=self.multihead_num,
                               num_layers=self.num_layers,
                               drop_rate=self.drop_rate)

        self.decoder = Decoder(source_size=self.source_size,
                               embedding_size=self.embedding_size,
                               multihead_num=self.multihead_num,
                               num_layers=self.num_layers,
                               drop_rate=self.drop_rate)

        self.weight = nn.Parameter(data=torch.zeros(size=(1, 1, self.target_size)))

        self.init_params()

    def sequence_mask(self, x):

        seq_mask = torch.triu(torch.ones(size=(x.size(1), )*2), diagonal=1)
        seq_mask = seq_mask.unsqueeze(dim=0)

        return seq_mask

    def forward(self, src_seq, tgt_seq):
        """
        For training, use parallel inference
        For prediction, use sequential inference
        """
        # discard the gradient of the param weight
        with torch.no_grad():
            weight = self.weight.expand(src_seq.size(0), -1, -1)
            if tgt_seq is not None:
                tgt_seq = torch.cat([weight, tgt_seq], dim=1)
            else:
                tgt_seq = weight

        mask = self.sequence_mask(tgt_seq).to(cfg.device)

        enc_input = self.linear_enc(src_seq)
        dec_input = self.linear_dec(tgt_seq)

        enc_outputs, _ = self.encoder(enc_input)
        dec_output, _ = self.decoder(dec_input, enc_outputs, mask)

        output = self.linear(dec_output)

        return output

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    stddev = 1/math.sqrt(param.size(0))
                    torch.nn.init.normal_(param, std=stddev)
                else:
                    torch.nn.init.zeros_(param)

    def get_weights(self):

        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.') == 'weight':
                    weights.append(param)
                else:
                    continue
        return weights
