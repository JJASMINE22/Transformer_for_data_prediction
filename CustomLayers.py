# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 source_size: int,
                 embedding_size: int,
                 multihead_num: int,
                 drop_rate: float
                 ):
        super(MultiHeadAttention, self).__init__()
        self.source_size = source_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.drop_rate = drop_rate

        self.linear = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_q = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_k = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_v = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.source_size)

    def forward(self, inputs, mask=None):

        assert isinstance(inputs, list)
        q = inputs[0]
        k = inputs[1]
        v = inputs[-1] if len(inputs) == 3 else k
        batch_size = q.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # note: attr split_size_or_sections indicates the size of the slice, not the num of the slice
        q = torch.cat(torch.split(q, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)

        attention = torch.matmul(q, k.transpose(2, 1))/torch.sqrt(torch.tensor(self.embedding_size
                                                                               //self.multihead_num).float())

        if mask is not None:
            # mask = mask.repeat(self.multihead_num, 1, 1)
            attention -= 1e+9 * mask
        attention = torch.softmax(attention, dim=-1)

        feature = torch.matmul(attention, v)
        feature = torch.cat(torch.split(feature, split_size_or_sections=batch_size, dim=0), dim=-1)

        output = self.linear(feature)
        output = torch.dropout(output, p=self.drop_rate, train=True)

        output = torch.add(output, inputs[0])

        output = self.layer_norm(output)

        return output, attention
