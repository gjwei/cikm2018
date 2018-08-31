# coding: utf-8
# Author: gjwei

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.utils.functions import compute_mask


class WordEmbedding(nn.Module):
    """
    Embedding Layer, also compute the mask of padding index
    Args:
        Inputs: (batch, seq_len): sequences with word index

    returns:
        output:  batch,seq_len, embedding_sizse
        mask: batch, seq_len: tensor show which index is padding
    """

    def __init__(self, vocab_size, embed_size, padding_index=0, weights=None):
        super(WordEmbedding, self).__init__()
        self.padding_idx = padding_index

        self.embedding_fixed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        self.embedding_updated = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        if weights is not None:
            self.embedding_fixed.weight.data = weights
            self.embedding_updated.data = weights
            self.embedding_fixed.weight.requires_grad = False
            self.embedding_updated.weight.requires_grad = True

    def forward(self, x):
        """
        :param x: batch, seq_len
        :return:
        """
        mask = compute_mask(x, padding_idx=self.padding_idx)
        embed_fixed = self.embedding_fixed(x)
        embed_updated = self.embedding_updated(x)
        return embed_fixed, embed_updated, mask


class CharCNNEmbedding(nn.ModuleList):
    """
    Char-level CNN
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (batch, seq_len, hidden_size)
    """

    def __init__(self, emb_size, filters_size, filters_num, dropout_p):
        super(CharCNNEmbedding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, fn, (fw, emb_size)) for fw, fn in zip(filters_size, filters_num)])

    def forward(self, x, char_mask, word_mask):
        x = self.dropout(x)

        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size).unsqueeze(1)  # (N, 1, word_len, embedding_size)

        x = [F.relu(cnn(x)).squeeze(-1) for cnn in self.cnns]  # (N, Cout, word_len - fw + 1) * fn
        x = [torch.max(cx, 2)[0] for cx in x]  # (N, Cout) * fn
        x = torch.cat(x, dim=1)  # (N, hidden_size)

        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x = x * word_mask.unsqueeze(-1)

        return x


class Highway(torch.nn.Module):
    def __init__(self, in_size, n_layers, dropout_p):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.normal_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x


class CharCNNEncoder(torch.nn.Module):
    """
    char-level cnn encoder with highway networks
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, hidden_size, filters_size, filters_num, dropout_p, enable_highway=True):
        super(CharCNNEncoder, self).__init__()
        self.enable_highway = enable_highway
        self.hidden_size = hidden_size

        self.cnn = CharCNNEmbedding(emb_size=emb_size,
                                    filters_size=filters_size,
                                    filters_num=filters_num,
                                    dropout_p=dropout_p)

        if enable_highway:
            self.highway = Highway(in_size=hidden_size,
                                   n_layers=2,
                                   dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        o = self.cnn(x, char_mask, word_mask)

        assert o.shape[2] == self.hidden_size
        if self.enable_highway:
            o = self.highway(o)

        return o


def