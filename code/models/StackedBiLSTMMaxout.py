# coding: utf-8
# Author: gjwei

import torch.nn as nn
from utils import torch_util
import torch


class StackBiLSTMMaxout(nn.Module):
    def __init__(self, config, ):
        super(StackBiLSTMMaxout, self).__init__()
        h_size = config.h_size
        v_size = config.vocab_size
        d = config.embed_size
        mlp_d = config.mlp_d
        dropout_r = config.rnn_dropout
        dropout = config.dropout
        max_l = config.max_lengths
        num_class = config.num_classes

        self.Embd = nn.Embedding(v_size, d)
        self.Embd.weight.data.copy_(
            torch.from_numpy(config.pretrained_emb)
        )

        self.Embd.weight.requires_grad = False

        self.embed_dropout = nn.Dropout(dropout_r)

        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True,
                            dropout=dropout_r)

        self.lstm_1 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[1],
                              num_layers=1, bidirectional=True,
                              dropout=dropout_r)

        self.lstm_2 = nn.LSTM(input_size=(d + (h_size[0] + h_size[1]) * 2), hidden_size=h_size[2],
                              num_layers=1, bidirectional=True,
                              dropout=dropout_r)

        self.max_l = max_l
        self.h_size = h_size

        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, num_class)
        # self.softmax = nn.Softmax(dim=-1)

        self.classifier = nn.Sequential(*[self.mlp_1,
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          self.mlp_2,
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          self.sm,
                                          # self.softmax
                                          ])

    def display(self):
        for param in self.parameters():
            print(param.data.size())

    def forward(self, s1, l1, s2, l2):
        # charge [batch, seq] to [seq, batch]
        s1 = torch.transpose(s1, 0, 1)
        s2 = torch.transpose(s2, 0, 1)

        if self.max_l:
            l1 = l1.clamp(max=self.max_l)
            l2 = l2.clamp(max=self.max_l)
            if s1.size(0) > self.max_l:
                s1 = s1[:self.max_l, :]
            if s2.size(0) > self.max_l:
                s2 = s2[:self.max_l, :]

        p_s1 = self.Embd(s1)
        p_s1 = self.embed_dropout(p_s1)
        p_s2 = self.Embd(s2)
        p_s2 = self.embed_dropout(p_s2)

        s1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s1, l1)
        s2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)

        # Length truncate
        len1 = s1_layer1_out.size(0)
        len2 = s2_layer1_out.size(0)
        p_s1 = p_s1[:len1, :, :]  # [T, B, D]
        p_s2 = p_s2[:len2, :, :]  # [T, B, D]

        # Using residual connection
        s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

        s1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s1_layer2_in, l1)
        s2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([p_s1, s1_layer1_out, s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([p_s2, s2_layer1_out, s2_layer2_out], dim=2)

        s1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s1_layer3_in, l1)
        s2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = torch_util.max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
                              torch.abs(s1_layer3_maxout - s2_layer3_maxout),
                              s1_layer3_maxout * s2_layer3_maxout],
                             dim=1)

        out = self.classifier(features)
        return out

    def optimizer_schedule(self, lr=1e-3, weight_decay=0):
        # ignore_parameters = list(map(id, self.embedding.parameters()))
        # update_parameters = filter(lambda p: id(p) not in ignore_parameters, self.parameters())
        update_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(update_parameters, weight_decay=weight_decay, lr=lr, amsgrad=True)
        return optimizer



