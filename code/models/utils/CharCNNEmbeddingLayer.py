#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 7/14/18
  
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import ipdb


class CharCNNEmbedding(nn.Module):
    
    def __init__(self,
                 char_vocab,
                 char_dim,
                 word_max_length=10,
                 embed_dropout=0.2,
                 kernel_sizes=[3, 4, 5],
                 feature_maps=[128, 128, 128],
                 output_dim=128):
        super(CharCNNEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab, char_dim,
                                           padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.cnns = nn.ModuleList(
                nn.Sequential(
                        nn.Conv1d(in_channels=char_dim,
                                  out_channels=fm,
                                  kernel_size=ks),
                        nn.Tanh(),
                        nn.MaxPool1d(kernel_size=word_max_length - ks + 1)
                )
                for fm, ks in zip(feature_maps, kernel_sizes)
        )
        self.dropout = nn.Dropout(0.5)
        
        self.output_layer = nn.Linear(sum(feature_maps), output_dim)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.char_embedding.weight, mean=0, std=0.01)
        nn.init.constant_(self.char_embedding.weight.data[0], 0.0)
        
        nn.init.xavier_normal_(self.output_layer.weight)
        
    def forward(self, x):
        """
        :param x: batch_size, seq_max_length, word_max_length
        :return: batch_size , seq_max_length, output_dim
        """
        batch_size, seq_max_length = x.size(0), x.size(1)
        x = x.view(-1, x.size(2))
    
        x = self.char_embedding(x)  # bs, ml, dim
        x = self.embed_dropout(x)
        
        x = x.permute(0, 2, 1)

        raw_cnn_ouputs = []
        
        for cnn in self.cnns:
            raw_cnn_ouputs.append(torch.squeeze(cnn(x)))
        
        # ipdb.set_trace()
        
        flatten = torch.cat(raw_cnn_ouputs, dim=-1)  # bs, sum(feature_maps)
        
        flatten = self.dropout(flatten)
        
        output = self.output_layer(flatten)  # bs output_dim
        
        output = output.view(batch_size, seq_max_length, -1)  # bs, seq_max_length, output_dim
        
        return output


if __name__ == '__main__':
    input = torch.from_numpy(np.random.randint(0, 99, size=(32, 20, 15)))
    model = CharCNNEmbedding(char_vocab=100, char_dim=50)
    output = model(input)
    
    print(output.size())
        
        
        
