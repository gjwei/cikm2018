#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 7/14/18
  
"""
import torch
import torch.nn as nn
from models.utils.CharCNNEmbeddingLayer import CharCNNEmbedding
import ipdb


class Embedding(nn.Module):
    
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        
        self.char_vocab_size = config.char_vocab_size
        self.char_dim = config.char_dim
        self.char_kernel_sizes = config.char_kernel_sizes
        self.char_kernel_nums = config.char_kernel_nums
        
        self.embedding_size = config.embed_size
        self.max_sentence_length = config.max_sentence_length
        self.word_max_length = config.max_word_length
        

        self.word_embedding = nn.Embedding(self.vocab_size,
                                           self.embedding_size,
                                           padding_idx=0)

        # pos embedding information
        # self.pos_vocab = config.pos_vocab
        # self.pos_vocab_size = config.pos_vocab_size
        # self.pos_embedding = nn.Embedding(self.pos_vocab_size + 1,
        #                                   self.config.pos_embedding_size,
        #                                   padding_idx=self.pos_vocab_size)
        
        self.char_embedding = CharCNNEmbedding(char_vocab=self.char_vocab_size,
                                               char_dim=self.char_dim,
                                               word_max_length=self.word_max_length,
                                               embed_dropout=config.embed_dropout,
                                               kernel_sizes=self.char_kernel_sizes,
                                               feature_maps=self.char_kernel_nums,
                                               output_dim=config.char_output_dim)
        # ipdb.set_trace()
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.pretrained_emb))

        self.word_embedding.weight.requires_grad = False
        
    def forward(self, words, chars):
        """
        :param words: batch_size, seq_max_length
        :param chars: batch_size, seq_max_length, word_max_length
        :return:batch, seq_max_length,  word_embedding + char_embedding
        """

        words_embedding = self.word_embedding(words)  # bs, seq_max_length, word_dim
        chars_embedding = self.char_embedding(chars)  # bs, seq_max_length, char_dim
        embedding = torch.cat([words_embedding,
                               chars_embedding
                               ], dim=-1)  # bs, seq_length, word_dim + char_dim

        return embedding
        
        
        
        

        

