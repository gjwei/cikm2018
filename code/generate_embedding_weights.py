#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2018/5/21
  
"""
import numpy as np
from tqdm import tqdm
from config import config
from utils.data_utils import load_vocab
from utils.utils import norm_weight
from nltk.stem.snowball import SnowballStemmer

spanish_stemer = SnowballStemmer('spanish')

def get_vocabs(vocab_file):
    words = load_vocab(vocab_file)
    word_dict = {}
    for i in range(len(words)):
        word_dict[words[i]] = i
    return word_dict


def stem_word(word):
    return spanish_stemer.stem(word)


def embed2vec(embedding_file, dim):
    """
    embedding -> numpy
    """
    vocab = get_vocabs(config.vocab_path)
    embedding_vec = load_embedding(embedding_file)
    
    weight_matrix = norm_weight(len(vocab), dim)

    words_found = 0
    unfind_words = []
    find_words = []

    for word, index in vocab.items():

        if word in embedding_vec:
            weight_matrix[index] = embedding_vec[word]
            words_found += 1
        else:
            if stem_word(word) in embedding_vec:
                weight_matrix[index] = embedding_vec[stem_word(word)]
                words_found += 1
            else:
                unfind_words.append(word)

    
    print('找到词向量的单词数有{}，没有找到的有{}'.format(words_found, len(vocab) - words_found))
    np.savez(config.embed_path, weights=weight_matrix)

    with open('./input/unfind_word.txt', 'wt', encoding='utf-8') as f:
        for line in unfind_words:
            f.write(line + '\n')

    print('Done')


def embed2vec_with_english(spanish_embeding_file, english_embedding_file, vocab_file, dim):
    """
    embedding -> numpy
    """

    vocab = load_vocab(vocab_file)

    print('vocab size is {}'.format(len(vocab)))

    weight_matrix = norm_weight(len(vocab), dim)

    weight_matrix[0] = 0.0

    words_found = 0
    unfind_words = []
    find_words = []

    spanish_embedding_vec = load_embedding(spanish_embeding_file)
    english_embedding_vec = load_embedding(english_embedding_file)

    for index, word in enumerate(vocab):

        if word in spanish_embedding_vec:
            weight_matrix[index] = spanish_embedding_vec[word]
            words_found += 1
            find_words.append(word)
        elif word in english_embedding_vec:
            weight_matrix[index] =english_embedding_vec[word]
            find_words.append(word)
        else:
            unfind_words.append(word)

    print('找到词向量的单词数有{}，没有找到的有{}'.format(words_found, len(vocab) - words_found))
    np.savez(config.all_vocab_embedding_file, weights=weight_matrix)

    with open('{}unfind_word.txt'.format(config.multi_task_path), 'wt', encoding='utf-8') as f:
        for line in unfind_words:
            f.write(line + '\n')

    print('Done')


def load_embedding(embedding_file):
    """从文件中加载现有的embedding模型"""
    embedding_dict = {}
    with open(embedding_file, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue
            else:
                try:
                    line = line.strip().split(' ')
                    word = line[0]
                    vec = np.array(line[1:], dtype=np.float32)
                    embedding_dict[word] = vec
                except ValueError:
                    print(line)
                    continue
    return embedding_dict
    

if __name__ == '__main__':
    embed2vec(config.fasttext_file_path, 300)
    print('spanish embedding generate done')
    embed2vec_with_english(config.fasttext_file_path, config.fasttext_file_english_path,
                           config.all_vocab_path, dim=300)


