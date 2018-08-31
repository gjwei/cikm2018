# coding: utf-8
# Author: gjwei
import numpy as np
from torch.utils.data import Dataset
from code.config import config
from code.utils import onehot


class MultiTaskCikmDataset(Dataset):
    def __init__(self, paths, max_length=20, one_hot=False):

        self.spanish1 = []
        self.spanish2 = []
        self.labels = []

        words = config.multi_task_vocabs
        char_vocabs = config.all_char_vocab

        self.word_dict = {words[i]: i for i in range(len(words))}
        self.char_vocabs = {char_vocabs[i]: i for i in range(len(char_vocabs))}

        self.max_length = max_length
        self.max_word_length = config.max_word_length

        self.spanish1, self.spanish2, self.labels, self.s1_lengths, self.s2_lengths, self.s1_chars, self.s2_chars \
            = self.tokenizer_sequences(paths)

        self.spanish1 = np.array(self.spanish1, dtype=int)
        self.spanish2 = np.array(self.spanish2, dtype=int)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.s1_lengths = np.array(self.s1_lengths, dtype=int)
        self.s2_lengths = np.array(self.s2_lengths, dtype=int)

        # ipdb.set_trace()
        #
        self.s1_chars = np.asarray(self.s1_chars, dtype=int)
        self.s2_chars = np.asarray(self.s2_chars, dtype=int)



        # ipdb.set_trace()

        if one_hot:
            self.labels = onehot(labels=self.labels, num_classes=config.num_classes)

    def __getitem__(self, index):
        return self.spanish1[index], self.spanish2[index], self.labels[index], \
               self.s1_lengths[index], self.s2_lengths[index], self.s1_chars[index], \
               self.s2_chars[index]

    def __len__(self):
        return len(self.labels)

    def tokenizer(self, seq, word_dict, max_length):
        """将一个句子进行转化成token，采用下截断的方法"""
        seq = seq.split()
        if len(seq) >= max_length:
            seq = seq[:max_length]
            len1 = max_length
        else:
            len1 = len(seq)
            seq = seq + [config.PAD_WORD for _ in range(max_length - len(seq))]

        result = []
        char_result = []
        for word in seq:
            if word not in word_dict:
                word = config.UNK_WORD
            result.append(word_dict[word])
            char_result.append(self.tokenizer_word(word, config.max_word_length))

        return result, len1, char_result

    def tokenizer_word(self, word, max_word_length):
        """将sequence转成成char_id形式，结果为[max_word_length, max_word_length, """
        word = word[:max_word_length]
        word_chars = []
        # 如果word是PAD，将结果全部填充为空格
        if word == config.PAD_WORD:
            word = ''

        for c in word:
            if c in self.char_vocabs:
                word_chars.append(self.char_vocabs[c])
            else:
                continue

        for i in range(max(0, max_word_length - len(word_chars))):
            word_chars.insert(0, self.char_vocabs[config.CHAR_PAD])

        assert len(word_chars) == max_word_length, 'wrong {}'.format(word)

        return word_chars

    def tokenizer_sequences(self, paths):
        """对整个文本进行token化，返回的分别是第一个问句，第二个问句和label"""
        s1, s2, labels = [], [], []
        s1_len, s2_len = [], []
        s1_chars, s2_chars = [], []

        for path in paths:
            with open(path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.split('\t')
                    label = line[0].strip()

                    seq1, len1, seq1_chars = self.tokenizer(line[1], self.word_dict, max_length=self.max_length)
                    seq2, len2, seq2_chars = self.tokenizer(line[2], self.word_dict, max_length=self.max_length)

                    s1.append(seq1)
                    s2.append(seq2)

                    s1_len.append(len1)
                    s2_len.append(len2)

                    s1_chars.append(seq1_chars)
                    s2_chars.append(seq2_chars)

                    labels.append(int(label))
        return s1, s2, labels, s1_len, s2_len, s1_chars, s2_chars