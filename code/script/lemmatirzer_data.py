# coding: utf-8
# Author: gjwei
import sys
sys.path.append('..')
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
import numpy as np



from code.utils import read_data


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


sbsp = SnowballStemmer('spanish')

unfind_words = read_data('../input/unfind_word.txt')

# word_embeds = load_embedding('../fasttext/wiki.es.vec')

# save embeds words
word_embeds = set()
with open('./embeds.txt', 'rt', encoding='utf-8') as f:
    for word in f.readlines():
        word_embeds.add(word.strip())

unfind_words = [word.strip() for word in unfind_words]

stem_unfind_words = [sbsp.stem(item) for item in unfind_words]

unfind_words_and_stem = dict(zip(stem_unfind_words, unfind_words))

# 查看下能在embedding找到的stem words
count = 0

stem_unfind_words_in_embeds = []

for word in stem_unfind_words:
    if word in word_embeds:
        count += 1
        stem_unfind_words_in_embeds.append(word)

with open('./stem_unfind_word_in_embeds.txt', 'wt', encoding='utf-8') as f:
    for word in stem_unfind_words_in_embeds:
        f.write("{}\t{}\n".format(unfind_words_and_stem[word], word))
print('共同发现stem words {}个有词向量'.format(count))
