# coding: utf-8
# Author: gjwei

# coding: utf-8

# TODO:
"""
1. 将大写转化为小写形式
2. 将倒挂的问好和感叹号转为正挂的，并加上空格

2. 将两个训练样本中的西班牙语和英语分别提取出来。应为比赛只是使用西班牙语进行测试的，所以，我们先只用西班牙语进行训练

西班牙语中的倒感叹号放在句首用来表示语气的感叹或者疑问。用倒挂问号¿和倒挂感叹号¡的初衷，是方便读者在阅读一些较长的句子时，从一开始就能知道该句子是疑问句或感叹句。

"""
import os
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from config import config
import re
from nltk.corpus import stopwords
from sklearn.utils import shuffle
import random
from collections import defaultdict
from utils.data_utils import save_data, read_data

core_nlp_path =  "/home/t-jiagao/workplace/corenlp/stanford-corenlp-full-2018-02-27/"

os.environ['CORENLP_HOME'] = core_nlp_path


def punctiation(s):
    s = s.lower()
    s = re.sub(r'[`|*;？_´]', r' ', s)
    s = re.sub(r"can′t", r"can not", s)
    s = re.sub(r'@', r' ', s)
    s = re.sub(r':', r' ', s)
    s = re.sub(r'¿', r'?', s)
    s = re.sub(r'¡', r'!', s)
    pattern = re.compile(r'([(),.?!\'\"\-$&;|/*#{}@])')
    s = re.sub(r'([(),.?!\'\"\-$&;|/*#{}@])\1*', r'\1', s)
    s = pattern.sub(r' \1 ', s)
    # 下面是对文本的预处理，就放在这个函数中
    s = re.sub(r'([(),.?!\'\"\-$&;|/*#]){2,}', r'\1', s)
    s = re.sub(r' +', ' ', s)
    return s


def text_processing_spanish(s, client):
    s = s.lower()

    # special = False
    # if s[0] == '?':
    #     s = s[1:]
    #     special = True

    result = client.word_tokenize(s)

    s = ' '.join(result)
    s = re.sub(r'[0-9]+', ' NUMs ', s)
    s = re.sub(r'[`|*;_´]', r' ', s)
    s = re.sub(r'？', r'?', s)
    s = re.sub(r'°', '.', s)

    s = re.sub(r'([(),.?!\'"\-$&;|/*#]){2,}', r'\1', s)

    s = re.sub(r"can′t", r"can not", s)
    s = re.sub(r'@', r' @ ', s)
    s = re.sub(r':', r' ', s)
    s = re.sub(r'-', r' - ', s)

    pattern = re.compile(r'([&;/-])')
    s = re.sub(r'([(),.?!\'\"\-$&;|/*#{}@])\1*', r'\1', s)
    s = pattern.sub(r' \1 ', s)
    # # 下面是对文本的预处理，就放在这个函数中
    s = re.sub(r' +', ' ', s)
    # if special:
    #     s = '? ' + s
    return s


def text_processing_english(text, client, remove_stopwords=None, stem_words=False):
    # Convert words to lower case and split them
    text = text.lower()



    # Optionally, remove stop words


    # text = " ".join(text)

    text = client.word_tokenize(text)
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)


    # Return a list of words
    if remove_stopwords:
        text = text.split(' ')
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    return text


def processing_data_1_step():
    """对数据集进行切分，先分成英文部分，西班牙文部分，不shuffle"""
    process_base_path = '../input/processing/'
    base_path = '../input/'

    if not os.path.exists(process_base_path):
        os.makedirs(process_base_path)

    train_en = base_path + 'cikm_english_train_20180516.txt'
    train_sp = base_path + 'cikm_spanish_train_20180516.txt'
    unlabel_data = base_path + 'cikm_unlabel_spanish_train_20180516.txt'

    english_file = process_base_path + 'english.txt'
    spanish_file = process_base_path + 'spanish.txt'
    unlabel_file = process_base_path + 'unlabel_spanish.txt'
    test_file = process_base_path + 'test_b_no_process.txt'

    # 将english_train中的英文和西班牙文划分开
    en_train = read_data(train_en)
    ens = []
    sps = []
    for line in en_train:
        line = line.strip()
        line_arr = line.split('\t')
        ens.append('{}\t{}\t{}\n'.format(line_arr[4], line_arr[0], line_arr[2]))
        sps.append('{}\t{}\t{}\n'.format(line_arr[4], line_arr[1], line_arr[3]))

    sp_train = read_data(train_sp)

    for line in sp_train:
        # line = punctiation(line)
        line = line.strip()
        line_arr = re.split('\t', line)
        sps.append('{}\t{}\t{}\n'.format(line_arr[4], line_arr[0], line_arr[2]))
        ens.append('{}\t{}\t{}\n'.format(line_arr[4], line_arr[1], line_arr[3]))

    # 讲分开的english和spanish文件保存下来
    save_data(english_file, data=ens)
    save_data(spanish_file, data=sps)

    print(u'对测试数据进行预处理，所有的label均设置为0')
    test_path = base_path + 'cikm_test_b_20180730.txt'
    test = read_data(test_path)
    sps = []
    for line in test:
        line = line.strip()
        line_arr = re.split('\t', line)
        sps.append('{}\t{}\t{}\n'.format(0, line_arr[0], line_arr[1]))
    save_data(test_file, sps)

    print('Done')


def processing_data_2_step():
    process_base_path = '../input/processing/'
    english_file = process_base_path + 'english.txt'
    spanish_file = process_base_path + 'spanish.txt'
    test_file = process_base_path + 'test_b_no_process.txt'

    english_core_nlp = StanfordCoreNLP(core_nlp_path, lang='en')

    with StanfordCoreNLP(core_nlp_path, lang='es') as client:


        english_processing_file = process_base_path + 'english_train.txt'
        spanish_processing_file = process_base_path + 'spanish_train.txt'
        test_processing_file = process_base_path + 'test_b.txt'
        #
        englishs = read_data(english_file)
        spanishs = read_data(spanish_file)

        english_processing = []
        spanish_processing = []

        # for english

        for line in tqdm(englishs):
            lines = line.strip().split('\t')
            assert len(lines) == 3
            lines[1] = text_processing_english(lines[1], english_core_nlp)
            lines[2] = text_processing_english(lines[2], english_core_nlp)

            english_processing.append("{}\t{}\t{}\n".format(lines[0], lines[1], lines[2]))

        save_data(english_processing_file, english_processing)

        # for spanish
        for line in tqdm(spanishs):
            line = line.strip().split('\t')
            assert len(line) == 3, print(line)
            line[1] = text_processing_spanish(line[1], client)
            line[2] = text_processing_spanish(line[2], client)

            spanish_processing.append("{}\t{}\t{}\n".format(line[0], line[1], line[2]))

        save_data(spanish_processing_file, spanish_processing)
        #
        # for test data
        test = read_data(test_file)
        test_processing = []
        for line in tqdm(test):
            # ipdb.set_trace()
            line = line.strip().split('\t')
            assert len(line) == 3
            line[2] = text_processing_spanish(line[2], client)
            line[1] = text_processing_spanish(line[1], client)


            test_processing.append("{}\t{}\t{}\n".format(line[0], line[1], line[2]))

        save_data(test_processing_file, test_processing)

    english_core_nlp.close()

    print('Done')


def build_spanish_vocab(min_freq=2):


    """
    将文件中所有的单词提取出来，用id进行表示
    ！包括test的单词
    """
    print('build spanish vocab')

    basepath = '../input/processing/'

    train_data = basepath + 'spanish_train.txt'
    test_data = basepath + 'test_b.txt'

    words = defaultdict(int)
    train1 = read_data(train_data)
    for line in train1:
        line_arr = line.split('\t')[1:]
        for seq in line_arr:
            seq = seq.strip()
            seq = re.split(r' +', seq)
            for w in seq:
                words[w] += 1

    test = read_data(test_data)
    for line in test:
        line_arr = line.split('\t')[1:]
        for seq in line_arr:
            seq = seq.strip()
            seq = re.split(r' +', seq)
            for w in seq:
                words[w] += 1

    print(len(words))
    # 降序排序
    words_sorted_count = sorted(words.items(), key=lambda x: -x[1])
    words_dict_list = [w[0] for w in words_sorted_count if w[1] >= min_freq]
    words_dict_list.insert(0, '<UNK>')
    words_dict_list.insert(0, '<PAD>')

    with open('../input/words.txt', 'wt', encoding='utf-8') as f:
        for word in words_dict_list:
            f.write(word + '\n')

    print('build spanish vocab done!')


def build_english_vocab(min_freq=3):
    print('build english word vocab')
    basepath = '../input/processing/'
    train_data = basepath + 'english_train.txt'

    words = defaultdict(int)
    train1 = read_data(train_data)
    for line in train1:
        line_arr = line.split('\t')[1:]
        for seq in line_arr:
            seq = seq.strip()
            seq = re.split(r' +', seq)
            for w in seq:
                words[w] += 1

    # 降序排序
    words_sorted_count = sorted(words.items(), key=lambda x: -x[1])
    words_dict_list = [w[0] for w in words_sorted_count if w[1] >= min_freq]

    with open('../input/english_word_vocabs.txt', 'wt', encoding='utf-8') as f:
        for word in words_dict_list:
            f.write(word + '\n')

    print('build spanish vocab done!')


def build_spanish_character_vocab():
    print('build spanish character vocab')
    train_path = '../input/processing/spanish_train_dedup.txt'
    test_path = '../input/processing/test.txt'

    vocabs = {}
    train = read_data(train_path)
    test = read_data(test_path)
    for data in [train, test]:
        for line in data:
            line = line.strip().split('\t')
            assert len(line) == 3, len(line)
            for c in line[1] + line[2]:
                if c in vocabs:
                    vocabs[c] += 1
                else:
                    vocabs[c] = 1

    # 写入到char_vocab.txt中
    char_vocabs = sorted(vocabs.items(), key=lambda x: x[1], reverse=True)
    fw = open('../input/char_vocab.txt', 'wt', encoding='utf-8')

    for (k, v) in char_vocabs:
        fw.write("{}\t{}\n".format(k,  v))

    fw.close()


def build_english_character_vocab():
    print('build english character vocab')
    train_path = '../input/processing/english_train.txt'

    vocabs = {}
    train = read_data(train_path)
    for data in [train]:
        for line in data:
            line = line.strip().split('\t')
            assert len(line) == 3, len(line)
            for c in line[1] + line[2]:
                if c in vocabs:
                    vocabs[c] += 1
                else:
                    vocabs[c] = 1

    # 写入到char_vocab.txt中
    char_vocabs = sorted(vocabs.items(), key=lambda x: x[1], reverse=True)
    fw = open('../input/char_english_vocab.txt', 'wt', encoding='utf-8')

    for (k, v) in char_vocabs:
        fw.write("{}\t{}\n".format(k,  v))

    fw.close()


def tenfold():
    """split train data for 10 fold"""
    base_path = '../input/processing/'

    train_file = base_path + 'train.txt'

    save_path = base_path + '10fold'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # split data for ten fold
    train = read_data(train_file)
    size = len(train)
    one_part_size = int(0.1 * size)
    random.shuffle(train)

    for i in range(10):
        save_file_path = "{}/train_{}.txt".format(save_path, i)
        if i < 9:
            save_data(save_file_path, train[i * one_part_size: (i + 1) * one_part_size])
        else:
            save_data(save_file_path, train[i * one_part_size:])

def five_fold():
    """split train data for 10 fold"""
    base_path = '../input/processing/'

    train_file = config.spanish_train_path

    save_path = base_path + '5fold'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # split data for ten fold
    train = read_data(train_file)

    size = len(train)
    one_part_size = int(0.2 * size)
    # random.shuffle(train)

    for i in range(5):
        save_file_path = "{}/train_{}.txt".format(save_path, i)
        if i < 4:
            save_data(save_file_path, train[i * one_part_size: (i + 1) * one_part_size])
        else:
            save_data(save_file_path, train[i * one_part_size:])


def split_train_valid(split_rate, is_use_real=False):
    train_file = config.spanish_train_path
    train_data_file = '../input/processing/train_data.txt'
    valid_data_file = '../input/processing/valid_data.txt'


    train = read_data(train_file)
    # random.shuffle(train)
    if is_use_real:
        valid_size = 1400

        train_data = train[: -valid_size]
        valid_data = train[-valid_size: ]


        random.shuffle(train_data)

        save_data(train_data_file, train_data)
        save_data(valid_data_file, valid_data)

    else:
        valid_size = int(split_rate * len(train))
        train_data = train[: -valid_size]
        valid_data = train[-valid_size:]



        # random.shuffle(train_data)

        save_data(train_data_file, train_data)
        save_data(valid_data_file, valid_data)


def de_duplicate():
    filename = '../input/processing/spanish_train.txt'
    data = read_data(filename)
    dumplicate = []
    new_data = []
    help_set = set()
    for line in data:
        new_line = line[1:]
        if new_line in help_set:
            dumplicate.append(line)
        else:
            new_data.append(line)
            help_set.add(new_line)

    save_data('../input/processing/dumplicate_spanish_data.txt', dumplicate,)
    save_data('../input/processing/spanish_train_dedup.txt', new_data)

def to_csv():
    train_path = "../input/processing/spanish_train_dedup.txt"
    test_path = "../input/processing/test_b.txt"
    columns = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
    train = read_data(train_path)
    test = read_data(test_path)

    all_question = set()
    for line in train + test:
        line = line.strip().split('\t')
        all_question.add(line[1])
        all_question.add(line[2])

    qids = dict(zip(all_question, list(range(1, len(all_question) + 1))))

    # write train.csv
    with open('../input/processing/train.csv', 'wt', encoding='utf-8') as f:
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(*columns))
        for i, line in enumerate(train):
            line = line.strip().split('\t')
            q1, q2, is_duplicate = line[1], line[2], line[0]
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i, qids[q1], qids[q2], q1, q2, is_duplicate))

    with open('../input/processing/test.csv', 'wt', encoding='utf-8') as f:
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(*columns))
        for i, line in enumerate(test):
            line = line.strip().split('\t')
            q1, q2, is_duplicate = line[1], line[2], line[0]
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i, qids[q1], qids[q2], q1, q2, is_duplicate))

def concat_english_spanish_vocab():
    english_vocab = read_data("../input/english_word_vocabs.txt")
    spanish_vocab = read_data("../input/words.txt")

    vocabs = spanish_vocab + english_vocab

    save_data(path='../input/processing/multi_task_learn/all_vocab.txt', data=vocabs)



if __name__ == '__main__':

    # 讲原始文本中的英文和西班牙文分开
    processing_data_1_step()
    # 使用corenlp对西班牙，英文进行分词处理，会耗时很久
    processing_data_2_step()
    # 生成西班牙文字典，
    de_duplicate()  # 去重操作
    to_csv()
    build_spanish_vocab(min_freq=3)
    build_spanish_character_vocab()
    # # 生成英文字典
    build_english_vocab(min_freq=3)
    build_english_character_vocab()
    #
    concat_english_spanish_vocab()
    #
    five_fold()



    # 划分数据集，划分比例为0.15
    # split_train_valid(split_rate=0.2)
