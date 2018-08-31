# coding: utf-8
from utils.data_utils import load_vocab, load_char_vocab
import numpy as np
import pandas as pd
import random


class Config():

    multi_task_path  ="../input/processing/multi_task_learn/"

    vocab_path = '../input/words.txt'
    vocab_char_path = '../input/char_vocab.txt'
    embed_path = './fasttext/embedding_matrix.npz'
    fasttext_file_path = './fasttext/wiki.es.vec'
    fasttext_file_english_path = './fasttext/wiki.en.vec'

    all_vocab_path = "{}{}".format(multi_task_path, 'all_vocab.txt')
    all_vocab_embedding_file = "{}{}".format(multi_task_path, 'embedding_matrix.npz')
    all_char_vocab_path = "{}{}".format(multi_task_path, 'all_char_vocab.txt')

    all_char_vocab = load_char_vocab(all_char_vocab_path)


    word_vocabs = load_vocab(vocab_path)
    char_vocabs = load_char_vocab(vocab_char_path)

    PAD_WORD = '<PAD>'
    UNK_WORD = '<UNK>'


    vocab_size = len(word_vocabs)

    # character parameters
    use_char_emb = True

    char_vocab_size = len(char_vocabs)
    # char_dim = 50
    char_hidden_size = 64  # for char lstm
    max_word_length = 10
    CHAR_PAD = '@'


    # data path
    input_path = '../input/processing/'

    train_path = input_path + 'train_data.txt'
    english_train_path = input_path + 'english_train.txt'
    spanish_train_path = input_path + 'spanish_train_dedup.txt'

    valid_path = input_path + 'valid_data.txt'
    test_path = input_path + 'test_b.txt'
    model_save_path = './checkpoints/'
    num_workers = 1
    pad_index = 0
    save_model = True
    restore = False  # for restore training

    eval_every = 100

    embed_size = 300

    # hidden_size = random.randint(256, 512)
    hidden_size = 256
    lstm_size = 256
    fc_hidden_size = 512
    linear_size = fc_hidden_size
    num_classes = 2

    max_lengths = random.choice([20, 25])
    batch_size = 64


    lr = random.choice([1e-3])
    # lr = 1e-3
    start_lr = lr
    lr_decay_step = 5
    epochs = 60

    dropout = 0.5
    rnn_dropout = 0.2

    clip = 5.0

    rnn_layers = 3

    batch_not_imporved_throld = 20

    five_fold_path = '../input/processing/5fold/'
    five_fold_save_path = './result/'

    # CNN character parameters
    import random
    user_char_emb = True
    char_dim = 50
    char_kernel_sizes = random.choices([1, 2, 3, 4, 5], k=3)
    char_kernel_nums = random.choices([64, 64, 128, 128], k=3)
    char_output_dim = 128

    embed_dropout = random.uniform(0.1, 0.3)

    pretrained_emb = np.load(embed_path)['weights']


    # for multi task learning
    # multi_task = random.choice([False, True])
    multi_task = True
    multi_task_vocabs = load_vocab(all_vocab_path)
    multi_task_pretraind_emb = np.load(all_vocab_embedding_file)['weights']

    if multi_task:
        char_vocabs = multi_task_vocabs
        pretrained_emb = multi_task_pretraind_emb
        vocab_size = len(multi_task_vocabs)
        char_vocab_size = len(all_char_vocab)
        print('vocab size is ', vocab_size)

    onehot = False

    # early stop

    # ESIM model
    # num_units = 300 + char_`output_dim + pos_embedding_size
    num_units = 300
    char_num_units = num_units + char_output_dim
    project_input = True  # whether to project input embeddings to  different dimensionality
    distance_biases = random.randint(15,  30)  # number of different distances with biases used in the intra-attention model
    max_sentence_length = max_lengths

    # StackBiLSTMMaxount(SSE)
    h_size = [512, 1024, 2048]
    mlp_d = 1600

    # Decoposable Attention model

    # BIMPM
    num_perspective = 50
    word_dim = embed_size
    word_vocab_size = vocab_size
    max_word_len = max_word_length


    # extra features
    use_extra = False
    extra_path = "../input/processing/with_extra_features/"
    train_data_extra = extra_path + 'train_data.csv'
    valid_data_extra = extra_path + 'valid_data.csv'
    test_data_extra = extra_path + 'test.csv'
    if use_extra:
        extra_feature_nums = len(pd.read_csv(train_data_extra, sep='\t').columns) - 3
    else:
        extra_feature_nums = 0




    word_max_lengths = max_word_length
    CHAR_PAD_INDEX = 0
    CHAR_PAD = ' '

    # simaese lstm
    residual = True
    num_layers = 2
    wdrop = 0.25
    dropouti = 0.25

    # MPCNN
    filter_widths =  [1, 2, 3, np.inf]
    hidden_layer_units = 512
    n_holistic_filters = 300
    n_per_dim_filters = 32



    msg = ('parameters is hidden_size: {}, max_lengths: {} batch_size {} lr: {}, dropout: {}, use_english: {}'.format(
        hidden_size, max_lengths, batch_size, lr, dropout, multi_task))
    print(msg)


    
    
config = Config()
