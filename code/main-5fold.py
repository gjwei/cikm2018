# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import datetime

from data.CikmDataset import CikmDataset_n_fold
from models.StackedBiLSTMMaxout import StackBiLSTMMaxout
from models.mpcnn import MPCNN
from models.bimpm import BIMPM
from models.ESIM import ESIM, ESIM_WithCharEmbedding
from models.SiameseLSTM import SiameseLSTM
from utils.utils import get_mask


def forward(model, s1, s2, s1_mask, s2_mask, s1_chars, s2_chars, s1_len, s2_len, extra):
    # 不同的模型输入可能会不同
    if isinstance(model, ESIM):
        logits = model(s1, s1_mask, s2, s2_mask)

    elif isinstance(model, StackBiLSTMMaxout):
        logits = model(s1, s1_len, s2, s2_len)
    elif isinstance(model, BIMPM):
        logits = model(s1, s2, s1_chars, s2_chars)
    elif isinstance(model, ESIM_WithCharEmbedding):
        logits = model(s1, s1_mask, s2, s2_mask, s1_chars, s2_chars)
    elif isinstance(model, SiameseLSTM):
        logits = model(s1, s2, s1_len, s2_len)
    elif isinstance(model, MPCNN):
        logits = model(s1, s2)

    return logits



def train(model_class):
    """
    train a model of ten fold

    """
    from config import Config

    config = Config()

    test_predicts = None

    # Let's begin
    reports = []
    eval_losses = []

    for fold_index in range(5):
        # reset model
        print('set learning rate to {}'.format(Config().lr))
        config.lr = Config().lr

        model = model_class(config)

        # mark for train model phrase
        model.train()
        restore_file = config.model_save_path + "{}/best_model.pt".format(model.__class__.__name__)
        restore_path = config.model_save_path + model.__class__.__name__
        if not os.path.exists(restore_path):
            os.system('mkdir -p {}'.format(restore_path))

        checkpoint_id = 1

        if config.restore and os.path.exists(restore_file):
            print('restore parameter from {}'.format(restore_file))
            model_file = torch.load(restore_file)
            model.load_sate_dict(model_file['model'], strict=False)
            checkpoint_id = int(model_file['checkpoint_id']) + 1
            min_loss = float(model_file['loss'])
            lr2 = config.lr

        fw = open('{}/log.txt'.format(restore_path), 'a', encoding='utf-8')

        fw.write(config.msg + '\n')
        model.cuda()


        min_loss = float('inf')

        print('fold:{} / {}'.format(fold_index + 1, 5))
        # ipdb.set_trace()
        train_paths = [config.five_fold_path + 'train_{}.txt'.format(i) for i in range(5) if i != fold_index]
        valid_paths = [config.five_fold_path + 'train_{}.txt'.format(fold_index)]
        if config.multi_task:
            train_paths.append(config.english_train_path)

        print('load dataset')

        dataset = CikmDataset_n_fold(paths=train_paths, max_length=config.max_lengths, one_hot=config.onehot)
        dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=4, drop_last=False)
        print('training data size is {}'.format(len(dataset)))
        optimizer = model.optimizer_schedule(config.start_lr)

        loss_function = nn.CrossEntropyLoss(size_average=True)

        batch = 0

        batch_not_imporve = 0
        loss = 0.0

        # begin training
        for epoch in range(1, config.epochs + 1):
            print('epoch:{} / {}'.format(epoch, config.epochs))
            # 更新新的learning rate
            if epoch % config.lr_decay_step == 0:
                i_decay = epoch // config.lr_decay_step

                config.lr = config.start_lr / (2 ** i_decay)
                config.lr = max(3e-5, config.lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = config.lr

            for i, (s1, s2, label, s1_len, s2_len, s1_chars, s2_chars) in enumerate(dataloader):
                s1_mask = get_mask(s1_len, max_length=config.max_lengths)
                s2_mask = get_mask(s2_len, max_length=config.max_lengths)

                s1, s2, label = Variable(s1).long().cuda(), Variable(s2).long().cuda(), Variable(label).long().cuda()
                s1_len, s2_len = Variable(s1_len).long().cuda(), Variable(s2_len).long().cuda()
                s1_chars, s2_chars = Variable(s1_chars).long().cuda(), Variable(s2_chars).long().cuda()
                s1_mask = Variable(torch.from_numpy(s1_mask)).float().cuda()
                s2_mask = Variable(torch.from_numpy(s2_mask)).float().cuda()

                logits = forward(model, s1, s2, s1_mask, s2_mask, s1_chars, s2_chars, s1_len, s2_len, extra=None)

                optimizer.zero_grad()
                batch_loss = loss_function(logits, label)
                batch_loss.backward()
                # clip grad
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), config.clip)

                optimizer.step()

                batch += 1

                # loss += (batch_loss.item() + tmp) / 2
                loss += batch_loss.item()
                # 每隔iternal个batch计算下平均loss情况
                if batch ==  config.eval_every:
                    epoch_loss, checkpoint_id = eval(model, valid_paths, min_loss, checkpoint_id, config)

                    # count batch not improve
                    if epoch_loss > min_loss:
                        batch_not_imporve += 1
                    else:
                        batch_not_imporve = 0

                    if batch_not_imporve > config.batch_not_imporved_throld:
                        print('early stop')
                        break

                    min_loss  = min(min_loss, epoch_loss)
                    msg = '{} fold:{:>2} epoch:{:>2} train_loss:{:,.6f} test_loss:{:,.6f} minloss:{:,.6f} lr:{:,.6f} \n'.format(
                        str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),fold_index, epoch, loss / config.eval_every,
                        epoch_loss, min_loss, config.lr)
                    loss = 0
                    batch = 0
                    print(msg)
                    fw.write(msg)
                    fw.flush()

            if batch_not_imporve == 7:
                config.lr /= 2

            if batch_not_imporve > config.batch_not_imporved_throld:
                print('early stop')
                break

        reports.append('fold:{} min valid loss is{:.4f}'.format(fold_index, min_loss))
        eval_losses.append(min_loss)

        # after training, we should generate result
        print("train phrase done, now let't test result")
        predict = test(model, config)

        predict = np.asarray(predict, dtype=np.float32)
        if test_predicts is None:
            test_predicts = predict
        else:
            test_predicts += predict

        model.train()
    
    fw.close()

    for  i in range(len(reports)):
        print(reports[i])

    # save 10 fold result
    test_predicts /= 5
    average_loss = sum(eval_losses) / 5.0
    print('average loss is {}'.format(average_loss))

    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    with open("{}_{}{}_5fold_min_loss{:.4f}.txt".format(config.five_fold_save_path,
                                                        model.__class__.__name__,
                                                        cur_time,
                                                        average_loss),
              'wt', encoding='utf-8') as f:
        for line in test_predicts:
            f.write(str(line) + '\n')



def eval(model, valid_paths, min_loss, checkpoint_id, config):

    """计算验证集分数"""
    dataset = CikmDataset_n_fold(paths=valid_paths, max_length=config.max_lengths, one_hot=config.onehot)
    model.eval()

    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    loss = 0.0
    step = 0
    loss_function = nn.CrossEntropyLoss(size_average=True)

    for i, (s1, s2, labels, s1_len, s2_len, s1_chars, s2_chars) in enumerate(dataloader):
        s1_mask = get_mask(s1_len, max_length=config.max_lengths)
        s2_mask = get_mask(s2_len, max_length=config.max_lengths)

        s1, s2, labels = Variable(s1).long().cuda(), Variable(s2).long().cuda(), Variable(labels).long().cuda()
        s1_len, s2_len = Variable(s1_len).long().cuda(), Variable(s2_len).long().cuda()
        s1_chars, s2_chars = Variable(s1_chars).long().cuda(), Variable(s2_chars).long().cuda()
        s1_mask = Variable(torch.from_numpy(s1_mask)).float().cuda()
        s2_mask = Variable(torch.from_numpy(s2_mask)).float().cuda()

        # labels = labels.view(labels.size(0), -1)

        # 不同的模型输入可能会不同
        logits = forward(model, s1, s2, s1_mask, s2_mask, s1_chars, s2_chars, s1_len, s2_len, extra=None)

        batch_loss = loss_function(logits, labels)
        loss += batch_loss.item()

        step += 1

    loss = loss/ step

    if config.save_model and loss < min_loss:
        print('current validation loss:{:,.6f} is less min_loss:{:,.6f} save model!'.format(loss, min_loss))
        torch.save(
            {'model': model.state_dict(),
             'checkpoint_id': checkpoint_id,
             'loss': loss},
            '{}{}/best_model.pt'.format(config.model_save_path, model.__class__.__name__)
        )

    model.train()
    return loss, checkpoint_id + 1



def test(model, config):

    model_file_path = config.model_save_path + model.__class__.__name__ + '/best_model.pt'
    print('加载模型{}'.format(model_file_path))
    model_file = torch.load(model_file_path)
    # ipdb.set_trace()
    model.load_state_dict(model_file['model'], strict=False)
    model.cuda()

    model.eval()
    dataset = CikmDataset_n_fold(paths=[config.test_path], max_length=config.max_lengths, one_hot=config.onehot)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=4, drop_last=False)
    result = []

    for i, (s1, s2, label, s1_len, s2_len, s1_chars, s2_chars) in enumerate(dataloader):
        s1_mask = get_mask(s1_len, max_length=config.max_lengths)
        s2_mask = get_mask(s2_len, max_length=config.max_lengths)

        s1, s2, label = Variable(s1).long().cuda(), Variable(s2).long().cuda(), Variable(label).float().cuda()
        s1_len, s2_len = Variable(s1_len).long().cuda(), Variable(s2_len).long().cuda()
        s1_chars, s2_chars = Variable(s1_chars).long().cuda(), Variable(s2_chars).long().cuda()

        s1_mask = Variable(torch.from_numpy(s1_mask)).float().cuda()
        s2_mask = Variable(torch.from_numpy(s2_mask)).float().cuda()

        # label = label.view(label.size(0), -1)

        # ipdb.set_trace()
        logits = forward(model, s1, s2, s1_mask, s2_mask, s1_chars, s2_chars, s1_len, s2_len, extra=None)

        # predict = model(s1, s1_len, s2, s2_len)
        predict = torch.nn.functional.softmax(logits, dim=-1)
        
        result += list(predict.squeeze().cpu().data.numpy()[:, 1])

    return result
            
            
if __name__ == '__main__':

    model_class = ESIM
    # model_class = MPCNN
    """
    以下四种模型收敛效果不如前两种
    # model_class = ESIM_WithCharEmbedding  
    # model_class = StackBiLSTMMaxout
    # model_class = BIMPM
    model_class = SiameseLSTM
    """

    print(model_class)

    train(model_class)