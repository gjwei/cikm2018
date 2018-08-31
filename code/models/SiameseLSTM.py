# coding: utf-8
# Author: gjwei
import torch
import torch.nn as nn
from models.utils.weight_drop import WeightDrop
from models.utils.locked_dropout import LockedDropout
from models.utils.embed_regularize import embedded_dropout

import ipdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


class SiameseLSTM(nn.Module):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """

    def __init__(self, config):
        super(SiameseLSTM, self).__init__()
        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size,
                                           config.embed_size,
                                           padding_idx=config.pad_index)

        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_emb))
        self.word_embedding.weight.requires_grad = False


        self.rnn = torch.nn.LSTM(config.embed_size,
                                 config.lstm_size, num_layers=config.num_layers,
                                 dropout=0, batch_first=True,
                                 bidirectional=True)

        self.dense = nn.Sequential(
                nn.BatchNorm1d(self.config.hidden_size * 2),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                nn.BatchNorm1d(self.config.hidden_size),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(self.config.hidden_size, self.config.num_classes)
        )

        self.dense.apply(weights_init)

        if config.wdrop:
            self.rnn = WeightDrop(self.rnn,
                                  ['weight_hh_l{}'.format(k) for k in range(config.num_layers)],
                                  dropout=config.wdrop)

        self.lockdrop = LockedDropout()
        self.rnn_batch_first = True



    def rnn_forward(self, sent, sent_lengths, sent_id=1):
        sorted_sent_lengths, indices = torch.sort(sent_lengths, descending=True)
        _, desort_indices = torch.sort(indices, descending=False)
        if self.rnn_batch_first:
            sent = sent[indices]
        else:
            sent = sent[:, indices]
        # ipdb.set_trace()
        sent = nn.utils.rnn.pack_padded_sequence(sent, sorted_sent_lengths.cpu().numpy(),
                                                 batch_first=self.rnn_batch_first)

        # self.rnn.flatten_parameters()
        output, _ = self.rnn(sent)  # output: batch_size, seq_len, hidden_size * num_direction

        padded_result, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.rnn_batch_first)

        if self.rnn_batch_first:
            desorted_result = padded_result[desort_indices]
        else:
            desorted_result = padded_result[:, desort_indices]

        idx = (sent_lengths - 1).view(-1, 1).expand(len(sent_lengths), desorted_result.size(2))
        time_dimention = 1 if self.rnn_batch_first else 0
        idx = idx.unsqueeze(time_dimention)
        last_output = desorted_result.gather(
            time_dimention, idx
        ).squeeze(time_dimention)
        return last_output

    def forward(self, sent1, sent2, sent1_lengths, sent2_lengths):
        embed1 = embedded_dropout(self.word_embedding, sent1, dropout=0.2 if self.training else 0)
        embed1 = self.lockdrop(embed1, self.config.dropouti)

        embed2 = embedded_dropout(self.word_embedding, sent2, dropout=0.2 if self.training else 0)
        embed2 = self.lockdrop(embed2, self.config.dropouti)

        rnn_result1 = self.rnn_forward(embed1, sent_lengths=sent1_lengths, sent_id=1)
        rnn_result2 = self.rnn_forward(embed2, sent_lengths=sent2_lengths, sent_id=2)
        # ipdb.set_trace()
        # concat = torch.cat([rnn_result1, rnn_result2], dim=-1)
        # output = self.dense(concat)
        # return output
        # ipdb.set_trace()
        distance = torch.sqrt(torch.sum(torch.pow(torch.add(rnn_result1, torch.neg(rnn_result2)), 2),
                                        dim=-1, keepdim=True))
        distance = torch.div(
            distance, torch.add(torch.sqrt(torch.sum(torch.pow(rnn_result1, 2), dim=1, keepdim=True)),
                                torch.sqrt(torch.sum(torch.pow(rnn_result2, 2), dim=1, keepdim=True)))
        )
        distance = distance.view([-1])
        return distance

    def optimizer_schedule(self, lr=1e-3, weight_decay=0):
        # ignore_parameters = list(map(id, self.embedding.parameters()))
        # update_parameters = filter(lambda p: id(p) not in ignore_parameters, self.parameters())
        update_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(update_parameters, weight_decay=weight_decay, lr=lr, amsgrad=True)
        return optimizer
