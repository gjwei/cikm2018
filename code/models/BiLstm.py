# coding: utf-8

import numpy as np
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.rnn_layers
        self.embed_size = config.embed_size
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.lstm = nn.LSTM(config.embed_size, hidden_size=config.lstm_size,
                            num_layers=config.rnn_layers, bidirectional=True,
                            batch_first=True)

        self.fc = nn.Linear(config.lstm_size * 8, 1)

        self.load_embedding()
        self.init_weights()
    
    def init_hidden(self, size):
        h0 = nn.Parameter(torch.zeros(size).cuda(), requires_grad=True)
        return h0
    
    def load_embedding(self):
        self.embedding.weight.requires_grad = False
        self.embedding.weight.data.copy_(torch.from_numpy(np.load(self.config.embed_path)['weights']))
        print('embedding weight load done')
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        nn.init.xavier_normal_(self.fc.weight)
        print('weights init done')
    
    def forward(self, s1, s2):
        
        # ipdb.set_trace()
        
        batch_size = s1.size(0)
        h1 = self.init_hidden((self.num_layers * 2, batch_size, self.hidden_size))
        h2 = self.init_hidden((self.num_layers * 2, batch_size, self.hidden_size))
        c1 = self.init_hidden((self.num_layers * 2, batch_size, self.hidden_size))
        c2 = self.init_hidden((self.num_layers * 2, batch_size, self.hidden_size))
        
        o1 = self.embedding(s1)
        out1, _ = self.lstm(o1, (h1, c1))
        out1_max = torch.max(out1, dim=1)[0]
        out1_mean = torch.mean(out1, dim=1)
        
        out2 = self.embedding(s2)
        out2, _ = self.lstm(out2, (h2, c2))
        out2_max = torch.max(out2, dim=1)[0]
        out2_mean = torch.mean(out2, dim=1)
        # ipdb.set_trace()

        out = torch.cat((out1_max, out1_mean, out2_max, out2_mean), dim=1)
        
        out = self.fc(out)
        return out
    
    def optimizer_schedule(self, lr=1e-3, weight_decay=0):
        # ignore_parameters = list(map(id, self.embedding.parameters()))
        # update_parameters = filter(lambda p: id(p) not in ignore_parameters, self.parameters())
        update_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(update_parameters, weight_decay=weight_decay, lr=lr, amsgrad=True)
        return optimizer


if __name__ == '__main__':
    from code.config import config
    model = BiLSTM(config=config)
    
    print(model.__class__.__name__)
