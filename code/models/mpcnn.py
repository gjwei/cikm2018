# coding: utf-8
# Author: gjwei

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class MPCNN(nn.Module):

    def __init__(self, config):
        super(MPCNN, self).__init__()

        self.config = config
        vocab_size = config.vocab_size
        n_word_dim = config.embed_size
        n_holistic_filters = config.n_holistic_filters
        n_per_dim_filters = config.n_per_dim_filters
        filter_widths = config.filter_widths
        hidden_layer_units = config.hidden_layer_units
        num_classes = config.num_classes
        dropout = config.dropout
        # ipdb.set_trace()
        self.embedding = nn.Embedding(vocab_size, n_word_dim, padding_idx=config.pad_index)
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_emb))
        self.embedding.weight.requires_grad = False

        self.pos_embedding = nn.Embedding(self.config.max_sentence_length + 1,
                                          n_word_dim,
                                          padding_idx=config.pad_index)
        self.pos_embedding.weight.data.copy_(position_encoding_init(self.config.max_sentence_length + 1,
                                                                    n_word_dim))

        torch.nn.init.uniform_(self.pos_embedding.weight, -0.02, 0.02)

        self.embed_drop = nn.Dropout(config.embed_dropout)

        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        holistic_conv_layers = []
        per_dim_conv_layers = []

        self.in_channels = n_word_dim

        for ws in filter_widths:
            if np.isinf(ws):
                continue

            holistic_conv_layers.append(nn.Sequential(
                nn.Conv1d(self.in_channels, n_holistic_filters, ws),
                nn.Tanh()
            ))

            per_dim_conv_layers.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * n_per_dim_filters, ws, groups=self.in_channels),
                nn.Tanh()
            ))

        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)
        self.per_dim_conv_layers = nn.ModuleList(per_dim_conv_layers)

        # compute number of inputs to first hidden layer
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + n_holistic_filters, 2 + self.in_channels, 2
        n_feat_h = 3 * len(self.filter_widths) * COMP_2_COMPONENTS
        n_feat_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 3 +
            # comparison units from per-dim conv
            2 * (len(self.filter_widths) - 1) * n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feat = n_feat_h + n_feat_v

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
        )

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {
                    'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'min': F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'mean': F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'min': F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'mean': F.avg_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

            per_dim_conv_out = self.per_dim_conv_layers[ws - 1](sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                batch_size = x1.size()[0]
                comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                comparison_feats.append(F.pairwise_distance(x1, x2).contiguous().view(batch_size, 1))
        # ipdb.set_trace()
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                batch_size = x1.size()[0]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                        comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                        comparison_feats.append(F.pairwise_distance(x1, x2).contiguous().view(batch_size, 1))
                        comparison_feats.append(torch.abs(x1 - x2))

        for pool in ('max', 'min'):
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, i]
                    x2 = oG_2B[:, :, i]
                    batch_size = x1.size()[0]
                    comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                    comparison_feats.append(F.pairwise_distance(x1, x2).contiguous().view(batch_size, 1))
                    comparison_feats.append(torch.abs(x1 - x2).view(batch_size, -1))

        return torch.cat(comparison_feats, dim=1)

    def get_pos(self, x):
        # input x shape is length, batch
        # ipdb.set_trace()
        result = torch.cat([torch.arange(1, self.config.max_sentence_length + 1).view(-1, 1) for _ in range(x.size(1))],dim=1)
        pad_index = x.eq(self.config.pad_index)
        result[pad_index] = self.config.pad_index
        return result.long().cuda()

    def forward(self, s1, s2):
        sent1 = self.embedding(s1).transpose(1, 2) + self.pos_embedding(self.get_pos(torch.transpose(s1, 0, 1))).permute(1, 2, 0)
        sent2 = self.embedding(s2).transpose(1, 2) + self.pos_embedding(self.get_pos(torch.transpose(s2, 0, 1))).permute(1, 2, 0)
        sent1 = self.embed_drop(sent1)
        sent2 = self.embed_drop(sent2)
        # ipdb.set_trace()
        # Sentence modeling module
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)
        # ipdb.set_trace()

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        combined_feats = [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds

    def optimizer_schedule(self, lr=1e-3, weight_decay=0.01):
        # ignore_parameters = list(map(id, self.embedding.parameters()))
        # update_parameters = filter(lambda p: id(p) not in ignore_parameters, self.parameters())
        update_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(update_parameters, weight_decay=weight_decay, lr=lr, amsgrad=True)
        return optimizer

