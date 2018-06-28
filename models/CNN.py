# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午4:48
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()

        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_sent_len = opt.max_sent_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.CLASS_SIZE = opt.label_size
        self.FILTERS = opt["FILTERS"]
        self.FILTER_NUM = opt["FILTER_NUM"]
        self.keep_dropout = opt.keep_dropout
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == "static" or self.embedding_type == "non-static" or self.embedding_type == "multichannel":
            self.WV_MATRIX = opt["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.embedding_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.embedding_dim * self.FILTERS[i],
                             stride=self.WORD_DIM)
            setattr(self, 'conv_%d' % i, conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.label_size)

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
        if self.embedding_type == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_sent_len - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x
