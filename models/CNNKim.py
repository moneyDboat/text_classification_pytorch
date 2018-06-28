# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午5:12
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KIMCNN1D(nn.Module):
    def __init__(self, opt):
        super(KIMCNN1D, self).__init__()

        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.kernel_sizes = opt.kernel_sizes
        self.kernel_nums = opt.kernel_nums
        self.keep_dropout = opt.keep_dropout
        self.in_channel = 1

        assert (len(self.kernel_sizes) == len(self.kernel_nums))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == "static" or self.embedding_type == "non-static" or self.embedding_type == "multichannel":
            self.embedding.weight = nn.Parameter(opt.embeddings)
            if self.embedding_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight = nn.Parameter(opt.embeddings)
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2
            else:
                pass

        self.convs = nn.ModuleList(
            [nn.Conv1d(self.in_channel, num, self.embedding_dim * size, stride=self.embedding_dim) for size, num in
             zip(opt.kernel_sizes, opt.kernel_nums)])
        self.fc = nn.Linear(sum(self.kernel_nums), self.label_size)

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_seq_len)
        if self.embedding_type == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_seq_len)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.convs[i](x)),
                         self.max_seq_len - self.kernel_sizes[i] + 1).view(-1, self.kernel_nums[i])
            for i in range(len(self.convs))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x