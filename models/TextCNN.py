# -*- coding: utf-8 -*-

"""
@Author  : Captain
@time    : 2018/11/21 13:57
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import numpy as np
from config import DefaultConfig

kernal_sizes = [1, 2, 3, 4, 5]


class TextCNN(nn.Module):
    def __init__(self, config, vectors=None):
        super(TextCNN, self).__init__()

        '''Embedding Layer'''
        # 使用预训练的词向量
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=config.embed_dim,
                          out_channels=config.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(config.kernel_num),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=config.kernel_num,
                          out_channels=config.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(config.kernel_num),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(config.max_seq_len - kernel_size * 2 + 2))
            )
            for kernel_size in kernal_sizes
        ]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(5 * config.kernel_num, config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.linear_hidden_size, config.label_size)
        )

    def forward(self, inputs):
        embeds = self.embedding(inputs)  # seq * batch * embed
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        conv_out = [conv(embeds.permute(1, 2, 0)) for conv in self.convs]
        conv_out = torch.cat(conv_out, dim=1)

        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits

    def get_optimizer(self, lr1, lr2):
        embedding_params = map(id, self.embedding.parameters())
        base_params = filter(lambda p: id(p) not in embedding_params, self.parameters())
        optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': base_params, 'lr': lr1}
        ])

        return optimizer
