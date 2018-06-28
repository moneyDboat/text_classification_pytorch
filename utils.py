# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午12:53
@ide     : PyCharm
"""

import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import numpy as np


def load_data(opt):
    # use torchtext to load data, no need to download dataset
    device = 0 if torch.cuda.is_available() else -1
    # set up fields
    text = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=opt.max_seq_len)
    label = data.Field(sequential=False)

    # make splits for data
    if opt.dataset == "imdb":
        train, test = datasets.IMDB.splits(text, label)
    elif opt.dataset == "sst":
        train, val, test = datasets.SST.splits(text, label, fine_grained=True, train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
    elif opt.dataset == "trec":
        train, test = datasets.TREC.splits(text, label, fine_grained=True)
    else:
        print("The dataset is not supported!")

    # build the vocabulary
    text.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    label.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(text.vocab))
    print('TEXT.vocab.vectors.size()', text.vocab.vectors.size())

    # make iterator for splits
    # train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size, device=device,
    #                                                    repeat=False, shuffle=True)
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size, device=device)

    opt.label_size = len(label.vocab)
    opt.vocab_size = len(text.vocab)
    opt.embedding_dim = text.vocab.vectors.size()[1]
    opt.embeddings = text.vocab.vectors

    return train_iter, test_iter


def evaluation(model, test_iter):
    model.eval()
    accuracy = []
    for index, batch in enumerate(test_iter):
        text = batch.text[0]
        predicted = model(text)
        prob, idx = torch.max(predicted, 1)
        acc = (idx == batch.label).float().mean()

        if torch.cuda.is_available():
            accuracy.append(acc.data.cpu().numpy())
        else:
            accuracy.append(acc.data.numpy())
    model.train()
    return np.mean(accuracy)
