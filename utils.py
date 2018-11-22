# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午12:53
@ide     : PyCharm
"""

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import numpy as np
import visdom


def load_data(config):
    # use torchtext to load data, no need to download dataset
    print("loading {} dataset".format(config.dataset))

    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, fix_length=config.max_seq_len)
    LABEL = data.Field(sequential=False)
    if config.dataset == "imdb":
        train, test = datasets.IMDB.splits(TEXT, LABEL)
    elif config.dataset == "sst":
        train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=True, train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
    elif config.dataset == "trec":
        train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
    else:
        print("The dataset is not supported.")

    # build the vocabulary
    TEXT.build_vocab(train, test, vectors=GloVe(name=config.embed_name, dim=config.embed_dim))
    LABEL.build_vocab(train, test)

    # print vocab information
    print('train nums: {}'.format(len(train)))
    print('test nums: {}'.format(len(test)))
    print('TEXT.vocab nums: {}'.format(len(TEXT.vocab)))
    print('LABEL.vocab nums: {}'.format(len(LABEL.vocab)))

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=config.batch_size, repeat=False,
                                                       shuffle=True)

    config.label_size = len(LABEL.vocab)
    config.vocab_size = len(TEXT.vocab)

    return train_iter, test_iter, TEXT.vocab.vectors


class Visualizer():
    '''
    封装了visdom的基本操作，但是可以通过self.vis.function调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 保存('loss', 23)，即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y):
        # self.plot('loss', 1.00)

        x = self.index.get(name, 0)
        self.vis.line(X=np.array([x]), Y=np.array([y]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append')
        self.index[name] = x + 1

    def log(self, info, win='log_txt'):
        # self.log({'loss':1, 'lr':0.0001})
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, item):
        return getattr(self.vis, item)
