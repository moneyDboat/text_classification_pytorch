# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午1:34
@ide     : PyCharm  
"""

import torch
import torch.nn.functional as F
import time
import models
import utils
from config import DefaultConfig
import fire
import random
from utils import Visualizer
import numpy as np


def main(**kwargs):
    config = DefaultConfig()
    config.parse(kwargs)
    config.env = str(config.id)
    vis = Visualizer

    # set random seed
    # cpu and gpu both need to set
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not torch.cuda.is_available():
        config.cuda = False
        config.device = None

    train_iter, test_iter, emb_vectors = utils.load_data(config)
    config.print_config()

    model = getattr(models, config.model)(config, emb_vectors)
    print(model)

    if config.cuda:
        torch.cuda.set_device(config.device)
        model.cuda()

    # 目标函数和优化器
    loss_f = F.cross_entropy
    lr1, lr2 = config.lr1, config.lr2
    optimizer = model.get_optimizer(lr1, lr2)

    model.train()
    for epoch in range(config.max_epochs):
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_i, batch in enumerate(train_iter):
            text, label = batch.text[0], batch.label
            if config.cuda:
                text, label = text.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = model(text)
            loss = loss_f(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = pred.max(dim=1)[1]
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if (batch_i + 1) % (10000 // config.batch_size) == 0:
                # 10000条训练数据输出一次统计指标
                print('[Epoch {}] loss: {:.5f} | Acc: {:.3f}%({}/{})'.format(epoch + 1, total_loss,
                                                                             100.0 * correct / total, correct, total))

        train_acc, train_acc_n, train_n = val(model, train_iter, config)
        print('Epoch {} time spends : {:.1f}s'.format(epoch + 1, time.time() - start_time))
        print('Epoch {} Train Acc: {:.2f}%({}/{})'.format(epoch + 1, train_acc, train_acc_n, train_n))
        test_acc, test_acc_n, test_n = val(model, test_iter, config)
        print('Epoch {} Test Acc: {:.2f}%({}/{})\n'.format(epoch + 1, test_acc, test_acc_n, test_n))


def val(model, data_iter, config):
    model.eval()

    acc_n = 0
    n = 0

    with torch.no_grad():
        for batch in data_iter:
            text, label = batch.text[0], batch.label
            if config.cuda:
                text, label = text.cuda(), label.cuda()

            predicted = model(text)
            pred_i = predicted.max(dim=1)[1]
            acc_n += (pred_i == label).sum().item()
            n += label.size(0)

    model.train()
    acc = 100. * acc_n / n
    return acc, acc_n, n


if __name__ == '__main__':
    fire.Fire()
