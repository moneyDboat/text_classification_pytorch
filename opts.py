# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午1:16
@ide     : PyCharm  
"""

import argparse


def parse_opt():
    # model parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="lstm",
                        help='model name')

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden_dim')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                        help='grad_clip')

    parser.add_argument('--dataset', type=str, default="imdb",
                        help='dataset')
    parser.add_argument('--position', type=bool, default=False,
                        help='position')

    parser.add_argument('--keep_dropout', type=float, default=0.8,
                        help='keep_dropout')
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='max_epoch')
    parser.add_argument('--emb_train', type=str, default="false",
                        help='if embedding adjust in training')
    # kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                        help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                        help='kernel_nums')
    parser.add_argument('--lstm_mean', type=str, default="mean",  # last
                        help='lstm_mean')
    parser.add_argument('--lstm_layers', type=int, default=1,  # last
                        help='lstm_layers')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number')
    parser.add_argument('--debug', type=str, default="true",
                        help='if debug mode')

    args = parser.parse_args()

    if args.debug.lower() == "true":
        args.debug = True
    else:
        args.debug = False
    if args.emb_train.lower() == "true":
        args.emb_train = True
    else:
        args.emb_train = False

    return args
