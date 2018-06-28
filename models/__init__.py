# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午1:39
@ide     : PyCharm  
"""

from .LSTM import LSTMBI


def setup(opt):
    if opt.model == 'lstm':
        model = LSTMBI(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
