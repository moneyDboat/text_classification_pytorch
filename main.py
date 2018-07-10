# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-6-28 下午1:34
@ide     : PyCharm  
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import models
import utils
import opts

opt = opts.parse_opt()
train_iter, test_iter = utils.load_data(opt)
opt.lstm_layers = 2

model = models.setup(opt)
if torch.cuda.is_available():
    model.cuda()
print("using model {}".format(opt.model))
model.train()
print("# parameters:", sum(param.numel() for param in model.parameters() if param.requires_grad))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
loss_funtion = F.cross_entropy

for i in range(opt.max_epoch):
    for epoch, batch in enumerate(train_iter):
        optimizer.zero_grad()
        start = time.time()
        text = batch.text[0]
        predicted = model(text)

        loss = loss_funtion(predicted, batch.label)
        loss.backward()
        # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if epoch % 100 == 0:
            if torch.cuda.is_available():
                loss_val = loss.cpu().data.numpy()
            else:
                loss_val = loss.data.numpy()
            print("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (
                i, epoch, loss_val, time.time() - start))

    accuracy = utils.evaluation(model, test_iter)
    print("%d iteration with accuracy %.7f" % (i, accuracy))
