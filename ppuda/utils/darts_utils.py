# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Some functionality in this script is based on the DARTS code: https://github.com/quark0/darts (Apache License 2.0)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AvgrageMeter:

    def __init__(self, dispersion_measure=None):
        self.dispersion_measure = dispersion_measure
        assert dispersion_measure in [None, 'std', 'se'], (
            'must be None, standard deviation (std) or standard error (se) of the mean')
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        if self.dispersion_measure is not None:
            self.values = []
            self.dispersion = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = np.sum(self.sum) / self.cnt
        if self.dispersion_measure is not None:
            self.values += [val] * n
            assert abs(self.avg - np.mean(self.values)) < 1e-3, (self.avg, np.mean(self.values))  # sanity check
            assert len(self.values) == self.cnt, (len(self.values), self.cnt)  # sanity check
            sd = np.std(self.values)
            self.dispersion = sd if self.dispersion_measure == 'std' else sd / np.sqrt(self.cnt)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def load_DARTS_pretrained(net, checkpoint='./checkpoints/imagenet_model.pt', device='cpu', load_class_layers=True):
    state_dict = torch.load(checkpoint, map_location=device)['state_dict']
    state_dict_new = {}
    for name, p in state_dict.items():
        if name.startswith('classifier.'):
            if load_class_layers:
                state_dict_new[name.replace('classifier.', 'classifier.0.')] = p
        elif name.startswith('auxiliary'):
            continue
        else:
            state_dict_new[name] = p

    res = net.load_state_dict(state_dict_new, strict=False)
    return net, res
