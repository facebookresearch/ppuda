# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Decoders to predict 1D-4D parameter tensors given node embeddings.

"""


import numpy as np
import torch.nn as nn
from .mlp import MLP
from .layers import get_activation


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_features=64,
                 hid=(128, 256),
                 out_shape=None,
                 num_classes=None):
        super(ConvDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(in_features,
                                          hid[0] * np.prod(out_shape[2:])),
                                nn.ReLU())

        conv = []
        for j, n_hid in enumerate(hid):
            n_out = np.prod(out_shape[:2]) if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1),
                         get_activation(None if j == len(hid) - 1 else 'relu')])

        self.conv = nn.Sequential(*conv)
        self.class_layer_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_shape[0], num_classes, 1))


    def forward(self, x, max_shape=(1,1), class_pred=False):

        N = x.shape[0]
        x = self.fc(x).view(N, -1, *self.out_shape[2:])  # N,128,11,11
        out_shape = self.out_shape
        if sum(max_shape) > 0:
            x = x[:, :, :max_shape[0], :max_shape[1]]
            out_shape = (out_shape[0], out_shape[1], max_shape[0], max_shape[1])

        x = self.conv(x).view(N, *out_shape)  # N, out, in, h, w

        if class_pred:
            x = self.class_layer_predictor(x[:, :, :, :, 0])  # N, num_classes, 64, 1
            x = x[:, :, :, 0]  # N, num_classes, 64

        return x



class MLPDecoder(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(64,),
                 out_shape=None,
                 num_classes=None):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.mlp = MLP(in_features=in_features,
                       hid=(*hid, np.prod(out_shape)),
                       activation='relu',
                       last_activation=None)
        self.class_layer_predictor = nn.Sequential(
            get_activation('relu'),
            nn.Linear(hid[0], num_classes * out_shape[0]))


    def forward(self, x, max_shape=(1,1), class_pred=False):
        if class_pred:
            x = list(self.mlp.fc.children())[0](x)  # shared first layer
            x = self.class_layer_predictor(x)  # N, 1000, 64, 1
            x = x.view(x.shape[0], self.num_classes, self.out_shape[1])
        else:
            x = self.mlp(x).view(-1, *self.out_shape)
            if sum(max_shape) > 0:
                x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x
