# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Decoders to predict 1D-4D parameter tensors given node embeddings.

"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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

        ch_prod = out_shape[0] * out_shape[1]
        spatial_prod = out_shape[2] * out_shape[3]
        out_features = hid[0] * spatial_prod

        self.fc = nn.Sequential(nn.Linear(in_features, out_features),
                                get_activation('relu'))
        self.register_buffer('cols_1d', torch.arange(0, out_features).view(hid[0], out_shape[2], out_shape[3]),
                             persistent=False)
        self.register_buffer('cols_4d', torch.arange(0, ch_prod),
                             persistent=False)

        conv = []
        for j, n_hid in enumerate(hid):
            n_out = ch_prod if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1),
                         get_activation(None if j == len(hid) - 1 else 'relu')])

        self.conv = nn.Sequential(*conv)
        self.class_layer_predictor = nn.Sequential(
            get_activation('relu'),
            nn.Conv2d(out_shape[0], num_classes, 1))


    def forward(self, x, max_shape=(1,1,1,1), class_pred=False):

        N = x.shape[0]
        if sum(max_shape[2:]) < sum(self.out_shape[2:]) and len(self.fc) == 2:
            ind = self.cols_1d[:, :max_shape[2], :max_shape[3]].flatten()
            x = self.fc[1](F.linear(x, self.fc[0].weight[ind],
                                    self.fc[0].bias[ind]).view(N, -1, max_shape[2], max_shape[3]))  # N,128,a,b
        else:
            x = self.fc(x).view(N, -1, *self.out_shape[2:])[:, :, :max_shape[2], :max_shape[3]]  # N,128,11,11

        out_shape = (*self.out_shape[:2], min(self.out_shape[2], max_shape[2]), min(self.out_shape[3], max_shape[3]))

        if (max_shape[1] <= out_shape[1] // 2 or (max_shape[0] < out_shape[0] and not class_pred)) and len(self.conv) == 4:
            x = self.conv[1](self.conv[0](x)) # N, 256, h, w
            if max_shape[1] < out_shape[1] and max_shape[1] % 3 == 0:
                n_in = max_shape[1] // 3 * 4
            else:
                n_in = min(max_shape[1], out_shape[1])

            n_out = out_shape[0] if class_pred else min(out_shape[0], max_shape[0])
            ind = self.cols_4d[:n_out * out_shape[1]]
            if n_in < out_shape[1]:
                ind = ind.reshape(-1, n_in)[::out_shape[1] // n_in].flatten()

            x = self.conv[3](F.conv2d(x, self.conv[2].weight[ind], self.conv[2].bias[ind]))

            x = x.reshape(N, n_out, n_in, *out_shape[2:])[:, :, :min(out_shape[1], max_shape[1])]
            if min(max_shape[2:]) > min(out_shape[2:]):
                x = x.repeat((1, 1, 1, 2, 2))
        else:
            x = self.conv(x).view(N, out_shape[0], -1, *out_shape[2:])  # N, out, in, h, w

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


    def forward(self, x, max_shape=(1,1,1,1), class_pred=False):
        if class_pred:
            x = list(self.mlp.fc.children())[0](x)  # shared first layer
            x = self.class_layer_predictor(x)  # N, 1000, 64, 1
            x = x.view(x.shape[0], self.num_classes, self.out_shape[1])
        else:
            x = self.mlp(x).view(-1, *self.out_shape)
            if sum(max_shape[2:]) > 0:
                x = x[:, :, :, :max_shape[2], :max_shape[3]]
        return x
