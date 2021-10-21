# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .layers import get_activation


class MLP(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(32, 32),
                 activation='relu',
                 last_activation='same'):
        super(MLP, self).__init__()

        assert len(hid) > 0, hid
        fc = []
        for j, n in enumerate(hid):
            fc.extend([nn.Linear(in_features if j == 0 else hid[j - 1], n),
                       get_activation(last_activation if
                                      (j == len(hid) - 1 and
                                       last_activation != 'same')
                                      else activation)])
        self.fc = nn.Sequential(*fc)


    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            x = x[0]
        return self.fc(x)
