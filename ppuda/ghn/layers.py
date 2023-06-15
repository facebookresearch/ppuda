# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper layers to build GHNs.

"""


import torch
import torch.nn as nn
import numpy as np
import copy


def get_activation(activation):
    if activation is not None:
        if activation == 'relu':
            f = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            f = nn.LeakyReLU()
        elif activation == 'selu':
            f = nn.SELU()
        elif activation == 'elu':
            f = nn.ELU()
        elif activation == 'rrelu':
            f = nn.RReLU()
        elif activation == 'sigmoid':
            f = nn.Sigmoid()
        else:
            raise NotImplementedError(activation)
    else:
        f = nn.Identity()

    return f


class ShapeEncoder(nn.Module):
    def __init__(self, hid, num_classes, max_shape, debug_level=0):
        super(ShapeEncoder, self).__init__()

        assert max_shape[2] == max_shape[3], max_shape
        self.debug_level = debug_level
        self.num_classes = num_classes
        self.ch_steps = (2**3, 2**6, 2**12, 2**13)
        self.channels = np.unique([1, 3, num_classes] +
                                  list(range(self.ch_steps[0], self.ch_steps[1], 2**3)) +
                                  list(range(self.ch_steps[1], self.ch_steps[2], 2**4)) +
                                  list(range(self.ch_steps[2], self.ch_steps[3] + 1, 2**5)))

        self.spatial = np.unique(list(range(1, max(12, max_shape[3]), 2)) + [14, 16])

        # create a look up dictionary for faster determining the channel shape index
        # include shapes not seen during training by assigning them the the closest seen values
        self.channels_lookup = {c: i for i, c in enumerate(self.channels)}
        self.channels_lookup_training = copy.deepcopy(self.channels_lookup)
        for c in range(4, self.ch_steps[0]):
            self.channels_lookup[c] = self.channels_lookup[self.ch_steps[0]]  # 4-7 channels will be treated as 8 channels
        for c in range(1, self.channels[-1]):
            if c not in self.channels_lookup:
                self.channels_lookup[c] = self.channels_lookup[self.channels[np.argmin(abs(self.channels - c))]]

        self.spatial_lookup = {c: i for i, c in enumerate(self.spatial)}
        self.spatial_lookup_training = copy.deepcopy(self.spatial_lookup)
        self.spatial_lookup[2] = self.spatial_lookup[3]  # 2x2 (not seen during training) will be treated as 3x3
        for c in range(1, self.spatial[-1]):
            if c not in self.spatial_lookup:
                self.spatial_lookup[c] = self.spatial_lookup[self.spatial[np.argmin(abs(self.spatial - c))]]

        n_ch, n_s = len(self.channels), len(self.spatial)
        self.embed_spatial = torch.nn.Embedding(n_s + 1, hid // 4)
        self.embed_channel = torch.nn.Embedding(n_ch + 1, hid // 4)

        self.register_buffer('dummy_ind', torch.tensor([n_ch, n_ch, n_s, n_s], dtype=torch.long).view(1, 4),
                             persistent=False)

    def tensor_shape_to_4d(self, sz):
        if len(sz) == 1:
            sz = (sz[0], 1)
        if len(sz) == 2:
            sz = (sz[0], sz[1], 1, 1)
        if len(sz) == 3:
            # Special treatment of 3D weights for some models like ViT.
            if sz[0] == 1 and min(sz[1:]) > 1:  # e.g. [1, 197, 768]
                s = int(np.floor(sz[1] ** 0.5))
                sz = (1, sz[2], s, s)
            else:
                sz = (sz[0], sz[1], sz[2], 1)
        return sz

    def forward(self, x, params_map, predict_class_layers=True):
        shape_ind = self.dummy_ind.repeat(len(x), 1)

        self.printed_warning = False
        for node_ind in params_map:
            sz = params_map[node_ind][0]['sz']
            if sz is None:
                continue

            sz_org = sz
            sz = self.tensor_shape_to_4d(sz)
            assert len(sz) == 4, sz

            if not predict_class_layers and params_map[node_ind][1] in ['cls_w', 'cls_b']:
                # keep the classification shape as though the GHN is used on the dataset it was trained on
                sz = (self.num_classes, *sz[1:])

            recognized_sz = 0
            for i in range(4):
                # if not in the dictionary, then use the maximum shape
                if i < 2:  # for out/in channel dimensions
                    shape_ind[node_ind, i] = self.channels_lookup[
                        sz[i] if sz[i] in self.channels_lookup else self.channels[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.channels_lookup_training)
                else:  # for kernel height/width
                    shape_ind[node_ind, i] = self.spatial_lookup[
                        sz[i] if sz[i] in self.spatial_lookup else self.spatial[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.spatial_lookup_training)

            if self.debug_level and not self.printed_warning:  # print a warning once per architecture
                if recognized_sz != 4:
                    print('WARNING: unrecognized shape %s, so the closest shape at index %s will be used instead.' % (
                        sz_org, ([self.channels[c.item()] if i < 2 else self.spatial[c.item()] for i, c in
                                  enumerate(shape_ind[node_ind])])))
                    self.printed_warning = True

        shape_embed = torch.cat(
            (self.embed_channel(shape_ind[:, 0]),
             self.embed_channel(shape_ind[:, 1]),
             self.embed_spatial(shape_ind[:, 2]),
             self.embed_spatial(shape_ind[:, 3])), dim=1)

        return x + shape_embed
