# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Some functionality in this script is based on the DARTS code: https://github.com/quark0/darts (Apache License 2.0)

"""


from collections import namedtuple
import torch.nn.functional as F
import torch
from torch.autograd import Variable


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


PRIMITIVES_DEEPNETS1M = [
    'max_pool',
    'avg_pool',
    'sep_conv',
    'dil_conv',
    'conv',
    'msa',
    'cse',
    'sum',
    'concat',
    'input',
    'bias',
    'bn',
    'ln',
    'pos_enc',
    'glob_avg',
]


DARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
                 reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                  reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

ViT = Genotype(normal=[('none', 0), ('msa', 1)], normal_concat=[2],
               reduce=[('none', 0), ('avg_pool_3x3', 1)], reduce_concat=[2])



def from_dict(genotype):
    return Genotype(normal=genotype['normal'],
                    normal_concat=genotype['normal_concat'],
                    reduce=genotype['reduce'],
                    reduce_concat=genotype['reduce_concat'])


def to_dict(genotype):
    return {'normal': list(genotype.normal),
            'normal_concat': list(genotype.normal_concat),
            'reduce': list(genotype.reduce),
            'reduce_concat': list(genotype.reduce_concat)}



def sample_genotype(steps=1, only_pool=False, allow_none=True, drop_concat=True, allow_transformer=False):

    # Extended set of primitives based on https://github.com/quark0/darts/blob/master/cnn/genotypes.py
    PRIMITIVES_DARTS_EXT = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'conv_1x1',
        'conv_7x1_1x7',
        'conv_3x3',
        'conv_5x5',
        'conv_7x7',
        'msa',
        'cse'
    ]

    multiplier = steps
    k = sum(1 for i in range(steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES_DARTS_EXT)
    alphas_normal = Variable(1e-3 * torch.randn(k, num_ops))
    alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops))

    if only_pool:
        assert PRIMITIVES_DARTS_EXT[3] == 'skip_connect', PRIMITIVES_DARTS_EXT
        assert PRIMITIVES_DARTS_EXT[4] == 'sep_conv_3x3', PRIMITIVES_DARTS_EXT
        alphas_reduce[:, 4:] = -1000  # prevent sampling operators with learnable params to sample the architectures similar to the best DARTS cell

    if not allow_transformer:
        ind = PRIMITIVES_DARTS_EXT.index('msa')
        assert ind == len(PRIMITIVES_DARTS_EXT) - 2, (ind, PRIMITIVES_DARTS_EXT)
        alphas_normal[:, ind] = -1000
        alphas_reduce[:, ind] = -1000

    def _parse(weights):
        # Based on https://github.com/quark0/darts/blob/master/cnn/model_search.py#L135
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x])) if (k != PRIMITIVES_DARTS_EXT.index('none') or allow_none)))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES_DARTS_EXT.index('none') or allow_none:
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES_DARTS_EXT[k_best], j))
            start = end
            n += 1
        return gene

    gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.numpy())
    gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.numpy())

    if drop_concat:
        concat = []
        for i in range(2 + steps - multiplier, steps + 2):
            if i == steps + 1 or torch.rand(1).item() > 0.5:  # always add the last otherwise the features from the previous sum nodes will be lost
                concat.append(i)
    else:
        concat = range(2 + steps - multiplier, steps + 2)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype
