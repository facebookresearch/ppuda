# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Initialization.

"""

import torch


def init(model, orth=True, beta=0, layer=0, max_sz=0, verbose=False):
    """
    Transforms the parameters (weights) of a neural net for initialization based on the paper
    "Boris Knyazev. Pretraining a Neural Network before Knowing Its Architecture.
    First Workshop of Pre-training: Perspectives, Pitfalls, and Paths Forward at ICML 2022."

    :param model: pytorch neural net
    :param orth: orthogonalize or not
    :param beta: noise scaler for conv/linear layers
    :param beta_norm: noise scaler for 1D batch/layer norm layers
    :param layer: the layer starting from which to transform the parameters
    :param max_sz: number of input/output channels starting from which to transform the parameters
    :param verbose: print debugging information
    :return: neural net with transformed parameters
    """

    if not (orth or beta > 0 or layer > 0 or max_sz > 0):
        return model

    if verbose:
        print(
            '\npostprocessing parameters: orth={} \t beta={} \t layer={} \t max_sz={}'.format(
                orth, beta, layer, max_sz
            ))

    layer_2d = 0  # layer counter

    has_2d_weight = lambda m: hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2
    num_2d_layers = len([m for m in model.modules() if has_2d_weight(m)])

    for module in model.modules():

        if not has_2d_weight(module):
            continue  # skip the layers that do not have the 'weight' attribute with at least 2 dimensions

        layer_2d += 1

        if layer_2d == num_2d_layers:
            continue  # skip the last classification layer

        weight = module.weight.data
        sz = weight.shape

        if layer_2d >= layer and max(sz[:2]) > max_sz:

            if verbose:
                ev_max = get_eigs(weight).max().item()  # to print statistics
                m1, m2 = weight.min().item(), weight.max().item()  # to print statistics

                def print_stats(w, ev_max, m1, m2, name):
                    print(
                        '{}\t layer = {} \t shape = {} \t{} min-max before:after = {:.2f}-{:.2f} : {:.2f}-{:.2f} \t{}'
                        ' max eig before:after = {:.2f} \t: {:.2f}'.format(
                            name, layer_2d, tuple(w.shape), '\t' if w.dim() <= 2 else '',
                            m1, m2, w.min().item(), w.max().item(), '\t' if w.min() > 0 else '',
                            ev_max, get_eigs(w).max().item()))

            if beta > 0:
                std_r = get_corr(weight).std()
                noise = beta * std_r * torch.randn_like(weight[max_sz:, max_sz:])
                weight[max_sz:, max_sz:].data += noise

                if verbose:
                    print_stats(weight, ev_max, m1, m2, 'add noise to conv-w: ')

            if orth:
                weight = orthogonalize(weight)
                if verbose:
                    print_stats(weight, ev_max, m1, m2, 'orthogonalization: ')

        # Can add noise to batch/layer norm layers and conv/linear biases, but was not found to be beneficial

        assert weight.shape == module.weight.shape, (weight.shape, module.weight.shape)
        module.weight.data = weight

    if verbose:
        print('done postprocessing!\n')

    return model


def orthogonalize(w):
    """
    Runs QR decomposition of weights and returns orthogonalized weights.
    :param w: 4D or 2D weights
    :return: orthogonalized weights of the same shape as input
    """
    flattened, is_transposed = weights2mat(w)
    rows, cols = flattened.shape
    assert rows >= cols, flattened.shape
    q, r = torch.linalg.qr(flattened)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph
    if is_transposed:
        q.t_()
    return q.view_as(w)


def get_eigs(w):
    """
    Computes eigenvalues of the weights.
    :param w: weights
    :return: eigenvalues
    """
    w = weights2mat(w)[0]
    evals, V = torch.linalg.eigh(w.t() @ w)
    return evals


def get_corr(w):
    """
    Computes correlation of the weight matrix.
    :param w: 2D weight
    :return: flattened correlation values
    """
    corr = torch.corrcoef(weights2mat(w)[0]).flatten()
    return corr


def weights2mat(w):
    """
    Converts n-D weights to 2D.
    :param w: n-D weights
    :param as_numpy:
    :return: 2D weights
    """
    w = w.reshape(w.shape[0], -1)
    is_transposed = w.shape[0] < w.shape[1]
    if is_transposed:
        w = w.t()
    return w, is_transposed
