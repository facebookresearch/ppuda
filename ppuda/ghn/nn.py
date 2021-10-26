# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph HyperNetworks.

"""


import torch
import torch.nn as nn
import os
from .mlp import MLP
from .gatedgnn import GatedGNN
from .decoder import MLPDecoder, ConvDecoder
from .layers import ShapeEncoder
from ..deepnets1m.ops import NormLayers, PosEnc
from ..deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
from ..deepnets1m.net import named_layered_modules
from ..deepnets1m.graph import Graph, GraphBatch
from ..utils import capacity, default_device
import time


def GHN1(dataset='imagenet'):
    """
        Loads GHN-1 trained on ImageNet or CIFAR-10.
        To load a GHN from an arbitrary checkpoint, use GHN.load(checkpoint_path).
        :param dataset: imagenet or cifar10
        :return: GHN-1 with trained weights
        """
    path = os.path.dirname(os.path.abspath(__file__))
    return GHN.load(os.path.join(path, '../../checkpoints/ghn1_%s.pt' % dataset))


def GHN2(dataset='imagenet'):
    """
    Loads GHN-2 trained on ImageNet or CIFAR-10.
    To load a GHN from an arbitrary checkpoint, use GHN.load(checkpoint_path).
    :param dataset: imagenet or cifar10
    :return: GHN-2 with trained weights
    """
    path = os.path.dirname(os.path.abspath(__file__))
    return GHN.load(os.path.join(path, '../../checkpoints/ghn2_%s.pt' % dataset))


def ghn_parallel(ghn):
    """
    For training a GHN on multiple GPUs.
    :param ghn: GHN instance
    :return: DataParallel wrapper of GHN
    """
    if isinstance(ghn, torch.nn.DataParallel):
        return ghn

    ghn = torch.nn.DataParallel(ghn)

    def scatter(inputs, kwargs, device_ids):
        nets_torch, graphs = inputs
        return graphs.scatter(device_ids, nets_torch), None

    def gather(outputs, output_device):
        return outputs  # nets_torch with predicted parameters on multiple devices

    ghn.scatter = scatter
    ghn.gather = gather

    return ghn


class GHN(nn.Module):
    r"""
    Graph HyperNetwork based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    (https://arxiv.org/abs/1810.05749)

    """
    def __init__(self,
                 max_shape,
                 num_classes,
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 debug_level=0):
        super(GHN, self).__init__()

        assert len(max_shape) == 4, max_shape
        self.layernorm = layernorm
        self.weight_norm = weight_norm
        self.ve = ve
        self.debug_level = debug_level
        self.num_classes = num_classes

        if layernorm:
            self.ln = nn.LayerNorm(hid)

        self.embed = torch.nn.Embedding(len(PRIMITIVES_DEEPNETS1M), hid)
        self.shape_enc = ShapeEncoder(hid=hid,
                                      num_classes=num_classes,
                                      max_shape=max_shape,
                                      debug_level=debug_level)
        if hypernet == 'gatedgnn':
            self.gnn = GatedGNN(in_features=hid, ve=ve)
        elif hypernet == 'mlp':
            self.gnn = MLP(in_features=hid, hid=(hid, hid))
        else:
            raise NotImplementedError(hypernet)

        if decoder == 'conv':
            fn_dec, layers = ConvDecoder, (hid * 4, hid * 8)
        elif decoder == 'mlp':
            fn_dec, layers = MLPDecoder, (hid * 2, )
        else:
            raise NotImplementedError(decoder)
        self.decoder = fn_dec(in_features=hid,
                              hid=layers,
                              out_shape=max_shape,
                              num_classes=num_classes)

        max_ch = max(max_shape[:2])
        self.decoder_1d = MLP(hid, hid=(hid * 2, 2 * max_ch),
                              last_activation=None)
        self.bias_class = nn.Sequential(nn.ReLU(),
                                        nn.Linear(max_ch, num_classes))


    @staticmethod
    def load(checkpoint_path, debug_level=1, device=default_device(), verbose=False):
        state_dict = torch.load(checkpoint_path, map_location=device)
        ghn = GHN(**state_dict['config'], debug_level=debug_level).to(device).eval()
        ghn.load_state_dict(state_dict['state_dict'])
        if verbose:
            print('GHN with {} parameters loaded from epoch {}.'.format(capacity(ghn)[1], state_dict['epoch']))
        return ghn


    def forward(self, nets_torch, graphs=None, return_embeddings=False, predict_class_layers=True, bn_train=True):
        r"""
        Predict parameters for a list of >=1 networks.
        :param nets_torch: one network or a list of networks, each is based on nn.Module.
                           In case of evaluation, only one network can be passed.
        :param graphs: GraphBatch object in case of training.
                       For evaluation, graphs can be None and will be constructed on the fly given the nets_torch in this case.
        :param return_embeddings: True to return the node embeddings obtained after the last graph propagation step.
                                  return_embeddings=True is used for property prediction experiments.
        :param predict_class_layers: default=True predicts all parameters including the classification layers.
                                     predict_class_layers=False is used in fine-tuning experiments.
        :param bn_train: default=True sets BN layers in nets_torch into the training mode (required to evaluate predicted parameters)
                        bn_train=False is used in fine-tuning experiments
        :return: nets_torch with predicted parameters and node embeddings if return_embeddings=True
        """

        if not self.training:
            assert isinstance(nets_torch,
                              nn.Module) or len(nets_torch) == 1, \
                'constructing the graph on the fly is only supported for a single network'

            if isinstance(nets_torch, list):
                nets_torch = nets_torch[0]

            if self.debug_level:
                if self.debug_level > 1:
                    valid_ops = graphs[0].num_valid_nodes(nets_torch)
                start_time = time.time()  # do not count any debugging steps above

            if graphs is None:
                graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.embed.weight.device)

        else:
            assert graphs is not None, \
                'constructing the graph on the fly is only supported in the evaluation mode'

        # Find mapping between embeddings and network parameters
        param_groups, params_map, max_sz = self._map_net_params(graphs, nets_torch, self.debug_level > 0)

        if self.debug_level or not self.training:
            n_params_true = sum([capacity(net)[1] for net in (nets_torch if isinstance(nets_torch, list) else [nets_torch])])
            if self.debug_level > 1:
                print('\nnumber of learnable parameter tensors: {}, total number of parameters: {}'.format(
                    valid_ops, n_params_true))

        # Obtain initial embeddings for all nodes
        x = self.shape_enc(self.embed(graphs.node_feat[:, 0]), params_map, predict_class_layers=predict_class_layers)

        # Update node embeddings using a GatedGNN, MLP or another model
        x = self.gnn(x, graphs.edges, graphs.node_feat[:, 1])

        if self.layernorm:
            x = self.ln(x)

        # Predict max-sized parameters for a batch of nets using decoders
        w = {}
        for key, inds in param_groups.items():
            if len(inds) == 0:
                continue
            x_ = x[torch.tensor(inds, device=x.device)]
            if key in ['4d', 'cls_w']:
                w[key] = self.decoder(x_, max_sz, key=='cls_w')
            else:
                w[key] = self.decoder_1d(x_).view(len(inds), 2, -1)
                if key == 'cls_b':
                    w[key] = self.bias_class(w[key])

        # Transfer predicted parameters (w) to the networks
        n_tensors, n_params = 0, 0
        for matched, key, w_ind in params_map.values():

            if w_ind is None:
                continue  # e.g. pooling

            if not predict_class_layers and key in ['cls_w', 'cls_b']:
                continue  # do not set the classification parameters when fine-tuning

            m, sz, is_w = matched['module'], matched['sz'], matched['is_w']
            for it in range(2 if (len(sz) == 1 and is_w) else 1):

                if len(sz) == 1:
                    # separately set for BN/LN biases as they are
                    # not represented as separate nodes in graphs
                    w_ = w[key][w_ind][1 - is_w + it]
                    if it == 1:
                        assert (type(m) in NormLayers and key == '1d'), \
                            (type(m), key)
                else:
                    w_ = w[key][w_ind]

                sz_set = self._set_params(m, self._tile_params(w_, sz), is_w=is_w & ~it)
                n_tensors += 1
                n_params += torch.prod(torch.tensor(sz_set))

        if not self.training and bn_train:

            def bn_set_train(module):
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True

            nets_torch.apply(bn_set_train)  # set BN layers to the training mode to enable evaluation without having running statistics

        if self.debug_level and not self.training:

            end_time = time.time() - start_time

            print('number of parameter tensors predicted using GHN: {}, '
                  'total parameters predicted: {} ({}), time to predict (on {}): {:.4f} sec'.format(
                n_tensors,
                n_params,
                ('matched!' if n_params_true == n_params else 'error! not matched').upper(),
                str(x.device).upper(),
                end_time))

            if self.debug_level > 1:
                assert valid_ops == n_tensors, (
                    'number of learnable tensors ({}) must be the same as the number of predicted tensors ({})'.format(
                        valid_ops, n_tensors))


            if self.debug_level > 2:
                print('predicted parameter stats:')
                for n, p in nets_torch.named_parameters():
                    print('{:30s} ({:30s}): min={:.3f} \t max={:.3f} \t mean={:.3f} \t std={:.3f} \t norm={:.3f}'.format(
                        n[:30],
                        str(p.shape)[:30],
                        p.min().item(),
                        p.max().item(),
                        p.mean().item(),
                        p.std().item(),
                        torch.norm(p).item()))
        elif self.debug_level or not self.training:
            assert n_params == n_params_true, ('number of predicted ({}) or actual ({}) parameters must match'.format(
                n_params, n_params_true))

        return (nets_torch, x) if return_embeddings else nets_torch


    def _map_net_params(self, graphs, nets_torch, sanity_check=False):
        r"""
        Matches the parameters in the models with the nodes in the graph.
        Performs additional steps.
        :param graphs: GraphBatch object
        :param nets_torch: a single neural network of a list
        :param sanity_check:
        :return: mapping, params_map, max_sz
        """
        mapping = {'1d': [], '4d': [], 'cls_w': [], 'cls_b': []}
        params_map = {}
        max_sz = (0, 0)  # maximum kernel spatial size in this batch of networks

        nets_torch = [nets_torch] if type(nets_torch) not in [tuple, list] else nets_torch

        for b, (node_info, net) in enumerate(zip(graphs.node_info, nets_torch)):
            target_modules = named_layered_modules(net)

            param_ind = torch.sum(graphs.n_nodes[:b]).item()

            for cell_id in range(len(node_info)):
                matched_names = []
                for (node_ind, param_name, name, sz, last_weight, last_bias) in node_info[cell_id]:

                    matched = []
                    for m in target_modules[cell_id]:
                        if m['param_name'].startswith(param_name):
                            matched.append(m)
                            if not sanity_check:
                                break
                    if len(matched) > 1:
                        raise ValueError(cell_id, node_ind, param_name, name, [
                            (t, (m.weight if is_w else m.bias).shape) for
                            t, m, is_w in matched])
                    elif len(matched) == 0:
                        if sz is not None:
                            params_map[param_ind + node_ind] = ({'sz': sz}, None, None)

                        if sanity_check:
                            for pattern in ['input', 'sum', 'concat', 'pool', 'glob_avg', 'msa', 'cse']:
                                good = name.find(pattern) >= 0
                                if good:
                                    break
                            assert good, \
                                (cell_id, param_name, name,
                                 node_info[cell_id],
                                 target_modules[cell_id])
                    else:
                        matched_names.append(matched[0]['param_name'])
                        sz = matched[0]['sz']
                        if len(sz) == 1:
                            key = 'cls_b' if last_bias else '1d'
                        else:
                            key = 'cls_w' if last_weight else '4d'
                            if len(sz) > 2:
                                max_sz = (max(max_sz[0], sz[2]), max(max_sz[1], sz[3]))
                        params_map[param_ind + node_ind] = (matched[0], key, len(mapping[key]))
                        mapping[key].append(param_ind + node_ind)

                assert len(matched_names) == len(set(matched_names)), (
                    'all matched names must be unique to avoid predicting the same paramters for different moduels',
                    len(matched_names), len(set(matched_names)))
                matched_names = set(matched_names)

                # Prune redundant ops in Network by setting their params to None
                for m in target_modules[cell_id]:
                    if m['is_w'] and m['param_name'] not in matched_names:
                        m['module'].weight = None
                        if hasattr(m['module'], 'bias') and m['module'].bias is not None:
                            m['module'].bias = None

        return mapping, params_map, max_sz


    def _tile_params(self, w, target_shape):
        r"""
        Makes the shape of predicted parameter tensors the same as the target shape by tiling/slicing across channels dimensions.
        :param w: predicted tensor, for example of shape (64, 64, 11, 11)
        :param target_shape: tuple, for example (512, 256, 3, 3)
        :return: tensor of shape target_shape
        """
        t, s = target_shape, w.shape

        # Slice first to avoid tiling a larger tensor
        if len(t) == 1:
            if len(s) == 2:
                w = w[:min(t[0], s[0]), 0]
            elif len(s) > 2:
                w = w[:min(t[0], s[0]), 0, 0, 0]
        elif len(t) == 2:
            if len(s) > 2:
                w = w[:min(t[0], s[0]), :min(t[1], s[1]), 0, 0]
        else:
            w = w[:min(t[0], s[0]), :min(t[1], s[1]), :min(t[2], s[2]), :min(t[3], s[3])]

        s = w.shape
        assert len(s) == len(t), (s, t)

        # Tile out_channels
        if t[0] > s[0]:
            n_out = t[0] // s[0] + 1
            if len(t) == 1:
                w = w.repeat(n_out)[:t[0]]
            elif len(t) == 2:
                w = w.repeat((n_out, 1))[:t[0]]
            else:
                w = w.repeat((n_out, 1, 1, 1))[:t[0]]

        # Tile in_channels
        if len(t) > 1:
            if t[1] > s[1]:
                n_in = t[1] // s[1] + 1
                if len(t) == 2:
                    w = w.repeat((1, n_in))[:, :t[1]]
                else:
                    w = w.repeat((1, n_in, 1, 1))[:, :t[1]]

        # Chop out any extra bits tiled
        if len(t) == 1:
            w = w[:t[0]]
        elif len(t) == 2:
            w = w[:t[0], :t[1]]
        else:
            w = w[:t[0], :t[1], :t[2], :t[3]]

        return w


    def _set_params(self, module, tensor, is_w):
        r"""
        Copies the predicted parameter tensor to the appropriate field of the module object.
        :param module: nn.Module
        :param tensor: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: the shape of the copied tensor
        """
        if self.weight_norm:
            tensor = self._normalize(module, tensor, is_w)
        key = 'weight' if is_w else 'bias'
        target_param = module.weight if is_w else module.bias
        sz_target = target_param.shape
        if self.training:
            module.__dict__[key] = tensor  # set the value avoiding the internal logic of PyTorch
            # update parameters, so that named_parameters() will return tensors
            # with gradients (for multigpu and other cases)
            module._parameters[key] = tensor
        else:
            assert isinstance(target_param, nn.Parameter), type(target_param)
            # copy to make sure there is no sharing of memory
            target_param.data = tensor.clone()

        set_param = module.weight if is_w else module.bias
        assert sz_target == set_param.shape, (sz_target, set_param.shape)
        return set_param.shape


    def _normalize(self, module, p, is_w):
        r"""
        Normalizes the predicted parameter tensor according to the Fan-In scheme described in the paper.
        :param module: nn.Module
        :param p: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: normalized predicted tensor
        """
        if p.dim() > 1:

            sz = p.shape

            if len(sz) > 2 and sz[2] >= 11 and sz[0] == 1:
                assert isinstance(module, PosEnc), (sz, module)
                return p    # do not normalize positional encoding weights

            no_relu = len(sz) > 2 and (sz[1] == 1 or sz[2] < sz[3])
            if no_relu:
                # layers not followed by relu
                beta = 1.
            else:
                # for layers followed by rely increase the weight scale
                beta = 2.

            # fan-out:
            # p = p * (beta / (sz[0] * p[0, 0].numel())) ** 0.5

            # fan-in:
            p = p * (beta / p[0].numel()) ** 0.5

        else:

            if is_w:
                p = 2 * torch.sigmoid(0.5 * p)  # BN/LN norm weight is [0,2]
            else:
                p = torch.tanh(0.2 * p)         # bias is [-1,1]

        return p
