# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loaders for DeepNets-1M.

"""


import numpy as np
import torch.utils.data
import torchvision
import json
import h5py
import os
from ..deepnets1m import genotypes
from ..utils import adjust_net, rand_choice
from .genotypes import from_dict, PRIMITIVES_DEEPNETS1M
from .ops import *
from .net import Network
from .graph import Graph, GraphBatch


MAX_NODES_BATCH = 2200  # to fit larger meta batches into GPU memory (decreasing this number further may create a bias towards smaller architectures)


class DeepNets1M(torch.utils.data.Dataset):
    r"""
    Default args correspond to training a baseline GHN on CIFAR-10.
    """

    def __init__(self,
                 split='train',
                 nets_dir='./data',
                 virtual_edges=1,
                 num_ch=(32, 128),
                 fc_dim=(64, 512),
                 num_nets=None,
                 arch=None,
                 large_images=False,
                 verbose=False):
        super(DeepNets1M, self).__init__()

        self.split = split
        assert self.split in ['train', 'val', 'test', 'search',
                              'wide', 'deep', 'dense', 'bnfree', 'predefined'],\
            ('invalid split', self.split)
        self.is_train = self.split == 'train'

        self.virtual_edges = virtual_edges
        assert self.virtual_edges >= 1, virtual_edges

        if self.is_train:
            # During training we will randomly sample values from this range
            self.num_ch = torch.arange(num_ch[0], num_ch[1] + 1, 16)
            self.fc_dim = torch.arange(fc_dim[0], fc_dim[1] + 1, 64)

        self.large_images = large_images  # this affects some network parameters
        self.verbose = verbose

        # Load one of the splits
        if self.verbose:
            print('\nloading %s nets...' % self.split.upper())

        if self.split == 'predefined':
            self.nets = self._get_predefined()
            n_all = len(self.nets)
            self.nodes = torch.tensor([net.n_nodes for net in self.nets])
        else:
            self.h5_data = None
            self.h5_file = os.path.join(nets_dir, 'deepnets1m_%s.hdf5' % (split if split in ['train', 'search'] else 'eval'))

            self.primitives_dict = {op: i for i, op in enumerate(PRIMITIVES_DEEPNETS1M)}
            assert os.path.exists(self.h5_file), ('%s not found' % self.h5_file)

            # Load meta data to convert dataset files to graphs later in the _init_graph function
            to_int_dict = lambda d: { int(k): v for k, v in d.items() }
            with open(self.h5_file.replace('.hdf5', '_meta.json'), 'r') as f:
                meta = json.load(f)[split]
                n_all = len(meta['nets'])
                self.nets = meta['nets'][:n_all if num_nets is None else num_nets]
                self.primitives_ext =  to_int_dict(meta['meta']['primitives_ext'])
                self.op_names_net = to_int_dict(meta['meta']['unique_op_names'])
            self.h5_idx = [ arch ] if arch is not None else None
            self.nodes = torch.tensor([net['num_nodes'] for net in self.nets])

        if arch is not None:
            arch = int(arch)
            assert 0 <= arch < len(self.nets), \
                'architecture with index={} is not available in the {} split with {} architectures in total'.format(
                    arch, split, len(self.nets))
            self.nets = [self.nets[arch]]
            self.nodes = torch.tensor([self.nodes[arch]])

        if self.verbose:
            print('loaded {}/{} nets with {}-{} nodes (mean\u00B1std: {:.1f}\u00B1{:.1f})'.
                  format(len(self.nets),n_all,
                         self.nodes.min().item(),
                         self.nodes.max().item(),
                         self.nodes.float().mean().item(),
                         self.nodes.float().std().item()))


    @staticmethod
    def loader(meta_batch_size=1, **kwargs):
        nets = DeepNets1M(**kwargs)
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=NetBatchSampler(nets, meta_batch_size) if nets.is_train else None,
                                             batch_size=1,
                                             pin_memory=False,
                                             collate_fn=GraphBatch,
                                             num_workers=2 if meta_batch_size <= 1 else min(8, meta_batch_size))
        return iter(loader) if nets.is_train else loader


    def __len__(self):
        return len(self.nets)


    def __getitem__(self, idx):

        if self.split == 'predefined':
            graph = self.nets[idx]
        else:

            if self.h5_data is None:  # A separate fd is opened for each worker process
                self.h5_data = h5py.File(self.h5_file, mode='r')

            args = self.nets[idx]
            idx = self.h5_idx[idx] if self.h5_idx is not None else idx
            cell, n_cells = from_dict(args['genotype']), args['n_cells']
            graph = self._init_graph(self.h5_data[self.split][str(idx)]['adj'][()],
                                     self.h5_data[self.split][str(idx)]['nodes'][()],
                                     n_cells)

            if self.is_train:
                is_conv_dense = sum([n[0] in ['conv_5x5', 'conv_7x7'] for n in
                                     cell.normal + cell.reduce]) > 0
                num_params = args['num_params']['imagenet' if self.large_images else 'cifar10'] / 10 ** 6

                fc = rand_choice(self.fc_dim, 4)            # 64-256
                if num_params > 0.8 or not args['glob_avg'] or is_conv_dense or n_cells > 12:
                    C = self.num_ch.min()
                elif num_params > 0.4 or n_cells > 10:
                    C = rand_choice(self.num_ch, 2)         # 16-32
                elif num_params > 0.2 or n_cells > 8:
                    C = rand_choice(self.num_ch, 3)         # 16-64
                else:
                    C = rand_choice(self.num_ch)            # 16-128
                    if C <= 64:
                        fc = rand_choice(self.fc_dim)
                args['C'] = C.item()
                args['fc_dim'] = fc.item()

            net_args = {'genotype': cell}
            for key in ['norm', 'ks', 'preproc', 'glob_avg', 'stem_pool', 'C_mult',
                        'n_cells', 'fc_layers', 'C', 'fc_dim', 'stem_type']:
                if key == 'C' and self.split == 'wide':
                    net_args[key] = args[key] * (2 if self.large_images else 4)
                else:
                    net_args[key] = args[key]

            graph.net_args = net_args
            graph.net_idx = idx

        return graph


    def _init_graph(self, A, nodes, layers):

        N = A.shape[0]
        assert N == len(nodes), (N, len(nodes))

        node_feat = torch.zeros(N, 1, dtype=torch.long)
        node_info = [[] for _ in range(layers)]
        param_shapes = []

        for node_ind, node in enumerate(nodes):
            name = self.primitives_ext[node[0]]
            cell_ind = node[1]
            name_op_net = self.op_names_net[node[2]]

            sz = None

            if not name_op_net.startswith('classifier'):
                # fix some inconsistency between names in different versions of our code
                if len(name_op_net) == 0:
                    name_op_net = 'input'
                elif name_op_net.endswith('to_out.0.'):
                    name_op_net += 'weight'
                else:
                    parts = name_op_net.split('.')
                    for i, s in enumerate(parts):
                        if s == '_ops' and parts[i + 2] != 'op':
                            try:
                                _ = int(parts[i + 2])
                                parts.insert(i + 2, 'op')
                                name_op_net = '.'.join(parts)
                                break
                            except:
                                continue

                name_op_net = 'cells.%d.%s' % (cell_ind, name_op_net)

                stem_p = name_op_net.find('stem')
                pos_enc_p = name_op_net.find('pos_enc')
                if stem_p >= 0:
                    name_op_net = name_op_net[stem_p:]
                elif pos_enc_p >= 0:
                    name_op_net = name_op_net[pos_enc_p:]
                elif name.find('pool') >= 0:
                    sz = (1, 1, 3, 3)  # assume all pooling layers are 3x3 in our DeepNets-1M

            if name.startswith('conv_'):
                if name == 'conv_1x1':
                    sz = (3, 16, 1, 1)          # just some random shape for visualization purposes
                name = 'conv'                   # remove kernel size info from the name
            elif name.find('conv_') > 0 or name.find('pool_') > 0:
                name = name[:len(name) - 4]     # remove kernel size info from the name
            elif name == 'fc-b':
                name = 'bias'

            param_shapes.append(sz)
            node_feat[node_ind] = self.primitives_dict[name]
            if name.find('conv') >= 0 or name.find('pool') >= 0 or name in ['bias', 'bn', 'ln', 'pos_enc']:
                node_info[cell_ind].append((node_ind, name_op_net, name, sz, node_ind == len(nodes) - 2, node_ind == len(nodes) - 1))

        A = torch.from_numpy(A).long()
        A[A > self.virtual_edges] = 0
        assert A[np.diag_indices_from(A)].sum() == 0, (
            'no loops should be in the graph', A[np.diag_indices_from(A)].sum())

        graph = Graph(node_feat=node_feat, node_info=node_info, A=A)
        graph._param_shapes = param_shapes

        return graph


    def _get_predefined(self):

        graphs = []
        for idx, arch in enumerate(['resnet50', 'ViT']):  # Here, other torchvision models can be added

            num_classes = 1000 if self.large_images else 10  # the exact number should not be important at the graph construction stage
            if arch == 'resnet50':
                model = adjust_net(eval('torchvision.models.%s(num_classes=%d)' % (arch, num_classes)),
                                   large_input=self.large_images)
                args = {'genotype': arch}
            else:
                args = {'C': 128,
                        'genotype': eval('genotypes.%s' % arch),
                        'n_cells': 12,
                        'glob_avg': True,
                        'preproc': False,
                        'C_mult': 1}
                model = Network(num_classes=num_classes, is_imagenet_input=self.large_images, **args)

            graphs.append(Graph(model, net_args=args, net_idx=idx))

        return graphs



class NetBatchSampler(torch.utils.data.BatchSampler):
    r"""
    Wrapper to sample batches of architectures.
    Allows for infinite sampling and filtering out batches not meeting certain conditions.
    """
    def __init__(self, deepnets, meta_batch_size=1):
        super(NetBatchSampler, self).__init__(
            torch.utils.data.RandomSampler(deepnets) if deepnets.is_train
            else torch.utils.data.SequentialSampler(deepnets),
            meta_batch_size,
            drop_last=False)
        self.max_nodes_batch = MAX_NODES_BATCH if deepnets.is_train else None

    def check_batch(self, batch):
        return (self.max_nodes_batch is None or
                self.sampler.data_source.nodes[batch].sum() <=
                self.max_nodes_batch)

    def __iter__(self):
        while True:  # infinite sampler
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    if self.check_batch(batch):
                        yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.check_batch(batch):
                    yield batch
