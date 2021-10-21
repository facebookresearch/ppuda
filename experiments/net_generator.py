# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generator for the splits of DeepNets-1M.

In-Distribution splits: TRAIN, VAL, TEST.

Out-of-Distribution splits: WIDE, DEEP, DENSE, BNFREE.

PREDEFINED is created on the fly in the ppuda.deepnets1m.loader.


Example:

    python experiments/net_generator.py train 1000000 ./data

"""


import time
import h5py
import json
import os
from os.path import join
import sys
import subprocess
from ppuda.deepnets1m.graph import Graph
from ppuda.utils import *
from ppuda.deepnets1m.genotypes import *
from ppuda.deepnets1m.net import Network, get_cell_ind


def main():

    try:
        split = sys.argv[1].lower()
        N = int(sys.argv[2])
        data_dir = sys.argv[3]
    except:
        print('\nExample of usage: python deepnets1m/net_generator.py train 1000000 ./data\n')
        raise

    device = 'cpu'  # no much benefit of using cuda

    print(split, N, data_dir, device, flush=True)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    set_seed(0 if split == 'val' else 1)

    min_steps = 1
    medium_steps = 2
    max_steps = 4
    min_layers = 4
    deep_layers_all = np.arange(7, 11)
    max_layers = 18
    max_params = 10 ** 7

    # for 'train', 'val', 'test' we have the same network generator
    # for 'wide' we re-use the 'test' split and increase the number of channels when evaluate the model
    # for 'bnfree' the generator is the same except that all nets have no BN
    # 'predefined' is created on the fly in the deepnets1m.loader

    if split == 'deep':
        min_layers = 10
        deep_layers_all = [18]
        max_layers = 36
        max_params = 10 ** 8
    elif split == 'dense':
        min_steps = 2
        medium_steps = 6
        max_steps = 10
        max_params = 10 ** 8
    elif split == 'search':
        # allow a bit larger networks for search, since larger networks are more likely to have better final results
        medium_steps = 3
        max_steps = 6
        min_layers = 6
        deep_layers_all = [10]
        max_layers = 20
    else:
        assert split in ['train', 'val', 'test', 'wide', 'bnfree'], ('unsupported split: %s' % split)


    try:
        gitcommit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print('gitcommit:', gitcommit, flush=True)
    except Exception as e:
        print(e, flush=True)

    start = time.time()

    meta_data = {}
    meta_data[split] = {'nets': [], 'meta': {}}
    op_types, op_types_back, primitives, primitives_back = {}, {}, {}, {}

    h5_file = join(data_dir, 'deepnets1m_%s.hdf5' % split)
    meta_file = join(data_dir, 'deepnets1m_%s_meta.json' % split)

    for f in [h5_file, meta_file]:
        if os.path.exists(h5_file):
            raise ValueError('File %s already exists. The script will exit now to avoid accidental overwriting of the file.' % f)


    with h5py.File(h5_file, 'w') as h5_data:

        h5_data.attrs['title'] = 'DeepNets-1M'
        group = h5_data.create_group(split)

        while len(meta_data[split]['nets']) < N:

            layers = int(np.random.randint(min_layers, max_layers + 1))  # number of cells in total (convert to int to make it serializable)
            deep_layers = np.random.choice(deep_layers_all)  # a threshold to consider a deep network

            steps = int(np.random.randint(min_steps, (max_steps if layers <= deep_layers else medium_steps) + 1))  # number of summation nodes in a cell
            genotype = sample_genotype(steps=steps,
                                       only_pool=bool(np.random.rand() > 0.5),  # True means no trainable layers in the reduction cell
                                       drop_concat=bool(np.random.rand() > 0.5) if steps > 1 else False,  # drop some edges from the sum node to the final concat
                                       allow_none=steps > 1,    # none is the zero operation to allow sparse connections
                                       allow_transformer=True)  # allow to sample msa

            ks = int(np.random.choice([3, 5, 7]))   # kernel size of the first convolutional layer
            is_vit = sum([n[0] == 'msa' for n in genotype.normal + genotype.reduce]) > 0  # Visual Transformer
            is_cse = sum([n[0] == 'cse' for n in genotype.normal + genotype.reduce]) > 0  # Model with CSE
            has_none = sum([n[0] == 'none' for n in genotype.normal + genotype.reduce]) > 0

            is_cse2 = (sum([n[0] == 'cse' for n in genotype.normal]) > 1) or (
                    sum([n[0] == 'cse' for n in genotype.reduce]) > 1)  # training GHNs on networks with CSE often leads to NaN losses, so we will avoid them

            is_conv = sum([n[0].find('conv') >= 0 for n in genotype.normal + genotype.reduce]) > 0  # at least one simple conv op

            is_conv_large = (sum([n[0] in ['conv_5x5', 'conv_7x7'] for n in genotype.normal]) > 1) or (
                    sum([n[0] in ['conv_5x5', 'conv_7x7'] for n in genotype.reduce]) > 1)  # dense convolutions are memory consuming, so we will avoid them

            if (is_cse and not is_conv) or is_cse2 or is_conv_large:
                continue  # avoid some networks that are difficult to train or too memory consuming

            if not (is_cse or is_vit or is_conv):
                # print('no lear layers', genotype, flush=True)
                continue

            C_mult = int(np.random.choice([1, 2]))

            # Use 1x1 convolutional layers to match the channel dimensionality at the input of each cell
            if steps > 1 or C_mult > 1:
                preproc = True
            else:
                # allow some networks without those 1x1 conv layers for diversity
                if split == 'search':
                    # not sure what's the logic was here, but keep for consistency
                    preproc = bool((not is_vit and np.random.rand() > 0.2) or (is_vit and np.random.rand() > 0.8))
                else:
                    preproc = bool(not is_vit or np.random.rand() > 0.8)

            # Use global pooling most of the time instead of VGG-style head
            glob_avg = bool(is_vit or layers > deep_layers or np.random.rand() > 0.1)

            if split == 'bnfree':
                norm = None
            elif split == 'search':
                norm = 'bnorm'
            else:
                # Allow no BN in case of shallow networks and few ops
                norm = np.random.choice(['bnorm', None]) if layers <= (min_layers + 1) and steps <= medium_steps else 'bnorm'
            stem_type = int(np.random.choice([0, 1]))  # style of the stem: simple or ImageNet-style from DARTS
            net_args = {'stem_type': stem_type,
                        'stem_pool': bool(stem_type == 0 and np.random.rand() > 0.5),  # add extra pooling layer in case of a simple cell
                        'norm': norm,
                        'preproc': preproc,
                        'fc_layers': int(np.random.randint(1, 3)),  # number of fully connected layers before classification
                        'glob_avg': glob_avg,
                        'genotype': genotype,
                        'n_cells': layers,
                        'ks': ks,
                        'C_mult': C_mult,
                        'fc_dim': 256
                        }

            skip = False
            graph = None
            num_params = {}

            for dset_name in ['cifar10', 'imagenet']:

                model = Network(C=32,               # default number of channels
                                num_classes=10,     # does not matter at this stage
                                is_imagenet_input=dset_name=='imagenet',
                                **net_args).to(device)

                c, n = capacity(model)
                num_params[dset_name] = n

                if n > max_params:
                    print('too large architecture: %.2f M params \n' % (float(n) / 10 ** 6), flush=True)
                    skip = True
                    break

                if dset_name == 'cifar10':
                    try:
                        graph = Graph(model, ve_cutoff=250, list_all_nodes=True)
                    except Exception as e:
                        print('\n%d: unable to construct the graph: it is likely to be disconnected' % len(meta_data[split]['nets']),
                              'has_none={}, genotype={}'.
                              format(has_none, net_args['genotype']), flush=True)
                        print(e, '\n')
                        assert has_none  # to be disconnected it has to have none nodes
                        skip = True
                        break

            if skip:
                continue


            assert layers == len(graph.node_info), (layers, len(graph.node_info))
            cell_ind, n_nodes, nodes_array = 0, 0, []
            for j in range(layers):

                n_nodes += len(graph.node_info[j])

                for node in graph.node_info[j]:

                    param_name, name, sz = node[1:4]
                    cell_ind_ = get_cell_ind(param_name, layers)
                    if cell_ind_ is not None:
                        cell_ind = cell_ind_

                    assert cell_ind == j, (cell_ind, j, node)

                    if name == 'conv' and (len(sz) == 2 or sz[2] == sz[3] == 1):
                        name = 'conv_1x1'

                    if name not in primitives:
                        ind = len(primitives)
                        primitives[name] = ind
                        primitives_back[ind] = name

                    if param_name.startswith('cells.'):
                        # remove cells.x. prefix
                        pos1 = param_name.find('.')
                        assert param_name[pos1 + 1:].find('.') >= 0, node
                        pos2 = pos1 + param_name[pos1 + 1:].find('.') + 2
                        param_name = param_name[pos2:]

                    if param_name not in op_types:
                        ind = len(op_types)
                        op_types[param_name] = ind
                        op_types_back[ind] = param_name

                    nodes_array.append([primitives[name], cell_ind, op_types[param_name]])

            nodes_array = np.array(nodes_array).astype(np.uint16)

            A = graph._Adj.cpu().numpy().astype(np.uint8)
            assert nodes_array.shape[0] == n_nodes == A.shape[0] == graph.n_nodes, (nodes_array.shape, n_nodes, A.shape, graph.n_nodes)

            idx = len(meta_data[split]['nets'])
            group.create_dataset(str(idx) + '/adj', data=A)
            group.create_dataset(str(idx) + '/nodes', data=nodes_array)

            net_args['num_nodes'] = int(A.shape[0])
            net_args['num_params'] = num_params

            net_args['genotype'] = to_dict(net_args['genotype'])
            meta_data[split]['nets'].append(net_args)
            meta_data[split]['meta']['primitives_ext'] = primitives_back
            meta_data[split]['meta']['unique_op_names'] = op_types_back

            if (idx + 1) % 100 == 0 or idx >= N - 1:
                all_n_nodes = np.array([net['num_nodes'] for net in meta_data[split]['nets']])
                all_n_params = np.array([net['num_params']['cifar10'] for net in meta_data[split]['nets']])  / 10 ** 6
                print('N={} nets created: \t {}-{} nodes (mean\u00B1std: {:.1f}\u00B1{:.1f}) '
                      '\t {:.2f}-{:.2f} params (M) (mean\u00B1std: {:.2f}\u00B1{:.2f}) '
                      '\t {} unique primitives, {} unique param names '
                      '\t total time={:.2f} sec'.format(
                    idx + 1,
                    all_n_nodes.min(),
                    all_n_nodes.max(),
                    all_n_nodes.mean(),
                    all_n_nodes.std(),
                    all_n_params.min(),
                    all_n_params.max(),
                    all_n_params.mean(),
                    all_n_params.std(),
                    len(primitives_back),
                    len(op_types_back),
                    time.time() - start),
                    flush=True)

    with open(meta_file, 'w') as f:
        json.dump(meta_data, f)

    print('saved to %s and %s' % (h5_file, meta_file))

    print('\ndone')

    if split == 'bnfree':
        merge_eval(data_dir)  # assume bnfree was generated the last


# Merge all eval splits into one file
def merge_eval(data_dir):

    print('merging the evaluation splits into one file')

    meta_new = {}
    for split in ['val', 'test', 'wide', 'deep', 'dense', 'bnfree']:
        with open(join(data_dir, 'deepnets1m_%s_meta.json' % split), 'r') as f:
            meta_new[split] = json.load(f)[split]
            print(split, len(meta_new[split]), len(meta_new[split]['meta']), len(meta_new[split]['nets']))
    print(list(meta_new.keys()))
    with open(join(data_dir, 'deepnets1m_eval_meta.json'), 'w') as f:
        json.dump(meta_new, f)


    with h5py.File(join(data_dir, 'deepnets1m_eval.hdf5'), "w") as h5_data:
        for split in ['val', 'test', 'wide', 'deep', 'dense', 'bnfree']:
            with h5py.File(join(data_dir, 'deepnets1m_%s.hdf5' % split), "r") as data_file:
                h5_data.attrs['title'] = 'DeepNets-1M'
                group = h5_data.create_group(split)
                for i in range(len(data_file[split])):
                    A, nodes = data_file[split][str(i)]['adj'][()], data_file[split][str(i)]['nodes'][()]
                    group.create_dataset(str(i)+'/adj', data=A)
                    group.create_dataset(str(i)+'/nodes', data=nodes)
                    if i == 0:
                        print(split, len(data_file[split]), A.dtype, nodes.dtype)
    print('\ndone')


if __name__ == '__main__':
    main()
