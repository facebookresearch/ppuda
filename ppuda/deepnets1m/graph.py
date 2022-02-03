# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Containers for computational graphs.

"""


import numpy as np
import heapq
import torch
import torch.nn as nn
import torch.nn.parallel.scatter_gather as scatter_gather
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from .ops import NormLayers, PosEnc
from .net import get_cell_ind
from .genotypes import PRIMITIVES_DEEPNETS1M


class GraphBatch():
    r"""
    Container for a batch of Graph objects.

    Example:

        batch = GraphBatch([Graph(torchvision.models.resnet50())])

    """

    def __init__(self, graphs):
        r"""
        :param graphs: iterable, where each item is a Graph object.
        """
        self.n_nodes, self.node_feat, self.node_info, self.edges, self.net_args, self.net_inds = [], [], [], [], [], []
        self._n_edges = []
        self.graphs = graphs
        if graphs is not None:
            for graph in graphs:
                self.append(graph)


    def append(self, graph):
        graph_offset = len(self.n_nodes)                    # current index of the graph in a batch
        self.n_nodes.append(len(graph.node_feat))           # number of nodes
        self._n_edges.append(len(graph.edges))              # number of edges
        self.node_feat.append(torch.cat((graph.node_feat,   # primitive type
                                         graph_offset + torch.zeros(len(graph.node_feat), 1, dtype=torch.long)), dim=1))    # graph index for each node
        self.edges.append(torch.cat((graph.edges,
                                     graph_offset + torch.zeros(len(graph.edges), 1, dtype=torch.long)), dim=1))            # graph index for each edge

        self.node_info.append(graph.node_info)      # op names, ids, etc.
        self.net_args.append(graph.net_args)        # a dictionary of arguments to construct a Network object
        self.net_inds.append(graph.net_idx)         # network integer identifier (optional)


    def scatter(self, device_ids, nets):
        """
        Distributes the batch of graphs and networks to multiple CUDA devices.
        :param device_ids: list of CUDA devices
        :param nets: list of networks
        :return: list of tuples of networks and corresponding graphs
        """

        n_graphs = len(self.n_nodes)  # number of graphs in a batch
        graphs_per_device = int(np.ceil(n_graphs / len(device_ids)))

        if len(device_ids) > 1:
            sorted_idx = self._sort_by_nodes(len(device_ids), graphs_per_device)
            nets = [nets[i] for i in sorted_idx]

        chunks_iter = np.arange(0, n_graphs, graphs_per_device)
        node_chunks = [sum(self.n_nodes[i:i + graphs_per_device]) for i in chunks_iter]
        edge_chunks = [sum(self._n_edges[i:i + graphs_per_device]) for i in chunks_iter]
        n_nodes_chunks = [len(self.n_nodes[i:i + graphs_per_device]) for i in chunks_iter]
        self._cat()

        self.node_feat = scatter_gather.Scatter.apply(device_ids, node_chunks, 0, self.node_feat)
        self.edges = scatter_gather.Scatter.apply(device_ids, edge_chunks, 0, self.edges)
        self.n_nodes = scatter_gather.Scatter.apply(device_ids, n_nodes_chunks, 0, self.n_nodes)

        batch_lst = []  # each item in the list is a GraphBatch instance
        for device, i in enumerate(chunks_iter):
            # update graph_offset for each device
            self.node_feat[device][:, -1] = self.node_feat[device][:, -1] - graphs_per_device * device
            self.edges[device][:, -1] = self.edges[device][:, -1] - graphs_per_device * device
            graphs = GraphBatch([])
            graphs.node_feat = self.node_feat[device]
            graphs.edges = self.edges[device]
            graphs.n_nodes = self.n_nodes[device]
            graphs.node_info = self.node_info[i:i + graphs_per_device]
            graphs.net_args = self.net_args[i:i + graphs_per_device]
            graphs.net_inds = self.net_inds[i:i + graphs_per_device]
            batch_lst.append((nets[i:i + graphs_per_device], graphs))  # match signature of the GHN forward pass

        return batch_lst


    def to_device(self, device):
        if isinstance(device, (tuple, list)):
            device = device[0]
        self._cat()
        self.n_nodes = self.n_nodes.to(device)
        self.node_feat = self.node_feat.to(device)
        self.edges = self.edges.to(device)
        return self


    def _sort_by_nodes(self, num_devices, graphs_per_device):
        """
        Sorts graphs and associated attributes in a batch by the number of nodes such
        that the memory consumption is more balanced across GPUs.
        :param num_devices: number of GPU devices (must be more than 1)
        :param graphs_per_device: number of graphs per GPU
                                (all GPUs are assumed to receive the same number of graphs)
        :return: indices of sorted graphs
        """
        n_nodes = np.array(self.n_nodes)
        sorted_idx = np.argsort(n_nodes)[::-1]
        n_nodes = n_nodes[sorted_idx]

        heap = [(0, idx) for idx in range(num_devices)]
        heapq.heapify(heap)
        idx_groups = {}
        for i in range(num_devices):
            idx_groups[i] = []

        for idx, n in enumerate(n_nodes):
            while True:
                set_sum, set_idx = heapq.heappop(heap)
                if len(idx_groups[set_idx]) < graphs_per_device:
                    break
            idx_groups[set_idx].append(sorted_idx[idx])
            heapq.heappush(heap, (set_sum + n, set_idx))

        idx = np.concatenate([np.array(v) for v in idx_groups.values()])

        # Sort everything according to the idx order
        self.n_nodes = [self.n_nodes[i] for i in idx]
        self._n_edges = [self._n_edges[i] for i in idx]
        self.node_info = [self.node_info[i] for i in idx]
        self.net_args = [self.net_args[i] for i in idx]
        self.net_inds = [self.net_inds[i] for i in idx]
        node_feat, edges = [], []
        for graph_offset, i in enumerate(idx):
            # update graph_offset for each graph
            node_feat_i = self.node_feat[i]
            edges_i = self.edges[i]
            node_feat_i[:, -1] = graph_offset
            edges_i[:, -1] = graph_offset
            node_feat.append(node_feat_i)
            edges.append(edges_i)
        self.node_feat = node_feat
        self.edges = edges
        return idx


    def _cat(self):
        if not isinstance(self.n_nodes, torch.Tensor):
            self.n_nodes = torch.tensor(self.n_nodes, dtype=torch.long)
        if not isinstance(self.node_feat, torch.Tensor):
            self.node_feat = torch.cat(self.node_feat)
        if not isinstance(self.edges, torch.Tensor):
            self.edges = torch.cat(self.edges)


    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.n_nodes)

    def __iter__(self):
        for graph in self.graphs:
            yield graph


class Graph():
    r"""
    Container for a computational graph of a neural network.

    Example:

        graph = Graph(torchvision.models.resnet50())

    """

    def __init__(self, model=None, node_feat=None, node_info=None, A=None, edges=None, net_args=None, net_idx=None, ve_cutoff=50, list_all_nodes=False):
        r"""
        :param model: Neural Network inherited from nn.Module
        """

        assert node_feat is None or model is None, 'either model or other arguments must be specified'

        self.model = model
        self._list_all_nodes = list_all_nodes  # True in case of dataset generation
        self.nx_graph = None  # NetworkX DiGraph instance

        if model is not None:
            sz = model.expected_input_sz if hasattr(model, 'expected_input_sz') else 224   # assume ImageNet image width/heigh by default
            self.expected_input_sz = sz if isinstance(sz, (tuple, list)) else (3, sz, sz)  # assume images by default
            self.n_cells = self.model._n_cells if hasattr(self.model, '_n_cells') else 1
            self._build_graph()   # automatically construct an initial computational graph
            self._filter_graph()  # remove redundant/unsupported nodes
            self._add_virtual_edges(ve_cutoff=ve_cutoff)  # add virtual edges
            self._construct_features()  # initialize torch.Tensor node and edge features
        else:
            self.n_nodes = len(node_feat)
            self.node_feat = node_feat
            self.node_info = node_info

            if edges is None:
                if not isinstance(A, torch.Tensor):
                    A = torch.from_numpy(A).long()
                ind = torch.nonzero(A)
                self.edges = torch.cat((ind, A[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)
            else:
                self.edges = edges
            self._Adj = A

        self.net_args = net_args
        self.net_idx = net_idx


    def num_valid_nodes(self, model=None):
        r"""
        Counts the total number of learnable parameter tensors.
        The function aims to find redundant parameter tensors that are disconnected from the computational graph.
        The function if based on computing gradients and, thus, is not reliable for all architectures.
        :param model: nn.Module based object
        :return: total number of learnable parameter tensors
        """
        if model is None:
            model = self.model
            expected_input_sz = self.expected_input_sz
        else:
            sz = model.expected_input_sz if hasattr(model, 'expected_input_sz') else 224
            expected_input_sz = sz if isinstance(sz, (tuple, list)) else (3, sz, sz)

        device = list(model.parameters())[0].device  # assume all parameters on the same device
        loss = model((torch.rand(1, *expected_input_sz, device=device) - 0.5) / 2)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss.mean()
        if torch.isnan(loss):
            print('could not estimate the number of learnable parameter tensors due the %s loss', str(loss))
            return -1
        else:
            loss.backward()
            valid_ops = 0
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert p.grad is not None and p.dim() > 0, (name, p.grad)
                    s = p.grad.abs().sum()
                    if s > 1e-20:
                        valid_ops += 1
                    # else:
                    #     print(name, p.shape, s)

        return valid_ops


    def _build_graph(self):
        r"""
        Constructs a graph of a neural network in the automatic way.
        This function is written based on Sergey Zagoruyko's https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py (MIT License)
        PyTorch 1.9+ is required to run this script correctly for some architectures.
        Currently, the function is not written very clearly and may be improved.
        """

        param_map = {id(weight): (name, module) for name, (weight, module) in self._named_modules().items()}
        nodes, edges, seen = [], [], {}

        def get_attr(fn):
            attrs = dict()
            for attr in dir(fn):
                if not attr.startswith('_saved_'):
                    continue
                val = getattr(fn, attr)
                attr = attr[len('_saved_'):]
                if torch.is_tensor(val):
                    attrs[attr] = "[saved tensor]"
                elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
                    attrs[attr] = "[saved tensors]"
                else:
                    attrs[attr] = str(val)
            return attrs

        def traverse_graph(fn):
            assert not torch.is_tensor(fn)
            if fn in seen:
                return seen[fn]

            fn_name = str(type(fn).__name__)
            node_link, link_start = None, None
            if fn_name.find('AccumulateGrad') < 0:
                leaf_nodes = []
                for u in fn.next_functions:
                    if u[0] is not None:
                        if hasattr(u[0], 'variable'):
                            var = u[0].variable
                            name, module = param_map[id(var)]
                            if type(module) in NormLayers and name.find('.bias') >= 0:
                                continue  # do not add biases of NormLayers as nodes
                            leaf_nodes.append({'id': u[0],
                                               'param_name': name,
                                               'attrs': {'size': var.size()},
                                               'module': module})
                            assert len(u[0].next_functions) == 0

                if len(leaf_nodes) == 0:
                    leaf_nodes.append({'id': fn,
                                       'param_name': fn_name,
                                       'attrs': get_attr(fn),
                                       'module': None})

                assert not hasattr(fn, 'variable'), fn.variable

                for leaf in leaf_nodes:
                    node_link = str(id(leaf['id']))
                    if link_start is None:
                        link_start = node_link

                    seen[leaf['id']] = (node_link, leaf['param_name'])
                    nodes.append({'id': node_link,
                                  'param_name': leaf['param_name'],
                                  'attrs': leaf['attrs'],
                                  'module': leaf['module']})

            seen[fn] = (node_link, fn_name)

            # recurse
            if hasattr(fn, 'next_functions'):
                for u in fn.next_functions:
                    if u[0] is not None:
                        link_, name_ = traverse_graph(u[0])
                        if link_ is not None and link_start != link_:
                            edges.append((link_start, link_) if name_.find('bias') >= 0 else (link_, link_start))

            return node_link, fn_name

        var = self.model(torch.randn(1, *self.expected_input_sz))
        # take only the first output, but can in principle handle multiple outputs, e.g. from auxiliary classifiers
        traverse_graph((var[0] if isinstance(var, (tuple, list)) else var).grad_fn)  # populate nodes and edges

        nodes_lookup = { node['id']: i  for i, node in enumerate(nodes) }
        A = np.zeros((len(nodes) + 1, len(nodes) + 1))  # +1 for the input node added below
        for out_node_id, in_node_id in edges:
            A[nodes_lookup[out_node_id], nodes_lookup[in_node_id]] = 1

        # Fix fc layers nodes and edge directions
        for i, node in enumerate(nodes):
            if isinstance(node['module'], nn.Linear) and node['param_name'].find('.weight') >= 0:
                # assert node['module'].bias is not None, ('this rewiring may not work in case of no biases', node)
                for out_neigh in np.where(A[i, :])[0]:  # all nodes where there is an edge from i
                    A[np.where(A[:, out_neigh])[0], i] = 1  # rewire edges coming to out_neigh (bias) to node i (weight)
                    A[:, out_neigh] = 0  # remove all edges to out_neigh except for the edge from i to out_neigh
                    A[i, out_neigh] = 1
                    A[i, i] = 0  # remove loop

        # Add input node
        nodes.append({'id': 'input', 'param_name': 'input', 'attrs': None, 'module': None})
        ind = np.where(A[:, :-1].sum(0) == 0)[0]
        assert len(ind) == 1, ind
        A[-1, ind] = 1

        # Sort nodes in a topological order consistent with forward propagation
        A[np.diag_indices_from(A)] = 0
        ind = np.array(list(nx.topological_sort(nx.DiGraph(A))))
        nodes = [nodes[i] for i in ind]
        A = A[ind, :][:, ind]

        # Adjust graph for Transformers to be consistent with our original code
        for i, node in enumerate(nodes):
            if isinstance(node['module'], PosEnc):
                nodes.insert(i + 1, { 'id': 'sum_pos_enc', 'param_name': 'AddBackward0', 'attrs': None, 'module': None })
                A = np.insert(A, i, 0, axis=0)
                A = np.insert(A, i, 0, axis=1)
                A[i, i + 1] = 1  # pos_enc to sum

        self._Adj = A
        self._nodes = nodes

        return


    def _filter_graph(self):
        r"""
        Remove redundant/unsupported nodes from the automatically constructed graphs.
        :return:
        """

        # These ops will not be added to the graph
        unsupported_modules = set()
        for i, node in enumerate(self._nodes):
            ind = node['param_name'].find('Backward')
            name = node['param_name'][:len(node['param_name']) if ind == -1 else ind]
            if type(node['module']) not in MODULES and name not in MODULES:
                unsupported_modules.add(node['param_name'])

        # Add ops requiring extra checks before removing
        unsupported_modules = ['Mul', 'Clone'] + list(unsupported_modules) + \
                              ['Mean', 'Add', 'Cat']

        for pattern in unsupported_modules:

            ind_keep = []

            for i, node in enumerate(self._nodes):
                op_name, attrs = node['param_name'], node['attrs']

                if op_name.find(pattern) >= 0:

                    keep = False
                    if op_name.startswith('Mean'):
                        # Avoid adding mean operations (in CSE)
                        if isinstance(attrs, dict) and 'keepdim' in attrs:
                            keep = attrs['keepdim'] == 'True'
                        else:
                            # In pytorch <1.9 the computational graph may be inaccurate
                            keep = i < len(self._nodes) - 1 and not self._nodes[i + 1]['param_name'].startswith('cells.')

                    elif op_name.startswith('Mul'):
                        keep = self._nodes[i - 2]['param_name'].startswith('Hard')      # CSE op

                    elif op_name.startswith('Clone'):
                        keep = self._nodes[i - 11]['param_name'].startswith('Softmax')  # MSA op

                    elif op_name.startswith('Cat') or op_name.startswith('Add'):        # Concat and Residual (Sum) ops
                        keep = len(np.where(self._Adj[:, i])[0]) > 1  # keep only if > 1 edges are incoming

                    if not keep:
                        # rewire edges from/to the to-be-removed node to its neighbors
                        for n1 in np.where(self._Adj[i, :])[0]:
                            for n2 in np.where(self._Adj[:, i])[0]:
                                if n1 != n2:
                                    self._Adj[n2, n1] = 1
                else:
                    keep = True

                if keep:
                    ind_keep.append(i)

            ind_keep = np.array(ind_keep)

            if len(ind_keep) < self._Adj.shape[0]:
                self._Adj = self._Adj[:, ind_keep][ind_keep, :]
                self._nodes = [self._nodes[i] for i in ind_keep]

        return


    def _add_virtual_edges(self, ve_cutoff=50):
        r"""
        Add virtual edges with weights equal the shortest path length between the nodes.
        :param ve_cutoff: maximum shortest path length between the nodes
        :return:
        """

        self.n_nodes = len(self._nodes)

        assert self._Adj[np.diag_indices_from(self._Adj)].sum() == 0, (
            'no loops should be in the graph', self._Adj[np.diag_indices_from(self._Adj)].sum())

        # Check that the graph is connected and all nodes reach the final output
        self._nx_graph_from_adj()
        length = nx.shortest_path(self.nx_graph, target=self.n_nodes - 1)
        for node in range(self.n_nodes):
            assert node in length, ('not all nodes reach the final node', node, self._nodes[node])

        # Check that all nodes have a path to the input
        length = nx.shortest_path(self.nx_graph, source=0)
        for node in range(self.n_nodes):
            assert node in length or self._nodes[node]['param_name'].startswith('pos_enc'), (
                'not all nodes have a path to the input', node, self._nodes[node])

        if ve_cutoff > 1:
            length = dict(nx.all_pairs_shortest_path_length(self.nx_graph, cutoff=ve_cutoff))
            for node1 in length:
                for node2 in length[node1]:
                    if length[node1][node2] > 0 and self._Adj[node1, node2] == 0:
                        self._Adj[node1, node2] = length[node1][node2]
            assert (self._Adj > ve_cutoff).sum() == 0, ((self._Adj > ve_cutoff).sum(), ve_cutoff)
        return self._Adj


    def _construct_features(self):
        r"""
        Construct pytorch tensor features for nodes and edges.
        :return:
        """

        self.n_nodes = len(self._nodes)
        self.node_feat = torch.zeros(self.n_nodes, 1, dtype=torch.long)
        self.node_info = [[] for _ in range(self.n_cells)]
        self._param_shapes = []

        primitives_dict = {op: i for i, op in enumerate(PRIMITIVES_DEEPNETS1M)}

        n_glob_avg = 0
        cell_ind = 0
        for node_ind, node in enumerate(self._nodes):

            param_name = node['param_name']
            cell_ind_ = get_cell_ind(param_name, self.n_cells)
            if cell_ind_ is not None:
                cell_ind = cell_ind_

            pos_stem = param_name.find('stem')
            pos_pos = param_name.find('pos_enc')
            if pos_stem >= 0:
                param_name = param_name[pos_stem:]
            elif pos_pos >= 0:
                param_name = param_name[pos_pos:]

            if node['module'] is not None:

                # Preprocess param_name to be consistent with the DeepNets dataset
                parts = param_name.split('.')
                for i, s in enumerate(parts):
                    if s == '_ops' and parts[i + 2] != 'op':
                        try:
                            _ = int(parts[i + 2])
                            parts.insert(i + 2, 'op')
                            param_name = '.'.join(parts)
                            break
                        except:
                            continue

                name = MODULES[type(node['module'])](node['module'], param_name)

            else:
                ind = param_name.find('Backward')
                name = MODULES[param_name[:len(param_name) if ind == -1 else ind]]
                n_glob_avg += int(name == 'glob_avg')

                if self.n_cells > 1:
                    # Add cell id to the names of pooling layers, so that they will be matched with proper modules in Network
                    if param_name.startswith('MaxPool') or param_name.startswith('AvgPool'):
                        param_name = 'cells.{}.'.format(cell_ind) + name

            sz = None
            attrs = node['attrs']
            if isinstance(attrs, dict):
                if 'size' in attrs:
                    sz = attrs['size']
                elif name.find('pool') >= 0:
                    if 'kernel_size' in attrs:
                        sz = (1, 1, *[int(a.strip('(').strip(')').strip(' ')) for a in attrs['kernel_size'].split(',')])
                    else:
                        # Pytorch 1.9+ is required to correctly extract pooling attributes, otherwise the default pooling size of 3 is used
                        sz = (1, 1, 3, 3)
            elif node['module'] is not None:
                sz = (node['module'].weight if param_name.find('weight') >= 0 else node['module'].bias).shape

            self._param_shapes.append(sz)
            self.node_feat[node_ind] = primitives_dict[name]
            if node['module'] is not None or name.find('pool') >= 0 or self._list_all_nodes:
                self.node_info[cell_ind].append(
                    (node_ind,
                     param_name if node['module'] is not None else name,
                     name,
                     sz,
                     node_ind == len(self._nodes) - 2,
                     node_ind == len(self._nodes) - 1))

        if n_glob_avg > 1:
            print(
                '\nWARNING: n_glob_avg should be 0 or 1 in most architectures, but is %d in this architecture\n' % n_glob_avg)

        self._Adj = torch.tensor(self._Adj, dtype=torch.long)

        ind = torch.nonzero(self._Adj)  # rows, cols
        self.edges = torch.cat((ind, self._Adj[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)
        return


    def _named_modules(self):
        r"""
        Helper function to automatically build the graphs.
        :return:
        """

        modules = {}
        for n, m in self.model.named_modules():
            is_w = hasattr(m, 'weight') and m.weight is not None
            is_b = hasattr(m, 'bias') and m.bias is not None
            if is_w:
                modules[n + '.weight'] = (m.weight, m)
            if is_b:
                modules[n + '.bias'] = (m.bias, m)
        return modules


    def _nx_graph_from_adj(self):
        """
        Creates NetworkX directed graph instance that is used for visualization, virtual edges and graph statistics.
        :return: nx.DiGraph
        """
        if self.nx_graph is None:
            A = self._Adj.data.cpu().numpy() if isinstance(self._Adj, torch.Tensor) else self._Adj
            A[A > 1] = 0  # remove any virtual edges for the visualization/statistics
            self.nx_graph = nx.DiGraph(A)
        return self.nx_graph


    def properties(self, undirected=True, key=('avg_degree', 'avg_path')):
        """
        Computes graph properties.
        :param undirected: ignore edge direction when computing graph properties.
        :param key: a tuple/list of graph properties to estimate.
        :return: dictionary with property names and values.
        """
        G = self._nx_graph_from_adj()
        if undirected:
            G = G.to_undirected()
        props = {}
        for prop in key:
            if prop == 'avg_degree':
                degrees = dict(G.degree())
                assert len(degrees) == self._Adj.shape[0] == self.n_nodes, 'invalid graph'
                props[prop] = sum(degrees.values()) / self.n_nodes
            elif prop == 'avg_path':
                props[prop] = nx.average_shortest_path_length(G)
            else:
                raise NotImplementedError(prop)

        return props


    def visualize(self, node_size=50, figname=None, figsize=None, with_labels=False, vis_legend=False, label_offset=0.001, font_size=10):
        r"""
        Shows the graphs/legend as in the paper using matplotlib.
        :param node_size: node size
        :param figname: file name to save the figure in the .pdf and .png formats
        :param figsize: (width, height) for a figure
        :param with_labels: show node labels (operations)
        :param vis_legend: True to only visualize the legend (graph will be ignored)
        :param label_offset: positioning of node labels when vis_legend=True
        :param font_size: font size for node labels, used only when with_labels=True
        :return:
        """

        self._nx_graph_from_adj()

        # first are conv layers, so that they have a similar color
        primitives_order = [2, 3, 4, 10, 5, 6, 11, 12, 13, 0, 1, 14, 7, 8, 9]
        assert len(PRIMITIVES_DEEPNETS1M) == len(primitives_order), 'make sure the lists correspond to each other'

        n_primitives = len(primitives_order)
        color = lambda i: cm.jet(int(np.round(255 * i / n_primitives)))
        primitive_colors = { PRIMITIVES_DEEPNETS1M[ind_org] : color(ind_new)  for ind_new, ind_org in enumerate(primitives_order) }
        # manually adjust some colors for better visualization
        primitive_colors['bias'] = '#%02x%02x%02x' % (255, 0, 255)
        primitive_colors['msa'] = '#%02x%02x%02x' % (10, 10, 10)
        primitive_colors['ln'] = '#%02x%02x%02x' % (255, 255, 0)

        node_groups = {'bn':        {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': 's'}},
                       'conv1':     {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': '^'}},
                       'bias':      {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'd'}},
                       'pos_enc':   {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 's'}},
                       'ln':        {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 's'}},
                       'max_pool':  {'style': {'edgecolors': 'k',       'linewidths': 1,    'node_shape': 'o', 'node_size': 1.75 * node_size}},
                       'glob_avg':  {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'o', 'node_size': 2 * node_size}},
                       'concat':    {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': '^'}},
                       'input':     {'style': {'edgecolors': 'k',       'linewidths': 1.5,  'node_shape': 's', 'node_size': 2 * node_size}},
                       'other':     {'style': {'edgecolors': 'gray',    'linewidths': 0.5,  'node_shape': 'o'}}}

        for group in node_groups:
            node_groups[group]['node_lst'] = []
            if 'node_size' not in node_groups[group]['style']:
                node_groups[group]['style']['node_size'] = node_size

        labels, node_colors = {}, []

        if vis_legend:
            node_feat = torch.cat((torch.tensor([n_primitives]).view(-1, 1),
                                   torch.tensor(primitives_order)[:, None]))
            param_shapes = [(3, 3, 1, 1)] + [None] * n_primitives
        else:
            node_feat = self.node_feat
            param_shapes = self._param_shapes

        for i, (x, sz) in enumerate(zip(node_feat.view(-1), param_shapes)):

            name = PRIMITIVES_DEEPNETS1M[x] if x < n_primitives else 'conv'
            labels[i] = name[:20] if x < n_primitives else 'conv_1x1'
            node_colors.append(primitive_colors[name])

            if name.find('conv') >= 0 and sz is not None and \
                    ((len(sz) == 4 and np.prod(sz[2:]) == 1) or len(sz) == 2):
                node_groups['conv1']['node_lst'].append(i)
            elif name in node_groups:
                node_groups[name]['node_lst'].append(i)
            else:
                node_groups['other']['node_lst'].append(i)

        if vis_legend:
            fig = plt.figure(figsize=(20, 3) if figsize is None else figsize)
            G = nx.DiGraph(np.diag(np.ones(n_primitives), 1))
            pos = {i: (3 * i * node_size, 0) for i in labels }
            pos_labels = { i: (x, y - label_offset) for i, (x, y) in pos.items() }
        else:
            fig = plt.figure(figsize=(10, 10) if figsize is None else figsize)
            G = self.nx_graph
            pos = nx.drawing.nx_pydot.graphviz_layout(G)
            pos_labels = pos

        for node_group in node_groups.values():
            nx.draw_networkx_nodes(G, pos,
                                   node_color=[node_colors[i] for i in node_group['node_lst']],
                                   nodelist=node_group['node_lst'],
                                   **node_group['style'])
        if with_labels:
            nx.draw_networkx_labels(G, pos_labels, labels, font_size=font_size)

        nx.draw_networkx_edges(G, pos, node_size=node_size,
                               width=0 if vis_legend else 1,
                               arrowsize=10,
                               alpha=0 if vis_legend else 1,
                               edge_color='white' if vis_legend else 'k',
                               arrowstyle='-|>')

        plt.grid(False)
        plt.axis('off')
        if figname is not None:
            plt.savefig(figname + '.pdf', dpi=fig.dpi)
            plt.savefig(figname + '.png', dpi=fig.dpi, transparent=True)
        plt.show()


def get_conv_name(module, param_name):
    if param_name.find('bias') >= 0:
        return 'bias'
    elif isinstance(module, nn.Conv2d) and module.groups > 1:
        return ('dil_conv' if min(module.dilation) > 1 else 'sep_conv')
    return 'conv'

# Supported modules
MODULES = {
            nn.Conv2d: get_conv_name,
            nn.Linear: get_conv_name,
            nn.BatchNorm2d: lambda module, param_name: 'bn',
            nn.LayerNorm: lambda module, param_name: 'ln',
            PosEnc: lambda module, param_name: 'pos_enc',
            'input': 'input',
            'Mean': 'glob_avg',
            'AdaptiveAvgPool2D': 'glob_avg',
            'MaxPool2DWithIndices': 'max_pool',
            'AvgPool2D': 'avg_pool',
            'Clone': 'msa',
            'Mul': 'cse',
            'Add': 'sum',
            'Cat': 'concat',
        }
