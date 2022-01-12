# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP


class GatedGNN(nn.Module):
    r"""
    Gated Graph Neural Network based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    Performs node feature propagation according to Eq. 3 and 4 in the paper.
    """
    def __init__(self,
                 in_features=32,
                 ve=False,
                 T=1):
        """
        Initializes Gated Graph Neural Network.
        :param in_features: how many features in each node.
        :param ve: use virtual edges defined according to Eq. 4 in the paper.
        :param T: number of forward+backward graph traversal steps.
        """
        super(GatedGNN, self).__init__()
        self.in_features = in_features
        self.hid = in_features
        self.ve = ve
        self.T = T
        self.mlp = MLP(in_features, hid=( (self.hid // 2) if ve else self.hid, self.hid))
        if ve:
            self.mlp_ve = MLP(in_features, hid=(self.hid // 2, self.hid))

        self.gru = nn.GRUCell(self.hid, self.hid)  # shared across all nodes/cells in a graph


    def forward(self, x, edges, node_graph_ind):
        r"""
        Updates node features by sequentially traversing the graph in the forward and backward directions.
        :param x: (N, C) node features, where N is the total number of nodes in a batch of B graphs, C is node feature dimensionality.
        :param edges: (M, 4) tensor of edges, where M is the total number of edges;
                       first column in edges is the row indices of edges,
                       second column in edges is the column indices of edges,
                       third column in edges is the shortest path distance between the nodes,
                       fourth column in edges is the graph indices (from 0 to B-1) within a batch for each edge.
        :param node_graph_ind: (N,) tensor of graph indices (from 0 to B-1) within a batch for each node.
        :return: updated (N, C) node features.
        """

        assert x.dim() == 2 and edges.dim() == 2 and edges.shape[1] == 4, (x.shape, edges.shape)
        n_nodes = torch.unique(node_graph_ind, return_counts=True)[1]

        B, C = len(n_nodes), x.shape[1]  # batch size, features

        ve, edge_graph_ind = edges[:, 2], edges[:, 3]

        assert n_nodes.sum() == len(x), (n_nodes.sum(), x.shape)

        is_1hop = ve == 1
        if self.ve:
            ve = ve.view(-1, 1)   # according to Eq. 4 in the paper

        traversal_orders = [1, 0]  # forward, backward

        edge_offset = torch.cumsum(F.pad(n_nodes[:-1], (1, 0)), 0)[edge_graph_ind]
        node_inds = torch.cat([torch.arange(n) for n in n_nodes]).to(x.device).view(-1, 1)

        # Parallelize computation of indices and masks of one/all hops
        # This will slightly speed up the operations in the main loop
        # But indexing of the GPU tensors (used in the main loop) for some reason remains slow, see
        # https://github.com/pytorch/pytorch/issues/29973 for more info
        all_nodes = torch.arange(edges[:, 1].max() + 1, device=x.device)
        masks_1hop, masks_all = {}, {}
        for order in traversal_orders:
            masks_all[order] = edges[:, order].view(1, -1) == all_nodes.view(-1, 1)
            masks_1hop[order] = masks_all[order] & is_1hop.view(1, -1)
        mask2d = node_inds == all_nodes.view(1, -1)
        edge_graph_ind = edge_graph_ind.view(-1, 1).expand(-1, C)

        hx = x  # initial hidden node features

        # Main loop
        for t in range(self.T):
            for order in traversal_orders:  # forward, backward
                start = edges[:, 1 - order] + edge_offset                           # node indices from which the message will be passed further
                for node in (all_nodes if order else torch.flipud(all_nodes)):

                    # Compute the message by aggregating features from neighbors
                    e_1hop = torch.nonzero(masks_1hop[order][node, :]).view(-1)
                    m = self.mlp(hx[start[e_1hop]])                                 # transform node features of all 1-hop neighbors
                    m = torch.zeros(B, C, dtype=m.dtype, device=m.device).scatter_add_(0, edge_graph_ind[e_1hop], m)     # sum the transformed features into a (B,C) tensor
                    if self.ve:
                        e = torch.nonzero(masks_all[order][node, :]).view(-1)       # virtual edges connected to node
                        m_ve = self.mlp_ve(hx[start[e]]) / ve[e].to(m)              # transform node features of all ve-hop neighbors
                        m = m.scatter_add_(0, edge_graph_ind[e], m_ve)              # sum m and m_ve according to Eq. 4 in the paper

                    # Udpate node hidden states in parallel for a batch of graphs
                    ind = torch.nonzero(mask2d[:, node]).view(-1)
                    if B > 1:
                        m = m[node_graph_ind[ind]]
                    hx[ind] = self.gru(m, hx[ind]).to(hx)  # 'to(hx)' is to make automatic mixed precision work

        return hx
