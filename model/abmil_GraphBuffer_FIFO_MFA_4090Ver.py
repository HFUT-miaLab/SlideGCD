import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_mean, scatter_max
import torch_cluster
from typing import Optional


class DSL(nn.Module):
    def __init__(self, k=4, feat_dim=512):
        super(DSL, self).__init__()

        self.k = k

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.fc(x)

        batch = torch.zeros(x.shape[0], dtype=torch.long).cuda()
        edge_index = self.graph_edge(x=x, k=self.k, batch=batch, cosine=True)
        edge_attr = scatter_mean(src=x[edge_index[0]], index=edge_index[1], dim=0)

        return x, edge_index, edge_attr

    def graph_edge(self, x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None, cosine: bool = False):
        edge_index = torch_cluster.knn(x, x, k, batch_x=batch, batch_y=batch, cosine=cosine)

        row, col = edge_index[1], edge_index[0]

        return torch.stack([row, col], dim=0)


# class DSLv2(nn.Module):
#     def __init__(self, k=4, feat_dim=512):
#         super(DSLv2, self).__init__()
#
#         self.k = k
#
#         self.fc = nn.Sequential(
#             nn.Linear(feat_dim, feat_dim // 2),
#             nn.LeakyReLU(),
#             nn.Linear(feat_dim // 2, feat_dim),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x, rehearsal):
#         batch_size = x.shape[0]
#
#         x = torch.cat([x, rehearsal], dim=0)
#         _x = self.fc(x)
#         print(_x.shape)
#         edge_index = self.graph_edge(x=_x, k=self.k, batch_size=batch_size)
#         edge_attr = scatter_mean(src=x[edge_index[0]], index=edge_index[1], dim=0)
#
#         # reorder_x = torch.index_select(x, dim=0, index=edge_index[0])
#         # edge_index[0] = torch.from_numpy(np.asarray([i for i in range(reorder_x.shape[0])]))
#         return x, edge_index, edge_attr
#
#     def graph_edge(self, x: torch.Tensor, k: int, batch_size: int):
#         batch = torch.zeros(x.shape[0], dtype=torch.long).cuda()
#         edge_index = torch_cluster.knn(x, x, k, batch_x=batch, batch_y=batch, cosine=True)
#         hyperedge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
#         print(hyperedge_index.shape)
#
#         # 与当前batch的样本属于同一个超边的关联缓存（含自身）（0+1-hop邻居）
#         neighbors_1_hop_idx = torch.zeros((batch_size * self.k), dtype=torch.long)
#         for i in range(batch_size):
#             index = torch.where(hyperedge_index[1] == i)[0]
#             neighbors_1_hop_idx[i * self.k:(i + 1) * self.k] = index
#         # 与1-hop邻居同属于一个超边的关联缓存（含自身）（0+1+2-hop邻居）
#         neighbors_2_hop_idx = torch.zeros((neighbors_1_hop_idx.shape[0] * self.k), dtype=torch.long)
#         for i in range(neighbors_1_hop_idx.shape[0]):
#             index = torch.where(hyperedge_index[1, :] == i)[0]
#             neighbors_2_hop_idx[i * self.k:(i + 1) * self.k] = index
#         subgraph_hyperedge_index = torch.index_select(hyperedge_index, dim=1, index=neighbors_2_hop_idx.cuda())
#
#         return subgraph_hyperedge_index


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, num_node, out_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_node = num_node
        self.out_dim = out_dim
        self.hid_dim = self.feat_dim // 2

        self.hgc1 = HypergraphConv(self.feat_dim, self.feat_dim, use_attention=True)
        self.hgc2 = HypergraphConv(self.feat_dim, self.feat_dim, use_attention=True)

        self.norm1 = GraphNorm(self.feat_dim)
        self.norm2 = GraphNorm(self.feat_dim)

        self.fc1 = nn.Sequential(nn.Linear(self.feat_dim, self.hid_dim),
                                 nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.feat_dim, self.hid_dim),
                                 nn.LeakyReLU())

        self.attention = nn.Sequential(
            nn.Linear(self.num_node, self.num_node),
            nn.ReLU(),
            nn.Linear(self.num_node, 1)
        )

        self.classifier = nn.Sequential(nn.Linear(self.feat_dim + self.hid_dim * 2, self.out_dim))

    def forward(self, x, edge_index, edge_attr, return_feat=False):
        _x = self.hgc1(x, edge_index, hyperedge_attr=edge_attr)
        _x = self.norm1(_x)
        _x = F.leaky_relu(_x)
        out_1 = self.fc1(_x)

        _x = self.hgc2(_x, edge_index, hyperedge_attr=edge_attr)
        _x = self.norm2(_x)
        _x = F.leaky_relu(_x)
        out_2 = self.fc2(_x)

        out = torch.cat([x, out_1, out_2], dim=1)  # N * _D
        attn_score = self.attention(out.T)  # _D * 1
        # attn_score = torch.squeeze(torch.sigmoid(attn_score))  # _D
        attn_score = torch.sigmoid(attn_score)
        attn_score = torch.squeeze(attn_score) - torch.mean(attn_score)  # _D

        logits = torch.einsum('ij,j->ij', out, attn_score)
        logits = self.classifier(logits)  # N * D

        if return_feat:
            return logits, out
        else:
            return logits


class BClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size=256):
        super(BClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.buffer_size = buffer_size
        self.num_classes = num_classes

        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, 1)
        )

        self.bag_classifier = nn.Sequential(
            nn.Linear(self.feat_dim, num_classes)
        )

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.gcn = GCN(self.feat_dim, num_node=self.buffer_size, out_dim=self.num_classes)
        # self.agg = AdaptiveGraphGenerator(self.feat_dim)
        # self.GAT = GAT(self.feat_dim, nhid=self.feat_dim, out_dim=num_classes, dropout=0.)

        self.register_buffer('rehearsal',
                             torch.rand(self.buffer_size, self.feat_dim))

    def forward(self, x, train=True):  # x: B * N * D
        B = x.shape[0]

        A = torch.transpose(self.attention(x), 2, 1)  # B * 1 * N
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.squeeze(torch.bmm(A, x), dim=1)  # B * D
        logits_mlp = self.bag_classifier(M)  # B * C

        x_concat = torch.cat([M, self.rehearsal.view((-1, M.shape[1]))])[:self.buffer_size, :]
        x, edge_index, edge_attr = self.dsl(x_concat)
        logits_graph = torch.squeeze(self.gcn(x, edge_index, edge_attr))[:B, :]

        if train is True:
            self.rehearsal = x_concat.detach()

        return logits_mlp, logits_graph

    # def update_rehearsal_buffer(self, x_concat):
    #     with torch.no_grad():
    #         self.rehearsal = x_concat[:self.buffer_size]


if __name__ == "__main__":
    batch_size = 64
    k = 12
    max_length = batch_size + 3072

    dsl_module = DSL(k=k).cuda()

    dslv2_module = DSLv2(k=k).cuda()
    gcn = GCN(feat_dim=512, num_node=max_length, out_dim=256).cuda()

    start = time.time()
    for i in range(10):
        input_x = torch.rand((batch_size, 512)).cuda()
        input_rehearsal = torch.rand((3072, 512)).cuda()

        # xv1, edge_indexv1, edge_attrv1 = dsl_module(x=torch.cat([input_x, input_rehearsal], dim=0)[:3072, :])
        # gcn(xv1, edge_indexv1, edge_attrv1)
        xv2, edge_indexv2, edge_attrv2 = dslv2_module(x=input_x, rehearsal=input_rehearsal)
        print(xv2.shape, edge_indexv2.shape, edge_attrv2.shape)
        # # print(edge_indexv2)
        _x = torch.zeros((max_length, 512)).cuda()
        _x[:xv2.shape[0]] = xv2
        print(_x.shape)
        out = gcn(xv2, edge_indexv2, edge_attrv2)[:batch_size, :]
        print(out.shape)
    print(time.time() - start)

    # agg = AdaptiveGraphGenerator(in_dim=512)

    # input = torch.rand((2, 8, 512))
    # print(input.shape)
    # adj = agg(input)
    # print(adj.shape)
    # print(adj)
    #
    # gal = GraphAttentionLayer(in_features=512, out_features=256, dropout=0.0)
    # output = gal(input, adj)
    # print(output.shape)
