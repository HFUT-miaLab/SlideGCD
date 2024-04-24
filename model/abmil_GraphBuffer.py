import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_mean, scatter_max
import torch_cluster
from typing import Optional


# class AdaptiveGraphGenerator(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.in_dim = in_dim
#
#         self.scale_layer = nn.Sequential(nn.Linear(self.in_dim, self.in_dim),
#                                          nn.ReLU())
#
#         self.theta = nn.Parameter(torch.rand(1, 1))
#         self.phi = nn.Parameter(torch.rand(1, 1))
#
#     def forward(self, x):
#         x = self.scale_layer(x)
#
#         m1 = torch.tanh(x * self.theta)  # B * N * D
#         m2 = torch.tanh(x * self.phi)  # B * N * D
#         A = F.relu(torch.mm(m1, m2.transpose(-2, -1)) - torch.mm(m2, m1.transpose(-2, -1)))
#         adj = F.relu(torch.tanh(A))
#
#         # 限制边的数量
#         mask = torch.zeros_like(adj)
#         s1, t1 = adj.topk(int(0.3 * adj.shape[0]), 1)
#         mask.scatter_(1, t1, s1.fill_(1))
#         adj = adj * mask
#
#         return adj


# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha=0.01, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.Q = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.Q.data, gain=1.414)
#         self.V = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.V.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, input, adj):
#         h = torch.matmul(input, self.W)
#         q = torch.matmul(input, self.Q)
#         v = torch.matmul(input, self.V)
#
#         B = h.size()[0]
#         N = h.size()[1]
#
#         a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), q.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
#
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, v)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, out_dim, dropout, alpha=0.01, nheads=4):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
#                            range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = GraphAttentionLayer(nhid * nheads, out_dim, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)


class DSL(nn.Module):
    def __init__(self, k=4, feat_dim=512):
        super(DSL, self).__init__()

        self.k = k

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.LeakyReLU(),
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


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, out_dim):
        super().__init__()
        self.feat_dim = feat_dim
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

        self.classifier = nn.Sequential(nn.Linear(self.hid_dim, self.out_dim))

    def forward(self, x, edge_index, edge_attr, return_feat=False):
        x = self.hgc1(x, edge_index, hyperedge_attr=edge_attr)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        out = self.fc1(x)

        x = self.hgc2(x, edge_index, hyperedge_attr=edge_attr)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        out += self.fc2(x)

        logits = self.classifier(out)

        if return_feat:
            return logits, out
        else:
            return logits


class BClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, buffer_size=256):
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

        self.dsl = DSL(feat_dim=self.feat_dim)
        self.gcn = GCN(self.feat_dim // 2, self.num_classes)
        # self.agg = AdaptiveGraphGenerator(self.feat_dim)
        # self.GAT = GAT(self.feat_dim, nhid=self.feat_dim, out_dim=num_classes, dropout=0.)

        self.register_buffer('rehearsal',
                             torch.rand(self.buffer_size, self.feat_dim))
        # self.register_buffer('rehearsal',
        #                      torch.rand(self.num_classes, int(self.buffer_size / self.num_classes), self.feat_dim))
        # self.rehearsal = nn.functional.normalize(self.rehearsal, dim=1)
        # self.register_buffer('rehearsal_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, x, train=True):  # x: B * N * D
        B = x.shape[0]

        A = torch.transpose(self.attention(x), 2, 1)  # B * 1 * N
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.squeeze(torch.bmm(A, x), dim=1)  # B * D
        logits_mlp = self.bag_classifier(M)  # B * C

        x_concat = torch.cat([M, self.rehearsal.view((-1, M.shape[1]))])
        x, edge_index, edge_attr = self.dsl(x_concat)
        logits_graph = torch.squeeze(self.gcn(x, edge_index, edge_attr))[:B, :]

        if train is True:
            self.update_rehearsal_buffer(M)

        return logits_mlp, logits_graph

    def update_rehearsal_buffer(self, x):
        with torch.no_grad():
            # print(self.rehearsal.shape[0], x.shape[0])
            index = torch.LongTensor(random.sample(range(self.buffer_size), self.buffer_size - x.shape[0])).cuda()
            self.rehearsal = torch.cat([x, torch.index_select(self.rehearsal, dim=0, index=index)])
            # print(self.rehearsal.shape)


if __name__ == "__main__":
    model = BClassifier(feat_dim=512, num_classes=5)
    input = torch.rand((2, 8, 512))
    output = model(input)
    print(output.shape)

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
