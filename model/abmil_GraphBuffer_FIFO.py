import random

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
        self.gcn = GCN(self.feat_dim, self.num_classes)
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

        x_concat = torch.cat([M, self.rehearsal.view((-1, M.shape[1]))])
        x, edge_index, edge_attr = self.dsl(x_concat)
        logits_graph = torch.squeeze(self.gcn(x, edge_index, edge_attr))[:B, :]

        if train is True:
            self.update_rehearsal_buffer(x_concat)

        return logits_mlp, logits_graph

    def update_rehearsal_buffer(self, x_concat):
        with torch.no_grad():
            self.rehearsal = x_concat[:self.buffer_size]


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
