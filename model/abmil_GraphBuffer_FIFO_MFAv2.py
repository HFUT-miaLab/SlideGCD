import math
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
        )

    def forward(self, x):
        x = self.fc(x)

        batch = torch.zeros(x.shape[0], dtype=torch.long).cuda()
        edge_index = self.graph_edge(x=x, k=self.k, batch=batch, cosine=True)
        edge_attr = scatter_mean(src=x[edge_index[0]], index=edge_index[1], dim=0)

        return edge_index, edge_attr

    def graph_edge(self, x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None, cosine: bool = False):
        edge_index = torch_cluster.knn(x, x, k, batch_x=batch, batch_y=batch, cosine=cosine)

        row, col = edge_index[1], edge_index[0]

        return torch.stack([row, col], dim=0)


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, num_node, out_dim, dropout=0.1, center_momentum=0.9):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_node = num_node
        self.out_dim = out_dim
        self.hid_dim = self.feat_dim // 2
        self.center_momentum = center_momentum

        self.hgc1 = HypergraphConv(self.feat_dim, self.feat_dim, use_attention=True, dropout=dropout)
        self.hgc2 = HypergraphConv(self.feat_dim, self.feat_dim, use_attention=True, dropout=dropout)

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

        self.norm3 = nn.LayerNorm(self.feat_dim + self.hid_dim * 2)
        self.register_buffer('center', torch.zeros(self.feat_dim + self.hid_dim * 2))
        self.classifier = nn.Sequential(nn.Linear(self.feat_dim + self.hid_dim * 2, self.out_dim))

    def forward(self, x, edge_index, edge_attr, return_feat=False):
        _x = self.hgc1(self.norm1(x), edge_index, hyperedge_attr=edge_attr)
        _x = F.leaky_relu(_x)
        out_1 = self.fc1(_x)

        _x = self.hgc2(self.norm2(_x), edge_index, hyperedge_attr=edge_attr)
        _x = F.leaky_relu(_x)
        out_2 = self.fc2(_x)

        out = torch.cat([x, out_1, out_2], dim=1)  # N * _D
        self.norm3(out)
        attn_score = self.attention(out.T).squeeze()  # _D * 1
        # attn_score = torch.squeeze(torch.sigmoid(attn_score))  # _D
        attn_score = torch.sigmoid(attn_score)
        attn_score = torch.squeeze(attn_score) - self.center  # _D

        logits = torch.einsum('ij,j->ij', out, attn_score)
        logits = self.classifier(logits)  # N * D

        self.update_center(attn_score)
        if return_feat:
            return logits, out
        else:
            return logits

    @torch.no_grad()
    def update_center(self, attn_score):
        self.center = self.center * self.center_momentum + torch.mean(attn_score) * (1 - self.center_momentum)


class BClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, k=4, buffer_size=256, dropout=0.1):
        super(BClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.k = k

        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, 1)
        )

        self.bag_classifier = nn.Sequential(
            nn.Linear(self.feat_dim, num_classes)
        )

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.gcn = GCN(self.feat_dim, num_node=self.buffer_size, out_dim=self.num_classes, dropout=dropout)

        self.register_buffer('rehearsal',
                             torch.rand(self.buffer_size, self.feat_dim))

    def forward(self, x, train=True):  # x: B * N * D
        B = x.shape[0]

        A = torch.transpose(self.attention(x), 2, 1)  # B * 1 * N
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.squeeze(torch.bmm(A, x), dim=1)  # B * D
        logits_mlp = self.bag_classifier(M)  # B * C

        x_concat = torch.cat([M, self.rehearsal.view((-1, M.shape[1]))])[:self.buffer_size, :]
        edge_index, edge_attr = self.dsl(x_concat)
        logits_graph = torch.squeeze(self.gcn(x_concat, edge_index, edge_attr))[:B, :]

        if train is True:
            self.rehearsal = x_concat.detach()

        return logits_mlp, logits_graph
