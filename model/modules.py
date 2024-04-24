# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_scatter import scatter_mean, scatter_max
import torch_cluster
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm


class DSL(nn.Module):
    def __init__(self, k=4, feat_dim=512):
        super(DSL, self).__init__()

        self.k = k

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        _x = self.fc(x)

        edge_index = self.graph_edge(x=_x, k=self.k)
        edge_attr = scatter_mean(src=x[edge_index[0]], index=edge_index[1], dim=0)

        return edge_index, edge_attr

    def graph_edge(self, x: torch.Tensor, k: int):
        batch = torch.zeros(x.shape[0], dtype=torch.long).cuda()
        edge_index = torch_cluster.knn(x, x, k, batch_x=batch, batch_y=batch, cosine=True)

        row, col = edge_index[1], edge_index[0]

        return torch.stack([row, col], dim=0)


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
