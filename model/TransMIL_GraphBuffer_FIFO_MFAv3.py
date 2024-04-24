import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_mean, scatter_max
import torch_cluster
from typing import Optional

from model.PatchGCN.PatchGCN_GraphBuffer_FIFO_MFAv3 import GCN

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


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


class TransMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size=256):
        super(TransMIL, self).__init__()
        self.feat_dim = feat_dim
        self.buffer_size = buffer_size
        self.num_classes = num_classes

        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(self.feat_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = self.num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.gcn = GCN(self.feat_dim, num_node=self.buffer_size, out_dim=self.n_classes)
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.feat_dim))

    def forward(self, x):
        h = x  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits_mlp = self._fc2(h)  # [B, n_classes]

        x_concat = torch.cat([h, self.rehearsal.view((-1, h.shape[1]))])[:self.buffer_size, :]
        x, edge_index, edge_attr = self.dsl(x_concat)
        logits_graph, logits_distill = self.gcn(x, edge_index, edge_attr, B)

        if self.training:
            self.rehearsal = x_concat.detach()

        return logits_mlp, logits_graph, logits_distill
