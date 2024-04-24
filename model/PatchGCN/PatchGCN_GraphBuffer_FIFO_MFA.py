from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GINConv, GENConv, DeepGCNLayer, SAGEConv
from torch_geometric.nn.pool import global_mean_pool, SAGPooling
from torch_geometric.transforms.normalize_features import NormalizeFeatures

from model.PatchGCN.model_utils import *

from model.abmil_GraphBuffer_FIFO_MFA_4090Ver import DSL, GCN


class PatchGCN(torch.nn.Module):
    def __init__(self, k, buffer_size, num_layers=4, edge_agg='spatial', feat_dim=384, hidden_dim=128, dropout=0.25,
                 n_classes=4):
        super(PatchGCN, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = num_layers - 1
        self.feat_dim = feat_dim
        self.buffer_size = buffer_size

        self.fc = nn.Sequential(*[nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * 4, D=hidden_dim * 4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim * 4, n_classes)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.gcn = GCN(self.feat_dim, num_node=self.buffer_size, out_dim=n_classes)
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.feat_dim))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_latent = data.edge_latent
        batch_size = data.y.shape[0]

        if self.edge_agg == 'spatial':
            edge_index = edge_index
        elif self.edge_agg == 'latent':
            edge_index = edge_latent
        else:
            raise NotImplementedError
        edge_attr = None

        x = self.fc(x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], dim=-1)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], dim=-1)

        # print(torch.reshape(x_, (64, 500, 512))[0])
        # h_path = x_
        h_path = torch.reshape(x_, (batch_size, 500, 512))
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, -1, -2)
        h_path = torch.bmm(F.softmax(A_path, dim=-1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        probs = torch.sigmoid(logits)

        x_concat = torch.cat([h, self.rehearsal.view((-1, self.feat_dim))])[:self.buffer_size, :]
        x, edge_index, edge_attr = self.dsl(x_concat)
        logits_graph = torch.squeeze(self.gcn(x, edge_index, edge_attr))[:batch_size, :]

        if self.training:
            self.rehearsal = x_concat.detach()

        return logits, A_path, logits_graph
