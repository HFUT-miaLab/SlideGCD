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


class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class PatchGCN(torch.nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial', feat_dim=384, hidden_dim=128, dropout=0.25, n_classes=4):
        super(PatchGCN, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = num_layers-1

        self.fc = nn.Sequential(*[nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)

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
        
        return logits, A_path, None