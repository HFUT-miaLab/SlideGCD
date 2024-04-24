import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GINConv, GENConv, DeepGCNLayer, SAGEConv

from model.PatchGCN.model_utils import *
from model.modules import DSL, GCN


class PatchGCN(torch.nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size, temp, num_layers=4, edge_agg='spatial', hidden_dim=128, dropout=0.25):
        super(PatchGCN, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = num_layers - 1
        self.feat_dim = feat_dim

        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.temp = temp

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

        self.classifier = torch.nn.Linear(hidden_dim * 4, self.num_classes)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.max_length = self.batch_size + self.buffer_size
        self.gcn = GCN(self.feat_dim, num_node=self.max_length, out_dim=self.num_classes)

        self.register_buffer('rehearsal', torch.rand(self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def forward(self, data, y=None):
        x = data.x
        edge_index = data.edge_index
        edge_latent = data.edge_latent
        B = data.y.shape[0]

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
        h_path = torch.reshape(x_, (B, 500, 512))
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, -1, -2)
        h_path = torch.bmm(F.softmax(A_path, dim=-1), h_path)
        bag_feat = self.path_rho(h_path).squeeze()
        logits_mlp = self.classifier(bag_feat)  # logits needs to be a [1 x 4] vector

        if y is not None or not self.training:  # 推理阶段 或者 非Warmup训练阶段
            x_concat = torch.cat([bag_feat, self.rehearsal.view((-1, self.feat_dim))])
            edge_index, edge_attr = self.dsl(x_concat)

            padded_x = torch.zeros((self.max_length, self.feat_dim)).cuda()
            padded_x[:bag_feat.shape[0]] = bag_feat
            logits_graph = torch.squeeze(self.gcn(padded_x, edge_index, edge_attr))[:B, :]

        if self.training:
            if y is None:  # Warmup训练阶段
                self.buffer_update_warmup(bag_feat)
                return logits_mlp
            else:  # 非Warmup训练阶段
                buffer_update_loss, reg_loss = self.buffer_update(bag_feat, y, temp=self.temp)
                return logits_mlp, logits_graph, buffer_update_loss, reg_loss
        else:  # 推理阶段
            return logits_mlp, logits_graph

    def buffer_update_warmup(self, x):
        x_concat = torch.cat([x, self.rehearsal.view((-1, self.feat_dim))])[:self.buffer_size, :].detach()
        self.rehearsal = x_concat.view((self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def buffer_update(self, x, y, temp=0.5):  # Moco's temp = 0.5; SimCLR's temp = 0.07;
        assert x.shape[0] == y.shape[0]
        # 监督对比损失计算
        cls_means = torch.mean(self.rehearsal, dim=1)

        rehearsal_label = torch.tensor([0, 1], dtype=torch.long).cuda()
        pos_position = (rehearsal_label.unsqueeze(0) == y.unsqueeze(1))

        similarity_matrix = torch.matmul(x, cls_means.T).cuda()

        positives = similarity_matrix[pos_position].view(x.shape[0], -1)
        negatives = similarity_matrix[~pos_position].view(positives.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # infoNCE_loss
        loss = nn.NLLLoss()

        # Regularization
        cls_center_sim_matrix = 1 + torch.cosine_similarity(cls_means.unsqueeze(0), cls_means.unsqueeze(1), dim=-1)
        mask = torch.eye(cls_center_sim_matrix.shape[0], dtype=torch.bool).cuda()
        reg_term = torch.sum(cls_center_sim_matrix[~mask]) / 2

        # Buffer更新
        cls_dists = []
        for i in range(self.rehearsal.shape[0]):
            cls_buffer = torch.squeeze(self.rehearsal[i, :, :])
            dist = torch.cosine_similarity(cls_means[i].unsqueeze(0), cls_buffer.unsqueeze(1), dim=-1).squeeze()

            cls_dists.append(dist)

        for i in range(y.shape[0]):
            dist = torch.cosine_similarity(cls_means[y[i]], x[i], dim=0)

            min_value, min_idx = torch.min(cls_dists[y[i]]), torch.argmin(cls_dists[y[i]])
            if dist.item() > min_value:
                self.rehearsal[y[i], torch.argmin(cls_dists[y[i]]), :] = x[i].detach()
                cls_dists[y[i]][min_idx] = dist.item()

        return loss(torch.log_softmax(logits / temp, dim=1), labels), reg_term