import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm

from model.modules import DSL, GCN


class BClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size, temp):
        super(BClassifier, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.temp = temp

        self.linear = nn.Sequential(
            nn.Linear(self.feat_dim, self.L),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.bag_classifier = nn.Sequential(
            nn.Linear(self.L, num_classes)
        )

        self.dsl = DSL(feat_dim=self.L, k=k)
        self.max_length = self.batch_size + self.buffer_size
        self.gcn = GCN(self.L, num_node=self.max_length, out_dim=self.num_classes)

        self.register_buffer('rehearsal', torch.rand(self.num_classes, self.buffer_size // self.num_classes, self.L))

    def forward(self, x, y=None):  # x: B * N * D
        B = x.shape[0]

        H = self.linear(x)
        A = torch.transpose(self.attention(H), 2, 1)  # B * 1 * N
        A = F.softmax(A, dim=2)  # softmax over N

        bag_feat = torch.squeeze(torch.bmm(A, H), dim=1)  # B * D
        logits_mlp = self.bag_classifier(bag_feat)  # B * C

        if y is not None or not self.training:  # 推理阶段 或者 非Warmup训练阶段
            x_concat = torch.cat([bag_feat, self.rehearsal.view((-1, self.L))])
            edge_index, edge_attr = self.dsl(x_concat)

            padded_x = torch.zeros((self.max_length, self.L)).cuda()
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
        x_concat = torch.cat([x, self.rehearsal.view((-1, self.L))])[:self.buffer_size, :].detach()
        self.rehearsal = x_concat.view((self.num_classes, self.buffer_size // self.num_classes, self.L))

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