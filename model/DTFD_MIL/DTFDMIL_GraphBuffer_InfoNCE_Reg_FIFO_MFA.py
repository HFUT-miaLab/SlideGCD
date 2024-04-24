# coding:utf-8
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
from typing import Optional
from torch_scatter import scatter_mean, scatter_max
import torch_cluster
from model.DTFD_MIL.Attention import Attention_Gated
from model.DTFD_MIL.network import Classifier_1fc, Attention_with_Classifier, DimReduction, get_cam_1d

from model.abmil_GraphBuffer_FIFO_MFA_4090Ver import GCN


class DTFD_MIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, batch_size, buffer_size, temp, total_instance, num_group, distill_type='AFS', numLayer_Res=0, droprate=0.):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.temp = temp

        self.droprate = droprate
        self.total_instance = total_instance
        self.num_group = num_group
        self.distill_type = distill_type

        self.dimReduction = DimReduction(self.feat_dim, self.feat_dim, numLayer_Res=numLayer_Res)
        self.attention = Attention_Gated(self.feat_dim)
        self.subClassifier = Classifier_1fc(self.feat_dim, self.num_classes, droprate=self.droprate)
        self.attCls = Attention_with_Classifier(L=self.feat_dim, num_cls=self.num_classes, droprate=self.droprate)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.max_length = self.batch_size + self.buffer_size
        self.gcn = GCN(self.feat_dim, num_node=self.max_length, out_dim=self.num_classes)

        self.register_buffer('rehearsal', torch.rand(self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def forward(self, x, y=None):
        B = x.shape[0]

        feat_index = list(range(x.shape[1]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.num_group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        pseudo_feat = []
        sub_preds = []

        for tindex in index_chunk_list:
            subFeat = torch.index_select(x, dim=1, index=torch.LongTensor(tindex).cuda())

            tmidFeat = self.dimReduction(subFeat)
            tAA = self.attention(tmidFeat).squeeze(-2)
            tattFeats = torch.einsum('bns,bn->bns', tmidFeat, tAA)  ### n x fs
            af_inst_feat = torch.sum(tattFeats, dim=-2)  # B x 1 x fs
            tPredict = self.subClassifier(af_inst_feat)  # 1 x 2
            sub_preds.append(tPredict)

            if self.distill_type == 'AFS':
                pseudo_feat.append(torch.unsqueeze(af_inst_feat, dim=1))

        slide_pseudo_feat = torch.cat(pseudo_feat, dim=1)  # B x num_group x fs
        bag_pred, bag_feat = self.attCls(slide_pseudo_feat)

        if y is not None or not self.training:  # 推理阶段 或者 非Warmup训练阶段
            x_concat = torch.cat([bag_feat, self.rehearsal.view((-1, self.feat_dim))])
            edge_index, edge_attr = self.dsl(x_concat)

            padded_x = torch.zeros((self.max_length, self.feat_dim)).cuda()
            padded_x[:bag_feat.shape[0]] = bag_feat
            logits_graph = torch.squeeze(self.gcn(padded_x, edge_index, edge_attr))[:B, :]

        if self.training:
            if y is None:  # Warmup训练阶段
                self.buffer_update_warmup(bag_feat)
                return bag_pred, torch.cat(sub_preds, dim=0)
            else:  # 非Warmup训练阶段
                buffer_update_loss, reg_loss = self.buffer_update(bag_feat, y, temp=self.temp)
                return bag_pred, torch.cat(sub_preds, dim=0), logits_graph, buffer_update_loss, reg_loss
        else:  # 推理阶段
            return bag_pred, torch.cat(sub_preds, dim=0), logits_graph

    def buffer_update_warmup(self, x):
        x_concat = torch.cat([x, self.rehearsal.view((-1, self.feat_dim))])[:self.buffer_size, :].detach()
        self.rehearsal = x_concat.view((self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def buffer_update(self, x, y, temp=0.5):  # Moco's temp = 0.5; SimCLR's temp = 0.07;
        assert x.shape[0] == y.shape[0]
        # 监督对比损失计算
        cls_means = torch.mean(self.rehearsal, dim=1)

        rehearsal_label = torch.tensor([i for i in range(self.num_classes)], dtype=torch.long).cuda()
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


if __name__ == '__main__':
    rehearsal = torch.rand((2, 10, 5))
    rehearsal_label = torch.tensor([0, 1], dtype=torch.long)
    x = torch.rand((3, 5))
    y = torch.tensor([1, 0, 0], dtype=torch.long)
    e = 0.1

    cls_means = torch.mean(rehearsal, dim=1)
    cls_center_sim_matrix = 1 + torch.cosine_similarity(cls_means.unsqueeze(0), cls_means.unsqueeze(1), dim=-1)
    mask = torch.eye(cls_center_sim_matrix.shape[0], dtype=torch.bool).cuda()
    reg_term = torch.sum(cls_center_sim_matrix[~mask]) / 2
    print(reg_term)

    # pos_matrix = (rehearsal_label.unsqueeze(0) == y.unsqueeze(1))
    # print(pos_matrix)
    #
    # cls_means = torch.mean(rehearsal, dim=1)
    # similarity_matrix = torch.matmul(x, cls_means.T)
    # print(similarity_matrix)
    # positives = similarity_matrix[pos_matrix].view(x.shape[0], -1)
    # negatives = similarity_matrix[~pos_matrix].view(positives.shape[0], -1)
    # print(positives, negatives)
    # logits = torch.cat([positives, negatives], dim=1)
    # labels = torch.zeros(logits.shape[0], dtype=torch.long)
    # print(logits, labels)
    #
    # # loss = 0
    # # for i in range(positives.shape[0]):
    # #     loss -= torch.log(torch.exp(positives[i]) / (torch.exp(positives[i]) + torch.exp(negatives[i])))
    # # loss /= positives.shape[0]
    # # print(loss)
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # print(criterion(logits, labels))
    # print(torch.log_softmax(logits / 1, dim=1), labels)
    # criterion = torch.nn.NLLLoss()
    # print(criterion(torch.log_softmax(logits / 1, dim=1), labels))
