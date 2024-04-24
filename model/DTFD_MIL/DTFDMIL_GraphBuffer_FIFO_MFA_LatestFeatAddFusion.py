import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_mean, scatter_max
import torch_cluster

from model.DTFD_MIL.Attention import Attention_Gated
from model.DTFD_MIL.network import Classifier_1fc, Attention_with_Classifier, DimReduction, get_cam_1d


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

        return x, edge_index, edge_attr

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

        out = torch.einsum('ij,j->ij', out, attn_score)

        return out


class DTFD_MIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, total_instance, num_group, distill_type='AFS', numLayer_Res=0, droprate=0.):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.droprate = droprate
        self.total_instance = total_instance
        self.num_group = num_group
        self.distill_type = distill_type

        self.dimReduction = DimReduction(self.feat_dim, self.feat_dim, numLayer_Res=numLayer_Res)
        self.attention = Attention_Gated(self.feat_dim)
        self.subClassifier = Classifier_1fc(self.feat_dim, self.num_classes, droprate=self.droprate)

        self.last_attention = Attention_Gated(L=self.feat_dim)
        self.mil_classifier = nn.Linear(self.feat_dim, self.num_classes)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.gcn = GCN(self.feat_dim, num_node=self.buffer_size, out_dim=self.num_classes)
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.feat_dim))

        self.mil_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.graph_proj = nn.Linear(self.feat_dim * 2, self.feat_dim)
        self.fusion_classifier = nn.Linear(feat_dim, self.num_classes)

    def forward(self, x):
        B = x.shape[0]
        instance_per_group = self.total_instance // self.num_group

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
        att = self.last_attention(slide_pseudo_feat)
        out_mil = torch.squeeze(torch.bmm(att, slide_pseudo_feat))  ## K x L
        logits = self.mil_classifier(out_mil)

        x_concat = torch.cat([out_mil, self.rehearsal.view((-1, slide_pseudo_feat.shape[-1]))])[:self.buffer_size, :]
        x, edge_index, edge_attr = self.dsl(x_concat)
        out_graph = torch.squeeze(self.gcn(x, edge_index, edge_attr))[:B, :]

        out_mil = self.mil_proj(out_mil)
        fusion_feat = self.graph_proj(out_graph) + out_mil
        fusion_logits = self.fusion_classifier(fusion_feat)
        # print(out_mil.shape, out_graph.shape)

        if self.training:
            self.rehearsal = x_concat.detach()

        return logits, torch.cat(sub_preds, dim=0), fusion_logits


if __name__ == '__main__':
    input = torch.rand((4, 256, 512)).cuda()
    model = DTFD_MIL(feat_dim=512, num_classes=2, k=1, buffer_size=1024, total_instance=256, num_group=2).cuda()
    logits, sub_preds, fusion_logits = model.forward(input)
    print(logits.shape, len(sub_preds), fusion_logits.shape)
