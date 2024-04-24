import os
import random
import numpy as np

import torch
import torch.nn as nn
from model.DTFD_MIL.Attention import Attention_Gated
from model.DTFD_MIL.network import Classifier_1fc, Attention_with_Classifier, DimReduction, get_cam_1d

from model.abmil_GraphBuffer_FIFO_MFA_4090Ver import DSL, GCN


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
        self.attCls = Attention_with_Classifier(L=self.feat_dim, num_cls=self.num_classes, droprate=self.droprate)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.gcn = GCN(self.feat_dim, num_node=self.buffer_size, out_dim=self.num_classes)
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.feat_dim))

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
        bag_pred, bag_feat = self.attCls(slide_pseudo_feat)

        x_concat = torch.cat([bag_feat, self.rehearsal.view((-1, slide_pseudo_feat.shape[-1]))])[:self.buffer_size, :]
        x, edge_index, edge_attr = self.dsl(x_concat)
        logits_graph = torch.squeeze(self.gcn(x, edge_index, edge_attr))[:B, :]

        if self.training:
            self.rehearsal = x_concat.detach()

        return bag_pred, torch.cat(sub_preds, dim=0), logits_graph



