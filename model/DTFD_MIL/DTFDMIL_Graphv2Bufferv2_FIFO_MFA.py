import os
import random
import numpy as np

import torch
import torch.nn as nn
from model.DTFD_MIL.Attention import Attention_Gated
from model.DTFD_MIL.network import Classifier_1fc, Attention_with_Classifier, DimReduction, get_cam_1d

from model.abmil_GraphBuffer_FIFO_MFA_4090Ver import DSLv2, GCN


class DTFD_MIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size, total_instance, num_group, distill_type='AFS', numLayer_Res=0, droprate=0.):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.droprate = droprate
        self.total_instance = total_instance
        self.num_group = num_group
        self.distill_type = distill_type

        self.dimReduction = DimReduction(self.feat_dim, self.feat_dim, numLayer_Res=numLayer_Res)
        self.attention = Attention_Gated(self.feat_dim)
        self.subClassifier = Classifier_1fc(self.feat_dim, self.num_classes, droprate=self.droprate)
        self.attCls = Attention_with_Classifier(L=self.feat_dim, num_cls=self.num_classes, droprate=self.droprate)

        self.dsl = DSLv2(feat_dim=self.feat_dim, k=k)
        self.max_length = self.batch_size * self.k * self.k
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

        if y is not None or not self.training:
            x, edge_index, edge_attr = self.dsl(bag_feat, self.rehearsal.view(-1, self.feat_dim))

            padded_x = torch.zeros((self.max_length, self.feat_dim)).cuda()
            padded_x[:x.shape[0]] = x
            logits_graph = torch.squeeze(self.gcn(padded_x, edge_index, edge_attr))[:B * self.k:self.k]
            assert logits_graph.shape[0] == bag_feat.shape[0]

        if self.training:
            if y is None:
                self.buffer_update_warmup(bag_feat)
            else:
                self.buffer_update(x, y)

        if y is not None or not self.training:
            return bag_pred, torch.cat(sub_preds, dim=0), logits_graph
        else:
            return bag_pred, torch.cat(sub_preds, dim=0)

    def buffer_update_warmup(self, x):
        x_concat = torch.cat([x, self.rehearsal.view((-1, self.feat_dim))])[:self.buffer_size, :].detach()
        self.rehearsal = x_concat.view((self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def buffer_update(self, x, y):
        cls_means, cls_dists = [], []
        for i in range(self.num_classes):
            cls_cache = torch.squeeze(self.rehearsal[i, :, :])
            cache_mean = torch.mean(cls_cache, dim=0).unsqueeze(dim=0)
            dist = torch.cosine_similarity(cache_mean.unsqueeze(0), cls_cache.unsqueeze(1), dim=-1).squeeze()

            cls_means.append(cache_mean)
            cls_dists.append(dist)

        for i in range(y.shape[0]):
            dist = torch.cosine_similarity(cls_means[y[i]], torch.unsqueeze(x[i], dim=0), dim=1)

            min = torch.min(cls_dists[y[i]])
            if dist.item() > min:
                self.rehearsal[y[i], torch.argmin(cls_dists[y[i]]), :] = x[i].detach()
