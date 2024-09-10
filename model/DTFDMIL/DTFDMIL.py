# Code integrated from https://github.com/hrzhang1123/DTFD-MIL
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: B X N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN

        if isNorm:
            A = F.softmax(A, dim=-1)  # softmax over N
        return A  ### K x N


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.squeeze(torch.bmm(AA, x)) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred, afeat


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


class DTFDMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, total_instance, num_group, distill_type='AFS', numLayer_Res=0, droprate=0.):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.droprate = droprate
        self.total_instance = total_instance
        self.num_group = num_group
        self.distill_type = distill_type

        self.dimReduction = DimReduction(self.feat_dim, self.feat_dim, numLayer_Res=numLayer_Res)
        self.attention = Attention_Gated(self.feat_dim)
        self.subClassifier = Classifier_1fc(self.feat_dim, self.num_classes, droprate=self.droprate)
        self.attCls = Attention_with_Classifier(L=self.feat_dim, num_cls=self.num_classes, droprate=self.droprate)

    def forward(self, x):
        instance_per_group = self.total_instance // self.num_group

        feat_index = list(range(x.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.num_group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        pseudo_feat = []
        sub_preds = []

        for tindex in index_chunk_list:
            subFeat = torch.index_select(x, dim=0, index=torch.LongTensor(tindex).cuda())

            tmidFeat = self.dimReduction(subFeat)
            tAA = self.attention(tmidFeat).squeeze(0)

            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.subClassifier(tattFeat_tensor)  ### 1 x 2
            sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(self.subClassifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill_type == 'MaxMinS':
                pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill_type == 'MaxS':
                pseudo_feat.append(max_inst_feat)
            elif self.distill_type == 'AFS':
                pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(pseudo_feat, dim=0)
        bag_prediction = self.attCls(slide_pseudo_feat)

        return bag_prediction, torch.cat(sub_preds, dim=0)
