import random
import numpy as np

import torch
import torch.nn as nn

from model.DTFDMIL.DTFDMIL import Attention_Gated, Classifier_1fc, Attention_with_Classifier, DimReduction
from model.modules import AdaptiveGraphGenerator, SlideGCN


class SlideGCD_DTFDMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size, total_instance, num_group,
                 distill_type='AFS', numLayer_Res=0, droprate=0.):
        super().__init__()

        # DTFDMIL's parameters
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.droprate = droprate
        self.total_instance = total_instance
        self.num_group = num_group
        self.distill_type = distill_type

        # SlideGCD's parameters
        self.hidden_dim = self.feat_dim
        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.dimReduction = DimReduction(self.feat_dim, self.feat_dim, numLayer_Res=numLayer_Res)
        self.attention = Attention_Gated(self.feat_dim)
        self.subClassifier = Classifier_1fc(self.feat_dim, self.num_classes, droprate=self.droprate)
        self.attCls = Attention_with_Classifier(L=self.feat_dim, num_cls=self.num_classes, droprate=self.droprate)

        # SlideGCD's modules
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.hidden_dim))

        self.agg = AdaptiveGraphGenerator(feat_dim=self.hidden_dim)

        self.max_length = self.batch_size + self.buffer_size
        self.gcn = SlideGCN(self.hidden_dim, out_dim=self.num_classes, num_node=self.max_length)

    def forward(self, x):
        batch_size = x.shape[0]

        # DTFDMIL's workflow (AFS)
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
        logits_mlp, slide_embeddings = self.attCls(slide_pseudo_feat)

        # concatenate current mini-batch slide embedding(s) with the nodes in buffer (rehearsal)
        # as the input of the graph branch;
        x_concat = torch.cat([slide_embeddings, self.rehearsal.view((-1, slide_embeddings.shape[1]))])
        # generate (hyper)graph edges with AGG module;
        edge_index, edge_attr = self.agg(x_concat)
        # graph interaction with designed SlideGCN and get the current mini-batch's response;
        padded_x = torch.zeros((self.max_length, self.hidden_dim)).to(x_concat.device)
        padded_x[:x_concat.shape[0]] = x_concat
        logits_graph = torch.squeeze(self.gcn(padded_x, edge_index, edge_attr))[:batch_size, :]

        # update the node buffer with First-In-First-Out strategy;
        if self.training is True:
            self.update_buffer_FIFO(x_concat)

        return logits_mlp, torch.cat(sub_preds, dim=0), logits_graph

    def update_buffer_FIFO(self, x_concat):
        with torch.no_grad():
            # only need to pop out the outdated slide embeddings;
            self.rehearsal = x_concat[:self.buffer_size]
