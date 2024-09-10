import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.TransMIL.TransMIL import TransLayer, PPEG
from model.modules import AdaptiveGraphGenerator, SlideGCN


class SlideGCD_TransMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size):
        super(SlideGCD_TransMIL, self).__init__()

        # TranMIL's parameters
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.hidden_dim = 512
        # SlideGCD's parameters
        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.pos_layer = PPEG(dim=self.hidden_dim)
        self._fc1 = nn.Sequential(nn.Linear(self.feat_dim, self.hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.n_classes = self.num_classes
        self.layer1 = TransLayer(dim=self.hidden_dim)
        self.layer2 = TransLayer(dim=self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self._fc2 = nn.Linear(self.hidden_dim, self.n_classes)

        # SlideGCD's modules
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.hidden_dim))

        self.agg = AdaptiveGraphGenerator(feat_dim=self.hidden_dim)

        self.max_length = self.batch_size + self.buffer_size
        self.gcn = SlideGCN(self.hidden_dim, out_dim=self.num_classes, num_node=self.max_length)

    def forward(self, x):
        batch_size = x.shape[0]

        # TransMIL's workflow
        h = self._fc1(x)

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)  # [B, N, 512]
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        h = self.layer2(h)  # [B, N, 512]
        slide_embeddings = self.norm(h)[:, 0]
        logits_mlp = self._fc2(slide_embeddings)  # [B, n_classes]

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

        return logits_mlp, logits_graph

    def update_buffer_FIFO(self, x_concat):
        with torch.no_grad():
            # only need to pop out the outdated slide embeddings;
            self.rehearsal = x_concat[:self.buffer_size]
