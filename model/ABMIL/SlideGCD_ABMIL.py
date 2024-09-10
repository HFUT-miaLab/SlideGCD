import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import AdaptiveGraphGenerator, SlideGCN


class SlideGCD_ABMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size):
        super(SlideGCD_ABMIL, self).__init__()

        # ABMIL's parameters
        self.L = 500
        self.D = 128
        self.K = 1
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # SlideGCD's parameters
        self.hidden_dim = self.L  # Dim of the slide embeddings
        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # ABMIL's modules
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

        # SlideGCD's modules
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.hidden_dim))

        self.agg = AdaptiveGraphGenerator(feat_dim=self.hidden_dim)

        self.max_length = self.batch_size + self.buffer_size
        self.gcn = SlideGCN(self.hidden_dim, out_dim=self.num_classes, num_node=self.max_length)

    def forward(self, x):  # x: B * N * D
        batch_size = x.shape[0]

        # ABMIL's workflow
        x = self.linear(x)
        A = torch.transpose(self.attention(x), 2, 1)  # B * 1 * N
        A = F.softmax(A, dim=2)  # softmax over N
        slide_embeddings = torch.squeeze(torch.bmm(A, x), dim=1)  # B * D
        logits_mlp = self.bag_classifier(slide_embeddings)  # B * C

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
