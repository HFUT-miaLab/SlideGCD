import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import AdaptiveGraphGenerator_LC, SlideGCN


class SlideGCD_ABMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size, temp_factor):
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
        self.buffer_size = buffer_size - buffer_size % self.num_classes  # Make the buffer size divisible by the number of categories
        self.batch_size = batch_size
        self.temp_factor = temp_factor

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
        # Class-aware node buffer
        self.register_buffer('rehearsal', torch.rand(self.num_classes, self.buffer_size // self.num_classes, self.hidden_dim))

        self.agg = AdaptiveGraphGenerator_LC(feat_dim=self.hidden_dim, k=self.k)

        self.max_length = self.batch_size + self.buffer_size
        self.gcn = SlideGCN(self.hidden_dim, out_dim=self.num_classes, num_node=self.max_length)

    def forward(self, x, y=None):  # x: B * N * D
        batch_size = x.shape[0]

        # ABMIL's workflow
        x = self.linear(x)
        A = torch.transpose(self.attention(x), 2, 1)  # B * 1 * N
        A = F.softmax(A, dim=2)  # softmax over N
        slide_embeddings = torch.squeeze(torch.bmm(A, x), dim=1)  # B * D
        logits_mlp = self.bag_classifier(slide_embeddings)  # B * C

        if y is not None or not self.training:  # non-warmup stage or inference stage
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
            if y is None:  # warmup stage
                self.update_buffer_FIFO(slide_embeddings)
                return logits_mlp
            else:  # formal training stage
                buffer_update_loss, reg_term = self.buffer_update_Contrastive(slide_embeddings, y, temp=self.temp_factor)
                return logits_mlp, logits_graph, buffer_update_loss, reg_term
        else:  # inference stage
            return logits_mlp, logits_graph

    def update_buffer_FIFO(self, x):
        with torch.no_grad():
            # only need to pop out the outdated slide embeddings;
            x_concat = torch.cat([x, self.rehearsal.view((-1, self.L))])[:self.buffer_size, :].detach()
            self.rehearsal = x_concat.view((self.num_classes, self.buffer_size // self.num_classes, self.L))

    def buffer_update_Contrastive(self, x, y, temp=0.5, loss_func=nn.NLLLoss()):  # Moco's temp = 0.5; SimCLR's temp = 0.07;
        assert x.shape[0] == y.shape[0]

        # buffer update loss calculation
        cls_centers = torch.mean(self.rehearsal, dim=1)
        rehearsal_label = torch.tensor([i for i in range(self.num_classes)], dtype=torch.long).to(x.device)

        similarity_matrix = torch.matmul(x, cls_centers.T).to(x.device)
        pos_position = (rehearsal_label.unsqueeze(0) == y.unsqueeze(1))
        positives = similarity_matrix[pos_position].view(x.shape[0], -1)
        negatives = similarity_matrix[~pos_position].view(positives.shape[0], -1)

        contrastive_logits = torch.cat([positives, negatives], dim=1)
        contrastive_targets = torch.zeros(contrastive_logits.shape[0], dtype=torch.long).to(x.device)
        loss_term = loss_func(torch.log_softmax(contrastive_logits / temp, dim=1), contrastive_targets)

        # regularization_term calculation
        cls_center_sim_matrix = 1 + torch.cosine_similarity(cls_centers.unsqueeze(0), cls_centers.unsqueeze(1), dim=-1)
        regularization_term = torch.sum(cls_center_sim_matrix[~torch.eye(cls_center_sim_matrix.shape[0], dtype=torch.bool).to(x.device)]) / 2

        # buffer update
        cls_dists = []
        for i in range(self.rehearsal.shape[0]):
            cls_buffer = torch.squeeze(self.rehearsal[i, :, :])
            dist = torch.cosine_similarity(cls_centers[i].unsqueeze(0), cls_buffer.unsqueeze(1), dim=-1).squeeze()

            cls_dists.append(dist)

        for i in range(y.shape[0]):
            dist = torch.cosine_similarity(cls_centers[y[i]], x[i], dim=0)

            min_value, min_idx = torch.min(cls_dists[y[i]]), torch.argmin(cls_dists[y[i]])
            if dist.item() > min_value:
                self.rehearsal[y[i], torch.argmin(cls_dists[y[i]]), :] = x[i].detach()
                cls_dists[y[i]][min_idx] = dist.item()

        return loss_term, regularization_term