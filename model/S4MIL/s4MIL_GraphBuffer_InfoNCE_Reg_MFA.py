import torch
import torch.nn as nn
from . import s4

from model.modules import DSL, GCN

class S4Model(nn.Module):
    def __init__(self, model_dim, state_dim, input_dim, n_classes, k, buffer_size, batch_size, temp):
        super(S4Model, self).__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.temp = temp

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.model_dim),
                                 nn.ReLU())
        self.s4_block = nn.Sequential(nn.LayerNorm(self.model_dim),
                                      s4.S4D(d_model=self.model_dim, d_state=self.state_dim, transposed=False))
        self.fc2 = nn.Linear(self.model_dim, self.n_classes)

        self.dsl = DSL(feat_dim=self.model_dim, k=k)
        self.max_length = self.batch_size + self.buffer_size
        self.gcn = GCN(self.model_dim, num_node=self.max_length, out_dim=self.n_classes)

        self.register_buffer('rehearsal', torch.rand(self.n_classes, self.buffer_size // self.n_classes, self.model_dim))
    
    def forward(self, x, y=None):
        B = x.shape[0]

        x = self.fc1(x)
        x = self.s4_block(x)
        bag_feat = torch.max(x, dim=1).values

        logits_mlp = self.fc2(bag_feat)

        if y is not None or not self.training:  # 推理阶段 或者 非Warmup训练阶段
            x_concat = torch.cat([bag_feat, self.rehearsal.view((-1, self.model_dim))])
            edge_index, edge_attr = self.dsl(x_concat)

            padded_x = torch.zeros((self.max_length, self.model_dim)).cuda()
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
        x_concat = torch.cat([x, self.rehearsal.view((-1, self.model_dim))])[:self.buffer_size, :].detach()
        self.rehearsal = x_concat.view((self.n_classes, self.buffer_size // self.n_classes, self.model_dim))

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
