import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

from model.modules import DSL, GCN


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size, temp):
        super(TransMIL, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.temp = temp

        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(self.feat_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = self.num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.dsl = DSL(feat_dim=self.feat_dim, k=k)
        self.max_length = self.batch_size + self.buffer_size
        self.gcn = GCN(self.feat_dim, num_node=self.max_length, out_dim=self.n_classes)

        self.register_buffer('rehearsal', torch.rand(self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def forward(self, x, y=None):
        h = x  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        bag_feat = self.norm(h)[:, 0]

        # ---->predict
        logits_mlp = self._fc2(bag_feat)  # [B, n_classes]

        if y is not None or not self.training:  # ÍÆÀí½×¶Î »òÕß ·ÇWarmupÑµÁ·½×¶Î
            x_concat = torch.cat([bag_feat, self.rehearsal.view((-1, self.feat_dim))])
            edge_index, edge_attr = self.dsl(x_concat)

            padded_x = torch.zeros((self.max_length, self.feat_dim)).cuda()
            padded_x[:bag_feat.shape[0]] = bag_feat
            logits_graph = torch.squeeze(self.gcn(padded_x, edge_index, edge_attr))[:B, :]

        if self.training:
            if y is None:  # WarmupÑµÁ·½×¶Î
                self.buffer_update_warmup(bag_feat)
                return logits_mlp
            else:  # ·ÇWarmupÑµÁ·½×¶Î
                buffer_update_loss, reg_loss = self.buffer_update(bag_feat, y, temp=self.temp)
                return logits_mlp, logits_graph, buffer_update_loss, reg_loss
        else:  # ÍÆÀí½×¶Î
            return logits_mlp, logits_graph

    def buffer_update_warmup(self, x):
        x_concat = torch.cat([x, self.rehearsal.view((-1, self.feat_dim))])[:self.buffer_size, :].detach()
        self.rehearsal = x_concat.view((self.num_classes, self.buffer_size // self.num_classes, self.feat_dim))

    def buffer_update(self, x, y, temp=0.5):  # Moco's temp = 0.5; SimCLR's temp = 0.07;
        assert x.shape[0] == y.shape[0]
        # ¼à¶½¶Ô±ÈËðÊ§¼ÆËã
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

        # Buffer¸üÐÂ
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