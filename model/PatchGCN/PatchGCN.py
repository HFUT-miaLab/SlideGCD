# Code integrated from https://github.com/mahmoodlab/Patch-GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class PatchGCN(torch.nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial', feat_dim=384, hidden_dim=128, dropout=0.25, n_classes=4):
        super(PatchGCN, self).__init__()
        self.edge_agg = edge_agg
        self.num_layers = num_layers-1

        self.fc = nn.Sequential(*[nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_latent = data.edge_latent
        batch_size = data.y.shape[0]

        if self.edge_agg == 'spatial':
            edge_index = edge_index
        elif self.edge_agg == 'latent':
            edge_index = edge_latent
        else:
            raise NotImplementedError
        edge_attr = None

        x = self.fc(x)
        x_ = x 
        
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], dim=-1)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], dim=-1)

        # print(torch.reshape(x_, (64, 500, 512))[0])
        # h_path = x_
        h_path = torch.reshape(x_, (batch_size, 500, 512))
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, -1, -2)
        h_path = torch.bmm(F.softmax(A_path, dim=-1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        probs = torch.sigmoid(logits)
        
        return logits, A_path, None


class PatchGCN_Surv(torch.nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial', feat_dim=384, hidden_dim=128, dropout=0.25, n_classes=4):
        super(PatchGCN_Surv, self).__init__()
        self.feat_dim = feat_dim
        self.edge_agg = edge_agg
        self.num_layers = num_layers - 1

        self.fc = nn.Sequential(*[nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * 4, D=hidden_dim * 4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim * 4, n_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch_size = data.y.shape[0]

        edge_attr = None

        x = self.fc(x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], dim=-1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], dim=-1)

        # h_path = torch.zeros((batch_size, 5000, x_.shape[-1])).cuda()
        # for i in range(batch_size):
        #     h_path[i, :, :] = x_[data.batch == i]
        h_path = torch.reshape(x_, (batch_size, -1, x_.shape[-1]))
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, -1, -2)
        h_path = torch.bmm(F.softmax(A_path, dim=-1), h_path)
        h_path = torch.reshape(h_path, (batch_size, h_path.shape[-1]))
        h = self.path_rho(h_path)
        logits = self.classifier(h)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=-1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=-1)
        return hazards, S, Y_hat, A_path, None