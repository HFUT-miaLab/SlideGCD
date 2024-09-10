from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn import GraphNorm

from model.PatchGCN.PatchGCN import *
from model.modules import AdaptiveGraphGenerator, SlideGCN


class SlideGCD_PatchGCN(torch.nn.Module):
    def __init__(self, feat_dim, num_classes, k, buffer_size, batch_size,
                 num_layers=4, edge_agg='spatial', hidden_dim=128, dropout=0.25):
        super(SlideGCD_PatchGCN, self).__init__()
        # PatchGCN's parameters
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.edge_agg = edge_agg
        self.num_layers = num_layers - 1

        # SlideGCD's parameters
        self.hid_dim = self.hidden_dim * 4
        self.k = k
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.fc = nn.Sequential(*[nn.Linear(feat_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(self.hidden_dim, self.hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(self.hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=self.hidden_dim * 4, D=self.hidden_dim * 4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(self.hidden_dim * 4, self.num_classes)

        # SlideGCD's modules
        self.register_buffer('rehearsal', torch.rand(self.buffer_size, self.hid_dim))

        self.agg = AdaptiveGraphGenerator(feat_dim=self.hid_dim)

        self.max_length = self.batch_size + self.buffer_size
        self.gcn = SlideGCN(self.hid_dim, out_dim=self.num_classes, num_node=self.max_length)

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

        h_path = torch.reshape(x_, (batch_size, 500, 512))
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, -1, -2)
        h_path = torch.bmm(F.softmax(A_path, dim=-1), h_path)
        slide_embeddings = self.path_rho(h_path).squeeze()
        logits_mlp = self.classifier(slide_embeddings)  # logits needs to be a [1 x 4] vector
        # Y_hat = torch.topk(logits_mlp, 1, dim=1)[1]
        # probs = torch.sigmoid(logits_mlp)

        # concatenate current mini-batch slide embedding(s) with the nodes in buffer (rehearsal)
        # as the input of the graph branch;
        x_concat = torch.cat([slide_embeddings, self.rehearsal.view((-1, slide_embeddings.shape[1]))])
        # generate (hyper)graph edges with AGG module;
        edge_index, edge_attr = self.agg(x_concat)
        # graph interaction with designed SlideGCN and get the current mini-batch's response;
        padded_x = torch.zeros((self.max_length, self.hid_dim)).to(x_concat.device)
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
