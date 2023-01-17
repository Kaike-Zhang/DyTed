import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot


class LSTMGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.in_feature_list.append(args.out_feat)
        self.dropout = nn.Dropout(p=args.gcn_drop)
        self.gcn_layers = nn.ModuleList([GCNConv(args.in_feature_list[i], args.in_feature_list[i + 1])
                                         for i in range(args.cov_num)])
        self.rnn = nn.GRU(input_size=args.out_feat, hidden_size=args.out_feat)
        self.rnn2 = nn.GRU(input_size=args.out_feat, hidden_size=args.out_feat)
        self.feats = nn.Parameter(torch.ones(args.node_num, args.nfeat).to(args.device), requires_grad=True)

        self._init_layers()

    def forward(self, graphs):
        struct_out = []
        for graph in graphs:
            x = graph.x
            if self.args.use_trainable_feature:
                x = x * self.feats
            for gcn in self.gcn_layers:
                x = gcn.forward(x, graph.edge_index)
            struct_out.append(x[None, :, :])  # N x dim - len(T)

        x = torch.cat(struct_out, dim=0)
        output, hn = self.rnn(x)
        output, hn = self.rnn2(output)

        return output.transpose(0, 1).contiguous()

    def _init_layers(self):
        glorot(self.feats)
