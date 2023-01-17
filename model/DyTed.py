import copy

from torch import nn

from model.component.sampling import Sampling


class DyTed(nn.Module):
    def __init__(self, model, args):
        super(DyTed, self).__init__()
        self.args = args
        self.sampling = Sampling(self.args).to(self.args.device)
        self.ti_model = model(self.args).to(self.args.device)
        self.df_model = model(self.args).to(self.args.device)

    def forward(self, graphs, type=0):
        ti_emb, ti_emb_other = self.forward_ti(graphs, type)
        df_emb = self.df_model(graphs)

        return ti_emb, ti_emb_other, df_emb

    def forward_ti(self, graphs, type=0):
        if type == 0:
            seq_begin, seq_1_end, seq_2_begin, seq_end, len_1, len_2 = self.sampling(len(graphs))
            ti_graphs = graphs[seq_begin:seq_end + 1]
            ti_graphs_1 = copy.deepcopy(ti_graphs)
            ti_graphs_2 = copy.deepcopy(ti_graphs)
            for i in range(len(len_1)):
                ti_graphs_1[i].x = ti_graphs_1[i].x * len_1[i]
                ti_graphs_2[i].x = ti_graphs_2[i].x * len_2[i]
            ti_emb = self.ti_model(ti_graphs_1[:seq_1_end+1])[:, -1, :]
            ti_emb_other = self.ti_model(ti_graphs_2[seq_2_begin:])[:, -1, :]
        else:
            ti_emb = self.ti_model(graphs)[:, -1, :]
            ti_emb_other = ti_emb

        return ti_emb, ti_emb_other

    def forward_single_step_df(self, graph):
        return self.df_model.inner_forward(graph.edge_index, graph.x)
