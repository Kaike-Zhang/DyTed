import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import BCEWithLogitsLoss

from model.HTGN.script.poincare import PoincareBall
from torch_geometric.utils import negative_sampling


EPS = 1e-15
MAX_LOGVAR = 10


class ReconLoss(nn.Module):
    def __init__(self, args):
        super(ReconLoss, self).__init__()
        self.negative_sampling = negative_sampling
        self.sampling_times = args.sampling_times
        # self.node_num = args.no
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.manifold = PoincareBall()
        self.use_hyperdecoder = args.use_hyperdecoder

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    # src的embedding和tar的embedding点积作为边的预测
    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def hyperdeoder(self, z, edge_index):
        def FermiDirac(dist):
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, c=1.0)
        return FermiDirac(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_loss = -torch.log(
            decoder(z, pos_edge_index) + EPS).mean()
        # pos_edge_index
        if neg_edge_index == None:
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=pos_edge_index.size(1) * self.sampling_times)
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        # decoder = self.decoder
        # 各项属性与z相同的 全1张量与全0张量
        pos_y = z.new_ones(pos_edge_index.size(1)).to(z.device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(z.device)
        # 将两个张量拼接作为标签
        y = torch.cat([pos_y, neg_y], dim=0)
        # 获得边的预测值
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        # 将预测值拼接对应了标签
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAEloss(ReconLoss):
    def __init__(self, args):
        super(VGAEloss, self).__init__(args)

    def kl_loss(self, mu=None, logvar=None):
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(
            max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def forward(self, x, pos_edge_index, neg_edge_index):
        z, mu, logvar = x
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        reconloss = pos_loss + neg_loss
        klloss = (1 / z.size(0)) * self.kl_loss(mu=mu, logvar=logvar)

        return reconloss + klloss


class DySATLoss(nn.Module):
    def __init__(self, args):
        super(DySATLoss, self).__init__()
        self.args = args
        self.bce_loss = BCEWithLogitsLoss()

    def forward(self, feed_dict, final_emb):
        node_1, node_2, node_2_negative = feed_dict.values()
        # print(final_emb)
        graph_loss = 0
        for t in range(final_emb.shape[1]):
            emb_t = final_emb[:, t, :].squeeze()  # [N, F]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()
            pos_score[pos_score.isnan()] = 0.5
            neg_score[neg_score.isnan()] = 0.5
            if len(pos_score) == 0 or len(neg_score) == 0:
                continue
            pos_loss = self.bce_loss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bce_loss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight * neg_loss
            graph_loss += graphloss
        return graph_loss

