import os
import time

import torch
from torch import optim, nn
from torch_geometric.data import Data
import torch_geometric as tg
import torch.nn.functional as F
import pickle as pkl
import numpy as np

from model.DyTed import DyTed
from model.component.loss import ReconLoss
from model.discriminate.discriminator import Discriminator
from utils.logistic_cls import evaluate_classifier
from utils.utilize import get_sample, get_evaluation_data


class Trainer:
    def __init__(self, graphs, adjs, feature, args, Model):

        self.graphs = graphs
        self.adjs = adjs
        self.features = feature
        self.args = args
        self.node_num = self.args.node_num

        self.init_model = Model

        self._create_model()
        self._build_pyg_graphs()
        self._init_path()

    def _init_path(self):
        if not os.path.exists(self.args.log_path + "/train_log.txt"):
            os.mknod(self.args.log_path + "/train_log.txt")
        if not os.path.exists(self.args.log_path + "/{}_result.txt".format(self.args.dataset)):
            os.mknod(self.args.log_path + "/{}_result.txt".format(self.args.dataset))

    def _create_model(self):
        model = self.init_model
        self.model = DyTed(model, self.args).to(self.args.device)

        self.discriminator = Discriminator(self.args.dis_in, self.args.dis_hid).to(self.args.device)

        if torch.cuda.is_available() and self.args.use_gpu:
            self.model.cuda()
            self.discriminator.cuda()
        print("create model!!!")

    def _build_pyg_graphs(self):
        pyg_graphs = []
        whole_node = set(range(self.args.node_num))

        for adj in self.adjs:
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)

            temp_node = set(edge_index[0].detach().cpu().numpy()) | set(edge_index[1].detach().cpu().numpy())
            del_node = list(whole_node - temp_node)
            temp_feats = self.features.clone()
            temp_feats[del_node] = torch.zeros(self.features.shape[1]).to(self.features.device)
            data = Data(x=temp_feats,
                        edge_index=edge_index.to(self.features.device),
                        edge_weight=edge_weight.to(self.features.device))

            pyg_graphs.append(data)
            self.pyg_graphs = pyg_graphs[:-1]

    def run(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        opt_dis = optim.AdamW(self.discriminator.parameters(), lr=self.args.lr,
                              weight_decay=self.args.weight_decay)
        t_total0 = time.time()
        patient = 0
        min_loss = 10000000

        # Load evaluation data for link prediction.
        train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
        test_edges_pos, test_edges_neg = get_evaluation_data(self.graphs)
        print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
            len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
            len(test_edges_pos), len(test_edges_neg)))

        time_list = []
        mem_list = []

        for epoch in range(0, self.args.max_epoch):
            t0 = time.time()
            opt.zero_grad()
            self.model.train()

            ti_emb, ti_emb_other, df_emb = self.model(self.pyg_graphs)
            global_emb = torch.stack([torch.cat((ti_emb, df_emb[:, t, :].squeeze()), dim=1)
                                      for t in range(df_emb.shape[1])], dim=1)
            gb_loss = self.get_loss(global_emb)
            loss = self.args.ti_weight * self.get_ti_loss(ti_emb, ti_emb_other) + gb_loss

            if epoch >= self.args.dis_start:
                self.discriminator.eval()
                pos, neg = get_sample(ti_emb, df_emb, len(self.pyg_graphs), self.args.dis_sample_num)
                pos, neg = self.discriminator(pos, neg)
                loss = loss - self.args.dis_weight * self.get_dis_loss(pos, neg)

            loss.backward()
            opt.step()

            if epoch >= self.args.dis_start:
                self.model.eval()
                self.discriminator.train()
                for i in range(self.args.dis_epoch):
                    opt_dis.zero_grad()
                    pos, neg = get_sample(ti_emb, df_emb, len(self.pyg_graphs), self.args.dis_sample_num)
                    pos, neg = self.discriminator(pos.detach(), neg.detach())
                    dis_loss = self.get_dis_loss(pos, neg)
                    dis_loss.backward()
                    opt_dis.step()

                for p in self.model.sampling.parameters():
                    p.data.clamp_(0.0, 1 / 3)

            self.model.eval()
            if loss < min_loss:
                min_loss = loss
                patient = 0
            else:
                patient += 1
                if patient > self.args.patience and epoch > self.args.min_epoch:
                    print('early stopping')
                    break

            gpu_mem_alloc = torch.cuda.max_memory_allocated(
                self.args.device) / 1000000 if torch.cuda.is_available() else 0

            time_list.append(time.time() - t0)
            mem_list.append(gpu_mem_alloc)

            with torch.no_grad():
                print("==" * 20)
                epoch_info = "Model:{} ,Epoch:{}/{}, Loss: {:.4f}, Time: {:.3f}, " \
                             "GPU: {:.1f}MiB, Use DyTed".format(self.args.model,
                                                                    epoch + 1,
                                                                    self.args.max_epoch,
                                                                    loss,
                                                                    time.time() - t0,
                                                                    gpu_mem_alloc)
                print(epoch_info)
                with open(self.args.log_path + "/train_log.txt", 'a') as f:
                    f.write(epoch_info + "\n")
                if (epoch + 1) % self.args.log_interval == 0:
                    print("-------time:{:.3f}------mem:{:.1f}".format(np.mean(time_list), np.mean(mem_list)))
                    auc_score, ap_score = self.test(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg,
                                                    test_edges_pos, test_edges_neg)
                    print("AUC: {:.4f}, AP: {:.4f}".format(auc_score, ap_score))
                    ti_score = self.test(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg,
                                         test_edges_pos, test_edges_neg, type=1)
                    df_score = self.test(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg,
                                         test_edges_pos, test_edges_neg, type=2)
                    print("TI: AUC: {:.4f}, AP: {:.4f}".format(ti_score[0], ti_score[1]))
                    print("DF: AUC: {:.4f}, AP: {:.4f}".format(df_score[0], df_score[1]))

        print("Total time: {:.3f}".format(time.time() - t_total0))
        auc_score, ap_score = self.test(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg,
                                        test_edges_pos, test_edges_neg)

        with open(self.args.log_path + "/{}_result.txt".format(self.args.dataset), 'a') as f:
            f.write(
                "Model: {}, AP: {:.4f}, AUC: {:.4f}, Time: {:.3f}, Use DyTed".format(self.args.model, auc_score,
                                                                                         ap_score,
                                                                                         time.time() - t_total0)
                + "\n"
            )
        ti_score = self.test(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg,
                             test_edges_pos, test_edges_neg, type=1)
        df_score = self.test(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg,
                             test_edges_pos, test_edges_neg, type=2)

        return auc_score, ap_score, ti_score, df_score

    def _normalize(self, *xs):
        data = [None if x is None else F.normalize(x, dim=-1) for x in xs]
        return data[0]

    def get_ti_loss(self, ti_emb, ti_emb_other):
        ti_emb = self._normalize(ti_emb)
        ti_emb_other = self._normalize(ti_emb_other)
        ti_pos = ((ti_emb * ti_emb_other).sum(dim=1) / self.args.t).exp()
        ti_neg = (torch.dsmm(ti_emb, ti_emb_other.T) / self.args.t).exp().sum(dim=1)
        ti_loss = -1 * torch.log(ti_pos / ti_neg).sum()
        return ti_loss

    def get_loss(self, emb):
        # emb = self._normalize(emb)
        all_loss = 0
        if self.args.model not in ["LSTMGCN", "EvolveGCN"]:
            pass
        else:
            train_shots = list(range(0, len(self.graphs) - self.args.testlength))
            loss = ReconLoss(self.args)
            for t in train_shots:
                all_loss += loss(emb[:, t, :], self.pyg_graphs[t].edge_index)

        return all_loss

    def get_dis_loss(self, pos, neg):
        loss = nn.BCELoss()
        pos_tar = torch.ones(pos.size()).to(self.args.device)
        neg_tar = torch.zeros(neg.size()).to(self.args.device)
        dis_loss = -self.args.dis_weight * (loss(pos, pos_tar) + loss(neg, neg_tar))

        return dis_loss

    def _get_emb(self):
        ti_emb, _, df_emb = self.model(self.pyg_graphs, type=1)
        emb = torch.cat((ti_emb, df_emb[:, -1, :].squeeze()), dim=1).detach().cpu().numpy()
        return emb

    def test(self, train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg,
             type=0):
        with torch.no_grad():
            emb = self._get_emb()
            if type == 1:
                emb = emb[:, :int(emb.shape[-1] / 2)]
            elif type == 2:
                emb = emb[:, int(emb.shape[-1] / 2):]
            auc_score, ap_score, _, _ = evaluate_classifier(train_edges_pos, train_edges_neg,
                                                            val_edges_pos, val_edges_neg,
                                                            test_edges_pos, test_edges_neg, emb)
        return auc_score, ap_score

    def save_embs(self):
        with torch.no_grad():
            emb = self._get_emb()
            save_path = self.args.log_path + "/{}/{}".format(self.args.model, self.args.dataset)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path + "/embeddings.pkl", 'wb') as f:
                pkl.dump(emb, f)
            torch.save(self.model.state_dict(), save_path + "/model.pt")

