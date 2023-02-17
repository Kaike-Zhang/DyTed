from model.EvolveGCN.EvolveGCN import EvolveGCN
from model.LSTMGCN.LSTM_GCN import LSTMGCN
from trainer.config import args
from utils.utilize import load_graphs, generate_feats
from trainer.trainer import Trainer
import torch
import numpy as np


def load_model(model):
    if model == "EvolveGCN":
        return EvolveGCN
    elif model == "LSTMGCN":
        return LSTMGCN

if __name__ == '__main__':
    np.random.seed(2022)
    torch.manual_seed(2022)
    # Load data
    graphs, adjs = load_graphs(args.dataset)
    # Load node features
    if args.pre_defined_feature is None:
        if args.use_trainable_feature:
            feats = torch.ones([args.node_num, 1]).to(args.device)
        else:
            feats = generate_feats(graphs, args.device)
    else:
        # Todo: load predefined features
        pass

    model = load_model(args.model)

    trainer = Trainer(graphs, adjs, feats, args, model)
    auc, ap, ti, df = trainer.run()
    result = "mean AUC: {:.4f}, mean AP: {:.4f}".format(auc, ap)

    print("**" * 10)
    print("Train Result:")
    print(result)
    print("**" * 10)

    result_ti = "Temporal-invariant: mean AUC: {:.4f}, mean AP: {:.4f}".format(ti[0], ti[1])
    result_df = "Dynamic-fluctuate: mean AUC: {:.4f}, mean AP: {:.4f}".format(df[0], df[1])
    print(result_ti)
    print(result_df)
