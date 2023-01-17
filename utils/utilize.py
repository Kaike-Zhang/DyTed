import numpy as np
import networkx as nx
import pickle as pkl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from model.HTGN.script.poincare import PoincareBall


def sigmoid(x):
    if x >= 0:
        x = 1.0 / (1 + np.exp(-float(x)))
    else:
        x = np.exp(float(x)) / (1 + np.exp(float(x)))
    return x


def get_score(n1, n2):
    n1 = np.array(n1)
    n2 = np.array(n2)

    rs = np.multiply(n1, n2)
    rs = [sigmoid(i) for i in rs]

    return rs


def get_link_feats(edge, emb):
    features = []
    for e in edge:
        src = e[0]
        tar = e[1]
        f = get_score(emb[src], emb[tar])
        # f = n
        features.append(f)

    features = np.array(features)

    return features


def get_evaluation_data(graphs):
    # Load train/val/test examples to evaluate link prediction performance
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx + 1]
    # print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2,
                           test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []
    # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive)  # [E, 2]
    if len(edges_positive) > 10000:
        idx = range(edges_positive.shape[0])
        idx = np.random.choice(idx, size=10000, replace=False)
        edges_positive = edges_positive[idx]
    # generate negative edges
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)

    # split train and zip(val,test) samples)
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
                                                                            edges_negative,
                                                                            test_size=val_mask_fraction + test_mask_fraction)
    # split val and test samples
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
                                                                                    test_neg,
                                                                                    test_size=test_mask_fraction / (
                                                                                            test_mask_fraction + val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg


def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    # length(positive samples) == length(negative samples)
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg


def load_graphs(dataset_str):
    # Load graph snapshots given the name of dataset
    with open("./data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]

    return graphs, adjs


def load_noise_graphs(dataset_str, rate):
    with open("./data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)

    graph_list = []
    for g in graphs:
        noise_len = int(len(g.edges()) * rate)
        nodes_num = len(g.nodes())
        new_g = nx.MultiGraph(g)
        i = 0
        while i <= noise_len:
            idx_i = np.random.randint(0, nodes_num)
            idx_j = np.random.randint(0, nodes_num)
            if idx_i == idx_j:
                continue
            if g.has_edge(idx_i, idx_j) and new_g.has_edge(idx_i, idx_j):
                new_g.remove_edge(idx_i, idx_j)
                i += 1
            if not g.has_edge(idx_i, idx_j) and not new_g.has_edge(idx_i, idx_j):
                new_g.add_edge(idx_i, idx_j)
                i += 1
        graph_list.append(new_g)
    adjs = [nx.adjacency_matrix(g) for g in graph_list]

    return graph_list, adjs


def generate_feats(graphs, device):
    return torch.eye(len(graphs[-1].nodes())).to(device)


def get_sample(sta, dyn, length, sample_num):
    each_t = int(sample_num / length)
    pos = []
    neg = []
    for t in range(length):
        # pos
        rg_idx = range(0, sta.shape[0])
        global_emb = torch.cat((sta, dyn[:, t, :].squeeze()), dim=1)
        cs_idx = np.random.choice(rg_idx, size=each_t, replace=False)
        pos.append(global_emb[cs_idx])
        # neg
        temp_pos = np.random.randint(sta.shape[0] - 1)
        idx = [i for i in range(temp_pos, sta.shape[0])] + [j for j in range(0, temp_pos)]
        neg_emb = torch.cat((sta, dyn[:, t, :].squeeze()[idx]), dim=1)
        neg.append(neg_emb[cs_idx])

    pos_sample = torch.vstack(pos)
    neg_sample = torch.vstack(neg)

    return pos_sample, neg_sample


def evaluate_product(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, embedding):
    test_roc_score, test_ap_score = get_roc_score_t(test_pos, test_neg, embedding, embedding)
    val_roc_score, val_ap_score = get_roc_score_t(val_pos, val_neg, embedding, embedding)
    tra_roc_score, tra_ap_score = get_roc_score_t(train_pos, train_neg, embedding, embedding)

    return np.mean([test_roc_score, val_roc_score, tra_roc_score]), \
           np.mean([test_ap_score, val_ap_score, tra_ap_score])


def get_roc_score_t(edges_pos, edges_neg, source_emb, target_emb):
    """Given test examples, edges_pos: +ve edges, edges_neg: -ve edges, return ROC scores for a given snapshot"""

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pred = []
    pos = []
    for e in edges_pos:
        pred.append(hyperdeoder(source_emb, e))
        pos.append(1.0)

    pred_neg = []
    neg = []
    for e in edges_neg:
        pred_neg.append(hyperdeoder(source_emb, e))
        neg.append(0.0)

    pred_all = np.hstack([pred, pred_neg])
    labels_all = np.hstack([pos, neg])
    roc_score = roc_auc_score(labels_all, pred_all)
    ap_score = average_precision_score(labels_all, pred_all)
    return roc_score, ap_score


def hyperdeoder(z, edge_index):
    def FermiDirac(dist):
        r = 2.0
        t = 1.0
        probs = 1. / (torch.exp((dist - r) / t) + 1.0)
        return probs

    # edge_i = edge_index[0]
    # edge_j = edge_index[1]
    # z_i = torch.nn.functional.embedding(edge_i, z)
    z_i = z[edge_index[0]]
    # z_j = torch.nn.functional.embedding(edge_j, z)
    z_j = z[edge_index[1]]
    manifold = PoincareBall()
    dist = manifold.sqdist(torch.tensor(z_i), torch.tensor(z_j), c=1.0)

    return FermiDirac(dist)
