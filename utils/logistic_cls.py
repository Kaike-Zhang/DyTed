from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import linear_model

from utils.utilize import *


def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, embedding):
    # Downstream logistic regression classifier to evaluate link prediction
    train_pos_feats = np.array(get_link_feats(train_pos, embedding))
    train_neg_feats = np.array(get_link_feats(train_neg, embedding))
    val_pos_feats = np.array(get_link_feats(val_pos, embedding))
    val_neg_feats = np.array(get_link_feats(val_neg, embedding))
    test_pos_feats = np.array(get_link_feats(test_pos, embedding))
    test_neg_feats = np.array(get_link_feats(test_neg, embedding))

    # label
    train_pos_labels = np.array([1] * len(train_pos_feats))
    train_neg_labels = np.array([-1] * len(train_neg_feats))
    val_pos_labels = np.array([1] * len(val_pos_feats))
    val_neg_labels = np.array([-1] * len(val_neg_feats))

    test_pos_labels = np.array([1] * len(test_pos_feats))
    test_neg_labels = np.array([-1] * len(test_neg_feats))

    # data
    train_data = np.vstack((train_pos_feats, train_neg_feats))
    train_labels = np.append(train_pos_labels, train_neg_labels)

    val_data = np.vstack((val_pos_feats, val_neg_feats))
    val_labels = np.append(val_pos_labels, val_neg_labels)

    test_data = np.vstack((test_pos_feats, test_neg_feats))
    test_labels = np.append(test_pos_labels, test_neg_labels)

    # Logistic Model
    logistic = linear_model.LogisticRegression(max_iter=10000)
    logistic.fit(train_data, train_labels)

    test_predict = logistic.predict_proba(test_data)[:, 1]
    val_predict = logistic.predict_proba(val_data)[:, 1]

    # score
    test_roc_score = roc_auc_score(test_labels, test_predict)
    val_roc_score = roc_auc_score(val_labels, val_predict)

    test_ap_score = average_precision_score(test_labels, test_predict)
    val_ap_score = average_precision_score(val_labels, val_predict)

    return test_roc_score, val_roc_score, test_ap_score, val_ap_score