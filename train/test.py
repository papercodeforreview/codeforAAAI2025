import os
import torch
import numpy as np
import torch.nn as nn
from dataset_0607util import load_ddi_dataset
from model import SA_DDI
from utils import *
from train_0607util import val
import pandas as pd
from data_preprocessing import CustomData
from dataset_0607util import DrugDataset, DrugDataLoader
import pickle


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def test():
    data_root = 'dataOpt/new_data/'
    save_dir = 'save'
    data_set = 'drugbig'
    data_path = os.path.join(data_root, f'{data_set}/clu/random')
    print(data_path)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda:0')


    train_loader, v_loader,test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)
    print(node_dim)
    print(edge_dim)
    model = causal(node_dim, edge_dim, n_iter=n_iter).cuda()
    model.load_state_dict(
        torch.load(""))

    test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap, test_aupr = val(model, criterion,
                                                                                               test_loader, device)
    test_msg = "test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f, test_aupr-%.4f" % (
        test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap, test_aupr)
    print(test_msg)


if __name__ == "__main__":
    test()
