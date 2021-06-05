import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from argparse import ArgumentParser
from transformers import * 
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import networkx as nx 
from networkx import json_graph 

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

device = torch.device('cuda')

# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# bert_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y 
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def split_data(X, Y):
    cv = StratifiedShuffleSplit(1,test_size=0.3, random_state=42)
    for train_idx, test_idx in cv.split(X, Y):
        X_train=X[train_idx]
        Y_train=Y[train_idx]
        X_test=X[test_idx]
        Y_test=Y[test_idx]
    train_data = CustomDataset(torch.Tensor(X_train), torch.Tensor(Y_train).unsqueeze_(-1))
    test_data = (torch.Tensor(X_test), torch.Tensor(Y_test).unsqueeze_(-1))
    return train_data, test_data

class LitModule(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(LitModule, self).__init__()
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.9)
        self.input_dim = input_dim
        self.ln1 = nn.Linear(self.input_dim, 2*self.input_dim)
        self.ln2 = nn.Linear(2*self.input_dim, 2*self.input_dim)
        self.ln3 = nn.Linear(2*self.input_dim, 2*self.input_dim)
        self.ln4 = nn.Linear(2*self.input_dim, self.input_dim)
        self.ln5 = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        # self.val_roc = pl.metrics.ROC()


    def forward(self, feature):
        feature = self.ln1(feature)
        feature = self.dropout(feature)
        feature = self.relu(feature)
        feature = self.ln2(feature)
        feature = self.relu(feature)
        # # feature = self.ln3(feature)
        # # feature = self.relu(feature)
        feature = self.ln4(feature)
        feature = self.relu(feature)
        feature = self.ln5(feature)
        # print(feature)
        feature = self.sigmoid(feature)
        return feature
    
    def loss(self, X, y):
        y_hat = self.forward(X)
        # print(y_hat)
        return self.criterion(y_hat, y)
        # return F.binary_cross_entropy(y_hat, y)  


def main(args):
    
    

    X = np.load('graph/ent_emb_bert.npy')
    id2idx = np.load('graph/iid2nid.npy',allow_pickle=True).item()
    print(X.shape)
    
    from tqdm import tqdm
    for epoch in range(100, 501, 100):
        graph = json_graph.node_link_graph(json.load(open('graph/merge-G_2404.json')))
        model = LitModule(512, 128)
        model = LitModule(512, 128)
        model.load_state_dict(torch.load('weight_epoch{}.pth'.format(epoch)))
        
        y_hat = model.forward(torch.Tensor(X))
        y_hat = (y_hat > 0.5).type(torch.int8)
        nontech = [id2idx[i] for i in range(len(y_hat)) if y_hat[i] == 0]
        graph.remove_nodes_from(nontech)
        graph.remove_nodes_from(list(nx.isolates(graph)))
        graph=nx.convert_node_labels_to_integers(graph)
        res = json_graph.node_link_data(graph)
        with open("graph/graph-{}-drop09.json".format(epoch), 'w') as outfile:
            json.dump(res, outfile)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
