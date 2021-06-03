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
        self.dropout = nn.Dropout(0.5)
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
        #feature = self.dropout(feature)
        feature = self.relu(feature)
        feature = self.ln2(feature)
        #feature = self.dropout(feature)
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
    
    # X_t, Y_t = load_data(["wiki_abstracts/abtract_tech.json", "wiki_abstracts/empty_tech_intro.json"], True)
    # X_n, Y_n = load_data(["wiki_abstracts/abtract_non_tech.json", "wiki_abstracts/empty_non_tech_intro.json"], False)

    # np.savez("abstract_embs.npz", X_t, Y_t, X_n, Y_n)
#     data = np.load("abstract_embs.npz")
    data = np.load("abstract_embs.npz")
    X_t, Y_t, X_n, Y_n = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # print(Y_t.shape)
    num_samples = X_t.shape[0]
    dim = X_t.shape[1]
    batch = 2048
#     X_t = np.ones((155000,dim))
#     X_n = np.ones((155000,dim))
    #X_t = X_t / np.sqrt((X_t**2).sum(axis = 1).reshape(-1,1))
    #X_n = X_n / np.sqrt((X_n**2).sum(axis = 1).reshape(-1,1))
#     Y_t = np.ones((155000,))
#     Y_n = np.zeros((155000,))

    X = np.concatenate((X_t, X_n[:num_samples]), axis=0)
    Y = np.concatenate((Y_t, Y_n[:num_samples]), axis=0)

    print(X.shape)
    print(Y.shape)

    model = LitModule(dim, batch)
    train_data, test_data = split_data(X, Y)
    train_data = DataLoader(dataset=train_data, batch_size=batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    from tqdm import tqdm
    for epoch in range(501):
        for data in tqdm(train_data):
            optimizer.zero_grad()
            loss = model.loss(data[0],data[1])
            loss.backward()
            print(loss)
            optimizer.step()
            # import pdb
            # pdb.set_trace()
        y_hat = model.forward(test_data[0])
        print(((y_hat-0.5)**2).sum())
        y_hat = (y_hat > 0.5).type(torch.int8)
        print('epoch: {} acc: {}'.format(epoch, (((y_hat-1) * (test_data[1]-1)).sum() + (y_hat * test_data[1]).sum())/test_data[1].shape[0]))
#         if epoch % 100 == 0:
            # model.load_state_dict(torch.load(PATH))
        if epoch in [1,5,100,200,300,400,500]:
            torch.save(model.state_dict(),'weight_tfidfi_normal_epoch{}.pth'.format(epoch))
        





if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

