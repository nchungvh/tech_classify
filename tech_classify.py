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

device = torch.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.ln1 = nn.Linear(self.input_dim, 2*self.input_dim)
        self.ln2 = nn.Linear(2*self.input_dim, 2*self.input_dim)
        self.ln3 = nn.Linear(2*self.input_dim, 2*self.input_dim)
        self.ln4 = nn.Linear(2*self.input_dim, self.input_dim)
        self.ln5 = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        # TODO: dropout

    def forward(self, feature):
        feature = self.ln1(feature)
        feature = self.dropout(feature)
        feature = self.relu(feature)
        # feature = self.ln2(feature)
        # feature = self.relu(feature)
        # feature = self.ln3(feature)
        # feature = self.relu(feature)
        feature = self.ln4(feature)
        feature = self.relu(feature)
        feature = self.ln5(feature)
        return feature

class LitModule(pl.LightningModule):
    def __init__(self, input_dim, X, Y, batch_size):
        super().__init__()
        self.model = Model(input_dim)
        self.train_data, self.test_data = self.split_data(X, Y)
        self.batch_size = batch_size
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        # self.val_roc = pl.metrics.ROC()

    def split_data(self, X, Y):
        cv = StratifiedShuffleSplit(1,test_size=0.3, random_state=42)
        for train_idx, test_idx in cv.split(X, Y):
            X_train=X[train_idx]
            Y_train=Y[train_idx]
            X_test=X[test_idx]
            Y_test=Y[test_idx]
        train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train).unsqueeze_(-1))
        test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test).unsqueeze_(-1))
        return train_data, test_data


    def forward(self, feature):
        emb = self.model(feature)
        return emb
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log('train_acc_step', self.train_acc(y_hat, y))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_acc.compute())
        
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        self.log('val_acc_step', self.val_acc(y_hat, y))
        return {'val_loss': self.loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.val_acc.compute())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer

    def loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size)


def get_bert_embeddings(word):
    word = ' '.join(word.replace('_',' ').replace('#','').split())
    input_ids = torch.tensor([tokenizer.encode(word, add_special_tokens=True)])                                                                                                                       
    input_ids = input_ids.to(device)                                                                                
    import pdb                                                                                                       
    #pdb.set_trace()                                                                                               
    with torch.no_grad():                                                                                        
        import pdb                                                                                                 
        #pdb.set_trace()                                                                                             
        last_hidden_states = bert_model(input_ids)[0].cpu().detach().numpy()  # Models outputs are now tuples       
    output = np.reshape(last_hidden_states, (-1, last_hidden_states.shape[-1])).sum(0)
    return output

def main(args):
    # x = json.load(open('abtract_data/output_0_50000.json'))
    x = json.load(open('test_data.json'))
    # x = json.load(open('small_data.json'))

    #TODO: use torchtext to preprocess

    X = []
    Y = []
    for i in tqdm(x):
        X.append(get_bert_embeddings(i['abtract']))
        Y.append(i['isTechnology'])
    X = np.asarray(X)
    Y = np.asarray(Y).astype(int)

    print(X.shape)
    print(Y.shape)

    model = LitModule(768, X, Y, 32)

    trainer = pl.Trainer.from_argparse_args(args, max_epochs=100)

    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

