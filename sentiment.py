from models import OneLayerNetwork, ConvNetwork
from trainer import Trainer
from processing import one_hot_encoding

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchtext.datasets as datasets
import torchtext.data as data
import torchtext
import torch.optim as optim
import torch.nn as nn

DATASET_DIR="../data/sentiment/data"
VECTORS_DIR="../data/sentiment/vectors"
SEED = 0
torch.manual_seed(SEED)
#import ipdb; ipdb.set_trace()

class Sentiment(Trainer):
    def __init__(self, n_epochs=4, batch_size=32):
        self.nout=3
        self.wordemb_len = 100
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_iter, self.val_iter, self.test_iter = self.load_data()
        self.nn_embeddings = nn.Embedding.from_pretrained(self.TEXT.vocab.vectors)

    def load_data(self):
        self.TEXT=data.Field(lower=True,include_lengths=False,batch_first=True)
        self.LABEL = data.Field(sequential=False, is_target=True)

        train, val, test = datasets.sst.SST.splits(self.TEXT, self.LABEL,
                                                   root=DATASET_DIR)

        wordemb = torchtext.vocab.GloVe("6B", dim=self.wordemb_len, cache=VECTORS_DIR)
        self.TEXT.build_vocab(train, vectors=wordemb)
        self.LABEL.build_vocab(train, specials_first=False)

        return data.BucketIterator.splits((train, val, test),
                                          batch_sizes=(self.batch_size, 1, 1),
                                          device=0)


    def process_input(self, x):
        x = self.nn_embeddings(x)
        x_pad = torch.zeros((x.shape[0], self.MAX_SENT_LEN, x.shape[2]))

        for i, sentence in enumerate(x):
            l = len(sentence)
            x_pad[i,0:l] = sentence

        return x_pad.transpose(1,2)

    def train_model(self, model, loss = nn.MSELoss(), optimizer=optim.SGD, optim_param={}):
        print('train model')

        torch.manual_seed(SEED)
        random.seed(SEED)

        train_loss, val_loss, train_acc, val_acc = [], [], [], []

        self.model = model(self.wordemb_len, self.nout)
        self.loss = loss
        self.opt = optimizer(self.model.parameters(), **optim_param)
        self.MAX_SENT_LEN = max([batch.shape[1] for batch,_ in self.train_iter])

        for epoch_n in range(self.n_epochs):
            for batch_idx, (x, y) in enumerate(self.train_iter):

                x = self.process_input(x)
                y = one_hot_encoding(y, self.nout)

                loss_t, acc_t = self.train(x.double(), y.double())
                train_loss.append(loss_t)
                train_acc.append(acc_t)

            epoch_val_loss, epoch_val_acc = [], []
            for x_val, y_val in self.val_iter:
                x_val = self.process_input(x_val)
                y_val = one_hot_encoding(y_val, self.nout)
                loss_v, acc_v = self.validation(x_val.double(), y_val.double())
                with torch.no_grad():
                    epoch_val_loss.append(loss_v)
                    epoch_val_acc.append(acc_v)
            val_loss.append(np.mean(epoch_val_loss))
            val_acc.append(np.mean(epoch_val_acc))
            print('Epoch %d. Val loss = %.3f. Val accuracy = %.3f'%(batch_idx,
                                                                    np.mean(epoch_val_loss),
                                                                    np.mean(epoch_val_acc)))

        plt.figure(figsize=(10, 8))
        plt.plot(train_loss, color='blue', alpha=.6)
        plt.title('Train loss')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(train_acc, color='red', alpha=.6)
        plt.title('Train acc')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(val_loss, color='blue', alpha=.6)
        plt.title('Validation loss')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(val_acc, color='red', alpha=.6)
        plt.title('Validation acc')
        plt.show()




if __name__ == '__main__':
    s = Sentiment(n_epochs=10, batch_size=200)
    s.train_model(ConvNetwork,
                  optimizer=optim.Adam,
                  optim_param={'lr' : 10e-3})


