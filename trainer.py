import numpy as np
import torch

class Trainer():
    def __init__(self):
        pass

    def train(self, x, y, retain_graph=False):
        self.opt.zero_grad()
        output = self.model(x)
        loss_t = self.loss.forward(output, y)
        acc_t = self.accuracy(output, y)
        loss_t.backward(retain_graph=retain_graph)
        self.opt.step()
        return float(loss_t), acc_t


    def accuracy(self, ypred, y):
        # for binary classification
        if y.size()[1]==1:
            y_encoded = torch.Tensor([[int(round(ypred_i))] for ypred_i in ypred])
            return float(y_encoded.eq(y).sum().item()/float(y.shape[0]))

        y_encoded = y.max(1, keepdim=True)[1].view(-1)
        ypred_encoded = ypred.max(1, keepdim=True)[1].view(-1)
        return float(ypred_encoded.eq(y_encoded).sum().item()/float(y_encoded.shape[0]))


    def validation(self, x, y):
        output = self.model(x)
        loss_v = self.loss.forward(output, y)
        acc_v = self.accuracy(output, y)
        return float(loss_v), acc_v
