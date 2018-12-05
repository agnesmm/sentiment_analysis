import torch

def one_hot_encoding(y, n_class):
    y_onehot = torch.FloatTensor(y.shape[0], n_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot

