import os
from os import path
import random
from random import shuffle
# from TensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F


SEED = 0
torch.set_default_tensor_type('torch.DoubleTensor')


class ConvNetwork(nn.Module):

    def __init__(self, nin, nout):
        super(ConvNetwork, self).__init__()

        self.conv1 = nn.Conv1d(nin, 10, kernel_size=2, stride=2, padding=0)
        self.max1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(10, 1, kernel_size=1, stride=2, padding=0)
        self.max2 = nn.MaxPool1d(2)

        self.fc = nn.Linear(nout, nout, bias=True)
        self.a = nn.Sigmoid()


    def forward(self, x):
        y = self.conv1(x)
        y = self.max1(y)
        y = self.conv2(y)
        y = self.max2(y)
        y=self.fc(y)
        y=self.a(y)
        y=y.view(y.shape[0],-1)
        return y



class OneLayerNetwork(nn.Module):

    def __init__(self, nin, nout=10, bias=True):
        super(OneLayerNetwork, self).__init__()
        self.fc = nn.Linear(nin, nout, bias=bias)
        self.a = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        return self.a(y)


class OneLayerTanhNetwork(nn.Module):

    def __init__(self, nin):
        super(OneLayerTanhNetwork, self).__init__()
        self.fc = nn.Linear(nin, 10, bias=True)
        self.a = nn.Tanh()

    def forward(self, x):
        y = self.fc(x)
        return self.a(y)


class TwoLayersNetwork(nn.Module):

    def __init__(self, nin):
        super(TwoLayersNetwork, self).__init__()
        self.m1 = nn.Linear(nin, 50, bias=True)
        self.tanh = nn.Tanh()
        self.m2 = nn.Linear(50, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.m1(x)
        y = self.tanh(y)
        y = self.m2(y)
        return self.sig(y)


class MultipleLayerNetwork(nn.Module):

    def __init__(self, nin):
        super(MultipleLayerNetwork, self).__init__()
        layers = [nn.Linear(nin, 300, bias=True),
        nn.Linear(300, 300, bias=True),
        nn.Linear(300, 200, bias=True),
        nn.Linear(200, 100, bias=True),
        nn.Linear(100, 50, bias=True),
        nn.Linear(50, 10)]

        self.num_layers = len(layers)
        self.linear = nn.ModuleList(layers)
        self.f = nn.Tanh()

        #self.writer = SummaryWriter()

    def forward(self, x):
        for layer in range(self.num_layers):
            x = self.f(self.linear[layer](x))
        return x


class SimpleHighwayNetwork(nn.Module):
    def __init__(self, nin):

        super(SimpleHighwayNetwork, self).__init__()

        layers = [nn.Linear(nin, 50, bias=True),
                  nn.Linear(50, 10)]

        self.num_layers = len(layers)

        self.nonlinear = nn.ModuleList(layers)
        self.linear = nn.ModuleList(layers)
        self.gate = nn.ModuleList(layers)
        self.f = nn.Tanh()

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x



class HighwayNetwork(nn.Module):
    def __init__(self, nin):

        super(HighwayNetwork, self).__init__()

        layers = [nn.Linear(nin, 300, bias=True),
                  nn.Linear(300, 300, bias=True),
                  nn.Linear(300, 200, bias=True),
                  nn.Linear(200, 100, bias=True),
                  nn.Linear(100, 50, bias=True),
                  nn.Linear(50, 10)]

        self.num_layers = len(layers)

        self.nonlinear = nn.ModuleList(layers)
        self.linear = nn.ModuleList(layers)
        self.gate = nn.ModuleList(layers)
        self.f = nn.Tanh()

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x



class DeepHighwayNetwork(nn.Module):
    def __init__(self, nin):

        super(DeepHighwayNetwork, self).__init__()

        layers = [nn.Linear(nin, 400, bias=True),
                  nn.Linear(400, 300, bias=True),
                  nn.Linear(300, 300, bias=True),
                  nn.Linear(300, 300, bias=True),

                  nn.Linear(300, 200, bias=True),
                  nn.Linear(200, 200, bias=True),
                  nn.Linear(200, 200, bias=True),

                  nn.Linear(200, 100, bias=True),
                  nn.Linear(100, 100, bias=True),
                  nn.Linear(100, 100, bias=True),

                  nn.Linear(100, 50, bias=True),
                  nn.Linear(50, 50, bias=True),
                  nn.Linear(50, 50, bias=True),

                  nn.Linear(50, 10)]

        self.num_layers = len(layers)

        self.nonlinear = nn.ModuleList(layers)
        self.linear = nn.ModuleList(layers)
        self.gate = nn.ModuleList(layers)
        self.f = nn.Tanh()

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x
