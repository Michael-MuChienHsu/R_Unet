import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
import torch.autograd as autograd
import psutil
import gc

cuda_gpu = torch.cuda.is_available()

class LSTMM(nn.Sequential):
    def __init__(self):
        super(LSTMM, self).__init__()
        self.lstm = nn.LSTM(10, 30, batch_first=True)
        self.h0 = Variable(torch.randn(1, 50, 30))
        self.c0 = Variable(torch.randn(1, 50 ,30))
    
    def forward(self, x):
        out, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        out.detach()
        return out

def train():
    
    x = Variable(torch.randn((50, 100, 10)))
    target = Variable(torch.randn((1, 100, 30)))
    
    ## use class
    
    lstm = LSTMM()
    
    critiria = nn.SmoothL1Loss()
    optimizer = optim.Adam( lstm.parameters(), lr = 0.001 )
    
    for i in range(0, 2):
        for j in range(0, 10):
            optimizer.zero_grad()
            out = lstm(x)

            loss = critiria( out, target)

            loss.backward(retain_graph = True)

            process = psutil.Process(os.getpid())
            print('epoch', i, 'step', j, 'used memory', round((int(process.memory_info().rss)/(1024*1024)), 2), 'MB' )

    # direct use RNN
    lstm = nn.LSTM(10, 30, batch_first=True)

    critiria = nn.SmoothL1Loss()
    optimizer = optim.Adam( lstm.parameters(), lr = 0.001 )

    h0 = Variable(torch.randn(1, 50, 30))
    c0 = Variable(torch.randn(1, 50 ,30))

    for i in range(0, 2):
        for j in range(0, 10):
            optimizer.zero_grad()

            out, (h0, c0) = lstm(x, (h0, c0))

            loss = critiria( out, target)

            loss.backward(retain_graph = True)

            process = psutil.Process(os.getpid())
            print('epoch', i, 'step', j, 'used memory', round((int(process.memory_info().rss)/(1024*1024)), 2), 'MB' )

if __name__ == "__main__":
    train()
