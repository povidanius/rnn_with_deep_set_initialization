import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable


NetIO = Union[FloatTensor, Variable]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  

class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)



        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        print("encoder")
        print(x.shape)
        x = torch.sum(x, dim=2, keepdim=False)        
        #x = torch.quantile(x, 0.75, dim=2, keepdim=False)

        print("representatioon shape")
        print(x.shape)

        # compute the output
        out = self.rho.forward(x)

        return out


class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)        
        self.fc2 = nn.Linear(20, output_dim)

        self.input_dim = input_dim
        self.output_dim  = output_dim

    def forward(self, x: NetIO) -> NetIO:
        #print("shape input {}".format(x.shape))

        seq_len = x.shape[2]

        x = x.reshape((-1, self.input_dim))
        #print("Shape before mlp : {}".format(x.shape))

        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))

        #print("Shape after mlp : {}".format(x.shape))

        x = x.reshape((-1, self.output_dim, seq_len))

        #print("encoding shape {}".format(x.shape))
        return x

class DeepSetDecoder(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.output_size)

    def forward(self, x: NetIO) -> NetIO:
        print("decoder input {}".format(x.shape))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RNNWithDeepSetInitialization(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        phi = DeepSetEncoder(input_dim, 32)
        rho = DeepSetDecoder(input_size=32, output_size=rnn_hidden_dim)
        self.deepset = InvariantModel(phi=phi, rho=rho)    
        self.gru = nn.GRU(input_dim, rnn_hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = self.deepset(x).unsqueeze(2).permute(2,0,1) # feed X to deep set, to get h0 of RNN
        #print("Initial hidden shape: {}".format(h0.shape))
        x = x.permute(0,2,1)
        #print("RNN input shape {}".format(x.shape))
        y, h = self.gru(x, h0) # call RNN with initial hidden state h0
        return y

        
if __name__ == "__main__":
    nb = 32
    seq_len = 128
    x_dim = 256
    rnn_hidden_dim = 64

    model = RNNWithDeepSetInitialization(x_dim, rnn_hidden_dim)
    x = torch.randn(nb,x_dim,seq_len)
    y = model(x)

    print("RNN output shape {}".format(y.shape))
    print("Parameter count: {}".format(count_parameters(model)))
    print("Parameter count gru only: {}".format(count_parameters(model.gru)))


    # possible recurssion in h0... 
