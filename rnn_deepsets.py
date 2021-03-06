import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable
from set_transformer import *

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
        x = torch.sum(x, dim=2, keepdim=False)        
        #x = torch.quantile(x, 0.75, dim=2, keepdim=False)
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
        #print("decoder input {}".format(x.shape))
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


class RNNWithSetTransformerInitialization(nn.Module):
        def __init__(self, input_dim, rnn_hidden_dim, rnn_input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.rnn_hidden_dim = rnn_hidden_dim
            self.rnn_input_dim = rnn_input_dim

            self.set_transformer= SetTransformer(x_dim, 1, rnn_hidden_dim)
            self.gru = nn.GRU(input_dim, rnn_hidden_dim, batch_first=True)

        def forward(self, x):
            x = x.permute(0,2,1)
            #print("Input shape {}".format(x.shape))
            h0 = self.set_transformer(x).permute(1,0,2) 
            #print("Initial hidden shape: {}".format(h0.shape))
            #print("RNN input shape {}".format(x.shape))
            y, h = self.gru(x, h0) # call RNN with initial hidden state h0
            return y

class ConditionalRNNProcess(nn.Module):
        def __init__(self, input_dim, rnn_hidden_dim, rnn_input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.rnn_hidden_dim = rnn_hidden_dim
            self.rnn_input_dim = rnn_input_dim

            self.set_transformer= SetTransformer(x_dim, 1, rnn_hidden_dim)
            self.gru = nn.GRU(input_dim, rnn_hidden_dim, batch_first=True)

            self.mu_net = self.mlp([rnn_hidden_dim, 32,16, rnn_hidden_dim])
            self.sigma_net = self.mlp([rnn_hidden_dim, 16,16, 1])

        def mlp(self, architecture):
            prev_neurones = architecture[0]
            net = []
            for neurones in architecture[1:]:
                net.append(nn.Linear(prev_neurones, neurones))
                net.append(nn.Tanh())
                prev_neurones = neurones
            return nn.Sequential(*net)    

        def forward(self, x_meta, x):
            x_meta = x_meta.permute(0,2,1)
            h0 = self.set_transformer(x_meta).squeeze()
            h0 =  torch.sum(h0, dim=0, keepdim=True).unsqueeze(1)
            h0 = h0.repeat(x.shape[0],1,1).permute(1,0,2) 

            x = x.permute(0,2,1)
            y, h = self.gru(x, h0) 

            mu = self.mu_net(y)
            sigma = self.sigma_net(y)

            return mu, sigma


if __name__ == "__main__":
    nb_meta = 512
    nb = 32
    seq_len = 128
    x_dim = 256
    rnn_hidden_dim = 64
    rnn_input_dim = 48

    model = RNNWithDeepSetInitialization(x_dim, rnn_hidden_dim)
    x = torch.randn(nb,x_dim,seq_len)
    y = model(x)

    model1 = RNNWithSetTransformerInitialization(x_dim, rnn_hidden_dim, rnn_input_dim)
    x1 = torch.randn(nb,x_dim,seq_len)
    y1 = model1(x1)


    model2 = ConditionalRNNProcess(x_dim, rnn_hidden_dim, rnn_input_dim)
    x2 = torch.randn(nb,x_dim,seq_len)
    x_cond = torch.randn(nb_meta,x_dim,seq_len)

    #y2 = model2(x_cond, x1)
    #x_cond = torch.randn(15,x_dim,seq_len)

    mu, sigma = model2(x_cond, x1)

    print("Deep set based:")
    print("RNN output shape {}".format(y.shape))
    print("Parameter count: {}".format(count_parameters(model)))
    print("Parameter count gru only: {}".format(count_parameters(model.gru)))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Set transformer based:")
    print("RNN output shape {}".format(y1.shape))
    print("Parameter set transformer only: {}".format(count_parameters(model1.set_transformer)))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Conditional RNN process:")
    print("RNN output shape {} + {}".format(mu.shape, sigma.shape))
    print("Parameter set transformer only: {}".format(count_parameters(model2.set_transformer)))    

