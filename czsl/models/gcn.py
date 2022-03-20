import numpy as np
import scipy.sparse as sp

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx

def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.layer = nn.Linear(in_channels, out_channels)

        if relu:
            # self.relu = nn.LeakyReLU(negative_slope=0.2)
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.layer.weight.T)) + self.layer.bias

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN(nn.Module):

    def __init__(self, adj, in_channels, out_channels, hidden_layers):
        super().__init__()

        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj.to(device)

        self.train_adj = self.adj

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x):
        if self.training:
            for conv in self.layers:
                x = conv(x, self.train_adj)
        else:
            for conv in self.layers:
                x = conv(x, self.adj)
        return F.normalize(x)

### GCNII
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=False, relu=True, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

        self.out_features = out_features
        self.residual = residual
        self.layer = nn.Linear(self.in_features, self.out_features, bias = False)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        if self.dropout is not None:
            input = self.dropout(input)

        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support

        mm_term = torch.mm(support, self.layer.weight.T)

        output = theta*mm_term+(1-theta)*r
        if self.residual:
            output = output+input

        if self.relu is not None:
            output = self.relu(output)

        return output

class GCNII(nn.Module):
    def __init__(self, adj, in_channels , out_channels, hidden_dim, hidden_layers, lamda, alpha, variant, dropout = True):
        super(GCNII, self).__init__()

        self.alpha = alpha
        self.lamda = lamda

        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj.to(device)

        i = 0
        layers = nn.ModuleList()
        self.fc_dim = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        for i, c in enumerate(range(hidden_layers)):
            conv = GraphConvolution(hidden_dim, hidden_dim, variant=variant, dropout=dropout)
            layers.append(conv)

        self.layers = layers
        self.fc_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        _layers = []
        layer_inner = self.relu(self.fc_dim(self.dropout(x)))
        # layer_inner = x
        _layers.append(layer_inner)

        for i,con in enumerate(self.layers):
            layer_inner = con(layer_inner,self.adj,_layers[0],self.lamda,self.alpha,i+1)
        
        layer_inner = self.fc_out(self.dropout(layer_inner))
        return layer_inner