import torch
import torch.nn as nn 
from torch_sparse import spmm, matmul
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

import numpy as np


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
       
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x, adj, scale=0):
        x = torch.spmm(adj, x)
        x = torch.mm(x, self.weight)
        if self.bias is not None:
            return x + self.bias
        return x

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(self.in_features, self.out_features)


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
    """
    def __init__(self, nn, eps=0., train_eps=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset(self, value):
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self.reset(child)

    def reset_parameters(self):
        self.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, size=None):
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'


class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
               in_features, out_perhead, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func. 
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
                    self.in_features, self.heads, self.out_perhead)


class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        # init 
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu')) # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
         
    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze()) # E
        n = len(input)
        alpha = softmax(alpha, edge[0], n)
        output = spmm(edge, self.dropout(alpha), n, n, h) # h_prime: N x out
        return output


class NormLayer(nn.Module):
    def __init__(self, args):
        """
            mode:
              'None' : No normalization 
              'PN'   : PairNorm
              'PN-SI'  : Scale-Individually version of PairNorm
              'PN-SCS' : Scale-and-Center-Simultaneously version of PairNorm
              'LN': LayerNorm
              'CN': ContraNorm
        """
        super(NormLayer, self).__init__()
        self.mode = args.norm_mode
        self.scale = args.norm_scale
                
    def forward(self, x, adj=None, tau=1.0):
        if self.mode == 'None':
            return x
        if self.mode == 'LN':
            x = x - x.mean(dim=1, keepdim=True)
            x = nn.functional.normalize(x, dim=1)

        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean
            
        if self.mode == 'CN':
            norm_x = nn.functional.normalize(x, dim=1)
            sim = norm_x @ norm_x.T / tau
            if adj.size(1) == 2:
                sim[adj[0], adj[1]] = -np.inf
            else:
                sim.masked_fill_(adj.to_dense() > 1e-5, -np.inf)
            sim = nn.functional.softmax(sim, dim=1)
            x_neg = sim @ x    
            x = (1 + self.scale) * x - self.scale * x_neg   
        
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


"""
    helpers
"""

from torch_scatter import scatter_max, scatter_add


def softmax(src, index, num_nodes=None):
    """
        sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

