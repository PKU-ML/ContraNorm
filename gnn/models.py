from cmath import log
import torch

from layers import *

import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_dense_adj

from metrics import calculate_erank, calculate_patch_sim, cal_variance
from collections import defaultdict


def log_metrics(metrics, x, pos, cal_erank=False):
    metrics[f'sim{pos}'].append(calculate_patch_sim(x)) 
    metrics[f'erank{pos}'].append(calculate_erank(x) if cal_erank else 0.) 
    metrics[f'var{pos}'].append(cal_variance(x))
    return metrics


def undirect_to_direct(adj, p=0.5):
    adj = adj.to_dense()
    mask_full = torch.rand_like(adj) > p
    mask_low = torch.tril(mask_full, diagonal=-1)
    mask_upper = torch.triu((~mask_low).T, diagonal=1)
    mask_all = torch.logical_or(mask_low, mask_upper)
    adj.masked_fill_(mask_all, 0)
    return adj.to_sparse()


def dropedge(adj, p=0.5):
    adj = adj.coalesce()
    # import pdb; pdb.set_trace()
    indices = adj.indices()
    n_node = len(indices[0])
    chosen_idx = torch.randperm(n_node, device=indices.device)[:int(p*n_node)]
    adj = torch.sparse_coo_tensor(indices[:, chosen_idx], adj.values()[chosen_idx], size=adj.size())
    return adj


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)
        self.norm = NormLayer(norm_mode, norm_scale)

    def forward(self, x, adj, cal_erank=False):
        metrics = dict()
        metrics = log_metrics(metrics, x, '1', cal_erank=cal_erank)

        x = self.dropout(x)
        x = self.gc1(x, adj)

        metrics = log_metrics(metrics, x, '2', cal_erank=cal_erank)

        x = self.norm(x)

        metrics = log_metrics(metrics, x, '3', cal_erank=cal_erank)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)

        metrics = log_metrics(metrics, x, '4', cal_erank=cal_erank)
        return x, metrics


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead, 
                 norm_mode='None', norm_scale=1,**kwargs):
        super(GAT, self).__init__()
        alpha_droprate = dropout
        self.gac1 = GraphAttConv(nfeat, nhid, nhead, alpha_droprate)
        self.gac2 = GraphAttConv(nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True) 
        self.norm = NormLayer(norm_mode, norm_scale)

    def forward(self, x, adj):
        x = self.dropout(x) 
        x = self.gac1(x, adj)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gac2(x, adj)
        return x


class SGC(nn.Module):
    # for SGC we use data without normalization
    def __init__(self, args, nfeat, nclass, **kwargs):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)
        self.dropout = nn.Dropout(p=args.dropout)
        self.nlayer = args.nlayer
        self.norm = NormLayer(args)
        self.initial_layer_norm = nn.LayerNorm(nfeat)
        self.layer_norm = nn.LayerNorm(args.hid)
        self.use_layer_norm = args.use_layer_norm  
        self.initial_norm = args.initial_norm    
        
    def forward(self, x, adj, cal_erank=False, cal_metrics=False):
        if self.initial_norm:
            x = self.norm(x, adj)
            if self.use_layer_norm:
                x = self.initial_layer_norm(x)
        metrics = defaultdict(list)
        for _ in range(self.nlayer):
            metrics = log_metrics(metrics, x, '1', cal_erank=cal_erank) if cal_metrics else metrics
            x = adj.mm(x)
            metrics = log_metrics(metrics, x, '2', cal_erank=cal_erank) if cal_metrics else metrics
            x = self.norm(x, adj)
            if self.use_layer_norm:
                x = self.layer_norm(x)
            metrics = log_metrics(metrics, x, '3', cal_erank=cal_erank) if cal_metrics else metrics  
        x = self.dropout(x)
        x = self.linear(x)
        return x, metrics


class DeepGCN(nn.Module):
    def __init__(self, args, nfeat, nclass, **kwargs):
        super(DeepGCN, self).__init__()
        assert args.nlayer >= 1 

        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i==0 else args.hid, args.hid) 
            for i in range(args.nlayer-1)
        ])
        self.out_layer = GraphConv(nfeat if args.nlayer==1 else args.hid , nclass)

        self.dropout = nn.Dropout(p=args.dropout)
        self.dropout_rate = args.dropout
        self.relu = nn.ReLU(True)
        self.norm = NormLayer(args)
        self.layer_norm = nn.LayerNorm(args.hid)
        self.skip = args.residual
        self.use_layer_norm = args.use_layer_norm

    def forward(self, x, adj, cal_erank=False, cal_metrics=False):

        x_old = 0
        metrics = defaultdict(list)
        for i, layer in enumerate(self.hidden_layers):
            metrics = log_metrics(metrics, x, '1', cal_erank=cal_erank) if cal_metrics else metrics

            x = self.dropout(x)
            x = layer(x, adj)
            metrics = log_metrics(metrics, x, '2', cal_erank=cal_erank) if cal_metrics else metrics

            x = self.norm(x, adj)
            metrics = log_metrics(metrics, x, '3', cal_erank=cal_erank) if cal_metrics else metrics

            if self.use_layer_norm:
                x = self.layer_norm(x)

            x = self.relu(x)
            if self.skip > 0 and i % self.skip==0:
                x = x + x_old
                x_old = x
            
            metrics = log_metrics(metrics, x, '4', cal_erank=cal_erank) if cal_metrics else metrics
         
        
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x, metrics


class DeepGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0, nhead=1,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGAT, self).__init__()
        assert nlayer >= 1 
        alpha_droprate = dropout
        self.hidden_layers = nn.ModuleList([
            GraphAttConv(nfeat if i==0 else nhid, nhid, nhead, alpha_droprate)
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphAttConv(nfeat if nlayer==1 else nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)
        self.norm = NormLayer(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip>0 and i%self.skip==0:
                x = x + x_old
                x_old = x
                
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, norm_mode='None', norm_scale=1.0, **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.norm_mode = norm_mode

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True).jittable()

            self.convs.append(conv)
            if norm_mode == 'BN':
                self.norms.append(BatchNorm(hidden_channels))
            else:
                self.norms.append(NormLayer(norm_mode, norm_scale))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            if self.norm_mode == 'BN':
                x = F.relu(norm(conv(x, edge_index)))
            else:
                x = F.relu(norm(conv(x, edge_index), torch.squeeze(to_dense_adj(edge_index, batch=None)).to_sparse()))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
