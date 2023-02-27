import torch
from einops import rearrange
import pdb
import numpy as np
import torch.nn.functional as F


def calculate_patch_sim(x):
    """
    x size: [n_nodes, dim]
    return: average patch-wise similarity in a batch
    """
    n = x.size(0)
    norm_x = F.normalize(x, dim=-1)

    sim = norm_x @ norm_x.T     # [n_nodes, n_nodes]
    sim = torch.triu(sim, diagonal=1)
    sim = torch.sum(sim) / ((n**2 - n) / 2)
    return sim.item()


def calculate_erank(x):
    """
    x size: [n_nodes, dim]
    return: average erank in a batch
    """
    _, S, _ = torch.linalg.svd(x)
    N = torch.linalg.norm(S, ord=1, keepdim=True)
    S = S / N
    erank = torch.exp(torch.sum(-S * torch.log(S)))
    return erank.item()


def cal_variance(x):
    v = torch.sum(torch.var(x, dim=-1))
    return v.item()


if __name__ == '__main__':
    layers = 8
    all_sims = np.zeros(shape=layers)
    sims = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    all_sims += sims
    print(all_sims)

