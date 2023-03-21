# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from .soft_decay import soft_decay_function
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def apply_mask(x,p):
    r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]
    x_prime = r * x
    return x_prime

def vis_tokenUni(tokens,picname,ifpca):
    # time_start = time.time()
    batch_id = 0
    tokens = tokens[batch_id].cpu().detach().numpy()
    if ifpca:
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(tokens)
        tokens = pca_result_50
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tokens)
    d = {'tsne-2d-one': tsne_results[:,0], 'tsne-2d-two':tsne_results[:,1]}
    df_subset = pd.DataFrame(data=d)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    # palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=None,
    )
    plt.savefig(picname)

class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True,llayer = False):
        super(LayerNormImpl, self).__init__()
        self.norm_mode = args.lnv
        self.hidden = 768
        self.decay_alpha = -0.2
        self.ifmask = False
        self.soft_decay = soft_decay_function(in_features=args.max_length,alpha=None,hidden_dim=self.hidden,decay_alpha=self.decay_alpha,ifmask=self.ifmask)
        self.exrank_nonlinear = nn.ReLU()
        self.rescale_weight = nn.Parameter(torch.Tensor(args.max_seq_length,hidden))

        self.logbase = Variable(torch.ones(1), requires_grad=True).cuda()

        if self.norm_mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_parameter('rescale_weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

            nn.init.ones_(self.rescale_weight)

    def forward(self, input):
        if self.norm_mode == 'origin':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input-mean)/(std+self.eps)
            output = self.weight*input_norm+self.bias
            return (output,self.weight)
        elif self.norm_mode == 'soft_decay':
            u,s,v = torch.svd(input)
            maxS = torch.max(s,dim=1).values.unsqueeze(-1)
            newS,alpha = self.soft_decay(input,s)#[8,128]
            maxNewS = torch.max(newS,dim=1).values.unsqueeze(-1)
            #make the maxS unchanged
            rescale_number = maxNewS/maxS
            newS = newS/rescale_number
            rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
            return (new_input,alpha)



def NormFuncs(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None):
    if args is not None:
        return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
