import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch


def calculate_erank(S):
    """
    x size: [batch_size, max_length, dim]
    return: average erank in a batch
    """
    N = torch.linalg.norm(S, ord=1)
    S = S / N
    erank = torch.exp(torch.sum(-S * torch.log(S)))
    return erank.item()


dir = "./output/"
ori_f = dir + "rte-singulars-0.0-7-12-epoch5.pkl"
cn_f = dir + "rte-singulars-1.0-7-12-epoch5.pkl"
ex_f = dir + "rte-sf-singulars-1.0-7-12-epoch5.pkl"


with open(ori_f, 'rb') as f:
    ori_S = pickle.load(f)
with open(cn_f, 'rb') as f:
    cn_S = pickle.load(f)
with open(ex_f, 'rb') as f:
    ex_S = pickle.load(f)

for k, v in ori_S.items():
    ori_S[k] = np.mean(v, axis=0)
for k, v in cn_S.items():
    cn_S[k] = np.mean(v, axis=0)
for k, v in ex_S.items():
    ex_S[k] = np.mean(v, axis=0)


L = 's-12'
ori_erank = calculate_erank(torch.FloatTensor(ori_S[L]))
cn_erank = calculate_erank(torch.FloatTensor(cn_S[L]))
print(f'original erank: {ori_erank}, contranorm erank: {cn_erank}')

plt.figure(figsize=(7, 5))
max_s = np.max(ori_S[L] + cn_S[L] + ex_S[L])
min_s = np.min(ori_S[L] + cn_S[L] + ex_S[L])
bins = 20
plt.xlim(0, 150)
plt.tick_params(labelsize=12)
plt.hist([ori_S[L], cn_S[L], ex_S[L]], bins, density=False, label=['BERT', '+ ContraNorm with $D^{-1}A$', '+ ContraNorm with $AD^{-1}$'], color=['green', 'orange', 'crimson'], alpha=0.8, edgecolor='slategrey') # 'mediumblue', 'crimson'
plt.xlabel('singular value', fontdict={'size': 18})
plt.ylabel('number', fontdict={'size': 18})
plt.subplots_adjust(bottom=0.13, top=0.92, left=0.14, right=0.92)
plt.legend(prop={'size': 16})
plt.grid(axis='y')
# plt.show()
plt.savefig('./visual/singular_distribution_ablation.pdf')
