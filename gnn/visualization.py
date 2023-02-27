from copyreg import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib as mpl

import seaborn as sns
import pickle

mpl.use('TkAgg')


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


MARKERS = ['o', 'D', 'x', 'v', '^', '<', '>', '+']

COLORS = [color.TABLEAU_COLORS['tab:blue'], color.TABLEAU_COLORS['tab:orange'],
          color.TABLEAU_COLORS['tab:green'], color.TABLEAU_COLORS['tab:red'],
          color.TABLEAU_COLORS['tab:purple'], color.TABLEAU_COLORS['tab:brown'],
          color.TABLEAU_COLORS['tab:pink'], color.TABLEAU_COLORS['tab:gray'],
          color.TABLEAU_COLORS['tab:olive'], color.TABLEAU_COLORS['tab:cyan']]

BG_COLORS = [color.TABLEAU_COLORS['tab:blue'], color.TABLEAU_COLORS['tab:orange'],
             color.TABLEAU_COLORS['tab:green'], color.TABLEAU_COLORS['tab:red'],
             color.CSS4_COLORS['lightcoral'], color.CSS4_COLORS['mediumaquamarine'],
             color.CSS4_COLORS['gold'], color.CSS4_COLORS['cornflowerblue'],
             color.CSS4_COLORS['mediumslateblue'], color.CSS4_COLORS['lightskyblue']]


def plot_acc_over_smoothing(dataset):
    sns.set_style("whitegrid")

    if dataset == "cora":
        layer_num = ['2', '4', '8', '16', '32', '64']

        PN = [82.58, 82.02, 75.94, 55.72, 54.82, 55.34]
        LN = [84.42, 83.62, 83.40, 31.36, 31.12, 30.52]
        IN = [82.82, 81.66, 32.80, 26.02, 19.16, 13.00]
        CN = [84.76, 84.06, 83.84, 81.86, 75.14, 60.34]
        NN = [82.22, 75.82, 31.90, 24.82, 15.80, 13.00]

    elif dataset == "citeseer":
        layer_num = ['2', '4', '8', '16', '32', '64']

        PN = [72.42, 70.12, 63.94, 42.56, 40.92, 36.38]
        LN = [72.76, 69.86, 69.42, 28.18, 19.62, 19.42]
        IN = [73.76, 70.10, 30.42, 27.54, 20.70, 7.70]
        CN = [72.32, 70.70, 70.98, 68.50, 54.50, 43.32]
        NN = [73.40, 67.46, 28.50, 23.58, 16.82, 7.7]

    elif dataset == 'IMDB-BINARY':
        layer_num = ['2', '4', '8', '16', '32', '64', '128']

        NN = [84, 75, 73, 69, 56, 43, 49]
        LN = [82, 68, 53, 51, 53, 53, 62]
        PN = [84, 80, 73, 83, 64, 72, 76]
        CN = [80, 84, 81, 78, 75, 79, 75]
        BN = [77, 81, 77, 73, 76, 74, 77]

    elif dataset == 'MUTAG':
        layer_num = ['2', '4', '8', '16', '32', '64', '128']

        NN = [100, 88.89, 94.44, 100, 88.89, 88.89, 100]
        LN = [100, 100, 94.44, 72.22, 83.33, 88.89, 88.89]
        PN = [88.89, 94.44, 94.44, 94.44, 94.44, 100, 88.89]
        CN = [94.44, 100, 100, 100, 100, 100, 100]
        BN = [88.89, 94.44, 100, 88.89, 88.89, 83.33, 94.44]

    elif dataset == 'ENZYMES':
        layer_num = ['2', '4', '8', '16', '32', '64', '128']

        NN = [31.67, 26.67, 31.67, 25.00, 31.67, 25.00, 23.33]
        LN = [28.33, 26.67, 20.00, 21.67, 16.67, 20.00, 20.00]
        PN = [46.67, 41.67, 38.33, 26.67, 33.33, 31.67, 26.67]
        CN = [55.00, 50.00, 43.33, 40.00, 41.67, 35.00, 30.00]
        BN = [25.00, 26.67, 33.33, 16.67, 26.67, 20.00, 13.00]

    elif dataset == 'PROTEINS':
        layer_num = ['2', '4', '8', '16', '32', '64', '128']

        NN = [75.68, 77.48, 73.87, 79.28, 77.48, 80.18, 79.28]
        LN = [72.97, 81.08, 78.38, 69.37, 72.07, 72.97, 72.07]
        PN = [72.97, 68.49, 74.77, 76.58, 81.08, 76.58, 74.77]
        CN = [84.68, 82.88, 84.68, 80.18, 82.88, 82.88, 82.88]
        BN = [70.27, 75.68, 72.07, 81.08, 79.28, 76.58, 76.58]

    else:

        raise ValueError

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(layer_num, PN, marker=MARKERS[0], color=COLORS[1], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
            label=f'PairNorm')
    ax.plot(layer_num, LN, marker=MARKERS[1], color=COLORS[2], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
            label=f'LayerNorm')
    ax.plot(layer_num, NN, marker=MARKERS[3], color=COLORS[5], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
            label=f'No norm')
    ax.plot(layer_num, CN, marker=MARKERS[4], color=COLORS[3], markersize=10, linewidth=3, alpha=0.8,
            label=f'ContraNorm')
    if dataset in ['cora', 'citeseer']:
        ax.plot(layer_num, IN, marker=MARKERS[2], color=COLORS[4], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
                label=f'InstanceNorm')
    else:
        ax.plot(layer_num, BN, marker=MARKERS[2], color=COLORS[4], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
                label=f'BatchNorm')

    ax.set_xlabel("Number of Layers", fontsize=24)
    # ax.grid(False, axis='x')
    ax.grid(True)
    ax.set_ylabel("Test Accuracy (%)", fontsize=24)
    ax.legend(loc='best', ncol=2, fontsize=20)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_ylim(bottom=0, top=110)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    plt.savefig(f"oversmoothing_{dataset}.pdf", bbox_inches='tight')
    plt.show()


def plot_similarity():
    data = 'cora'
    nlayer = 12
    scale = 0.1
    npos = 4
    ori_f = f'results/{data}-None-0.0-{nlayer+1}.pkl'
    cn_f = f'results/{data}-CN-{scale}-{nlayer+1}.pkl'
    
    with open(ori_f, 'rb') as f:
        ori_metrics = pickle.load(f)

    for pos in range(npos):
        plt.figure(figsize=(7, 5.5))
        plt.plot(range(nlayer), ori_metrics[f'sim{pos+1}'], label=f'feature', 
                 color='yellowgreen', linewidth=4, 
                 marker='o', markersize=8, markerfacecolor='darkgreen', markeredgewidth=0, markeredgecolor='darkgreen')
        # plt.plot(range(layer), ori_metrics['sim_a'][:, 1], label=f'attention map',
        #          color='orange', linewidth=4, marker='s', 
        #          markersize=8, markerfacecolor='red', markeredgewidth=0, markeredgecolor='red')

        plt.ylim(0, 1.1)
        # plt.hlines(max(ori_metrics['sim_f'][:, pos]), 0, 12, colors='grey', linestyles='dashed')
        plt.grid(axis='y')
        plt.legend(prop={'size': 20})
        plt.xlabel('layer', fontdict={'size': 22,})
        plt.ylabel('cosine similarity', fontdict={'size': 22,})  # 'weight': 'bold'
        # plt.title('BERT', fontdict={'size': 16, 'weight': 'bold'})
        # plt.show()
        plt.savefig(f'results/fig/gcn-{data}-similarity-pos{pos+1}.pdf')



def plot_accuracy(dataset):
    sns.set_style("whitegrid")

    if dataset == "cora":
        layer_num = ['2', '4', '8', '16', '32']

        NN = [81.75, 72.61, 17.71, 20.71, 19.69]
        PN = [75.32, 72.64, 71.86, 54.11, 36.62]
        LN = [79.96, 77.45, 39.09, 7.79, 7.79]
        CN = [79.75, 77.02, 74.01, 68.75, 46.39]
        

    elif dataset == "citeseer":
        layer_num = ['2', '4', '8', '16', '32']
        
        NN = [69.18, 55.01, 19.65, 19.65, 19.30]
        PN = [61.59, 53.01, 55.76, 44.21, 36.68]
        LN = [63.27, 60.91, 33.74, 19.65, 19.65]
        CN = [64.06, 60.55, 59.30, 49.01, 36.94]
        

    elif dataset == 'chameleon':
        layer_num = ['2', '4', '8', '16', '32']

        NN = [45.79, 37.85, 22.37, 22.37, 22.37]
        PN = [62.24, 58.38, 49.12, 37.54, 30.66]
        LN = [63.95, 55.79, 34.08, 22.37, 22.37]
        CN = [64.78, 58.73, 48.99, 40.92, 35.44]

    elif dataset == 'squirrel':
        layer_num = ['2', '4', '8', '16', '32']

        NN = [29.47, 19.31, 19.31, 19.31, 19.31]
        PN = [43.86, 40.25, 36.03, 29.55, 29.05]
        LN = [43.04, 29.64, 19.63, 19.96, 19.40]
        CN = [47.24, 40.31, 35.85, 32.37, 27.80]

    elif dataset == 'ENZYMES':
        layer_num = ['2', '4', '8', '16', '32', '64', '128']

        NN = [31.67, 26.67, 31.67, 25.00, 31.67, 25.00, 23.33]
        PN = [46.67, 41.67, 38.33, 26.67, 33.33, 31.67, 26.67]
        CN = [55.00, 50.00, 43.33, 40.00, 41.67, 35.00, 30.00]

    elif dataset == 'PROTEINS':
        layer_num = ['2', '4', '8', '16', '32', '64', '128']

        NN = [75.68, 77.48, 73.87, 79.28, 77.48, 80.18, 79.28]
        PN = [72.97, 68.49, 74.77, 76.58, 81.08, 76.58, 74.77]
        CN = [84.68, 82.88, 84.68, 80.18, 82.88, 82.88, 82.88]

    else:

        raise ValueError

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(layer_num, PN, marker=MARKERS[0], color=COLORS[1], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
            label=f'PairNorm')
    ax.plot(layer_num, NN, marker=MARKERS[3], color=COLORS[5], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
            label=f'No norm')
    ax.plot(layer_num, LN, marker=MARKERS[1], color=COLORS[2], markersize=10, linewidth=3, linestyle='--', alpha=0.8,
            label=f'LayerNorm')
    ax.plot(layer_num, CN, marker=MARKERS[4], color=COLORS[3], markersize=10, linewidth=3, alpha=0.8,
            label=f'ContraNorm')

    ax.set_xlabel("Number of Layers", fontsize=24)
    ax.grid(False, axis='x')
    # ax.grid(True)
    ax.set_ylabel("Test Accuracy (%)", fontsize=24)
    ax.legend(loc='best', ncol=2, fontsize=20)
    ax.set_yticks(np.arange(0, 100, 10))
    ax.set_ylim(bottom=0, top=100)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    plt.savefig(f"results/fig/accuracy_{dataset}.pdf", bbox_inches='tight')
    # plt.show()


def plot_similarity():
    data = 'cora'
    nlayer = 12
    scale = 0.1
    npos = 4
    ori_f = f'results/{data}-None-0.0-{nlayer+1}.pkl'
    cn_f = f'results/{data}-CN-{scale}-{nlayer+1}.pkl'
    
    with open(ori_f, 'rb') as f:
        ori_metrics = pickle.load(f)

    for pos in range(npos):
        plt.figure(figsize=(7, 5))
        plt.plot(range(nlayer), ori_metrics[f'sim{pos+1}'], label=f'feature', 
                 color='yellowgreen', linewidth=5, 
                 marker='o', markersize=8, markerfacecolor='darkgreen', markeredgewidth=0, markeredgecolor='darkgreen')
        # plt.plot(range(layer), ori_metrics['sim_a'][:, 1], label=f'attention map',
        #          color='orange', linewidth=4, marker='s', 
        #          markersize=8, markerfacecolor='red', markeredgewidth=0, markeredgecolor='red')

        # plt.ylim(0, 1)
        # plt.hlines(max(ori_metrics['sim_f'][:, pos]), 0, 12, colors='grey', linestyles='dashed')
        plt.grid(axis='y')
        plt.legend(prop={'size': 20})
        plt.xlabel('layer index', fontdict={'size': 22,})
        plt.ylabel('cosine similarity', fontdict={'size': 22,})  # 'weight': 'bold'
        plt.tick_params(labelsize=14)
        plt.subplots_adjust(bottom=0.13, top=0.92, left=0.14, right=0.92)
        # plt.title('BERT', fontdict={'size': 16, 'weight': 'bold'})
        # plt.show()
        plt.savefig(f'results/fig/gcn-{data}-similarity-pos{pos+1}.pdf')


def plot_erank():
    data = 'cora'
    nlayer = 12
    scale = 1.0
    npos = 4
    ori_f = f'results/{data}-None-0.0-{nlayer+1}.pkl'
    cn_f = f'results/{data}-CN-{scale}-{nlayer+1}.pkl'

    with open(ori_f, 'rb') as f:
        ori_metrics = pickle.load(f)
    with open(cn_f, 'rb') as f:
        cn_metrics = pickle.load(f)

    for pos in range(4):
        plt.figure(figsize=(7, 5))
        plt.plot(range(nlayer), ori_metrics[f'erank{pos+1}'], label=f'No norm', 
                 color='steelblue', linewidth=5, linestyle='dashed', 
                 marker='o', markersize=8, markerfacecolor='darkslategrey', markeredgewidth=0, markeredgecolor='darkslategrey')
        plt.plot(range(nlayer), cn_metrics[f'erank{pos+1}'], label=f'ContraNorm',
                 color='hotpink', linewidth=5, marker='s', 
                 markersize=8, markerfacecolor='deeppink', markeredgewidth=0, markeredgecolor='deeppink')

        # plt.ylim(0, 1)
        # plt.hlines(max(ori_metrics['sim_f'][:, pos]), 0, 12, colors='grey', linestyles='dashed')
        plt.grid(axis='y')
        plt.legend(prop={'size': 20})
        plt.xlabel('layer index', fontdict={'size': 22,})
        plt.ylabel('effective rank', fontdict={'size': 22,})  # 'weight': 'bold'
        plt.tick_params(labelsize=14)
        plt.subplots_adjust(bottom=0.13, top=0.92, left=0.14, right=0.92)
        # plt.title('BERT', fontdict={'size': 16, 'weight': 'bold'})
        # plt.show()
        plt.savefig(f'results/fig/gcn-{data}-erank-pos{pos+1}.pdf')

plot_similarity()
# plot_erank()
#for d in ['cora', 'citeseer', 'chameleon', 'squirrel']:
#    plot_accuracy(d)