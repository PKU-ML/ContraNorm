import os, torch, logging, argparse
import models
import torch.nn as nn
from utils import train, test, val
from data import load_data
import pickle

# out dir 
OUT_PATH = "results/"
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

# parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='DeepGCN', help='{SGC, DeepGCN, DeepGAT, GIN}')
parser.add_argument('--hid', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--nhead', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--residual', type=int, default=0, help='Residual connection')

# for normalization
parser.add_argument('--norm_mode', type=str, default='None', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS, CN}')
parser.add_argument('--norm_scale', type=float, default=1.0, help='Row-normalization scale')
parser.add_argument('--use_layer_norm', action='store_true')
parser.add_argument('--initial_norm', action='store_true')

# for data
parser.add_argument('--no_fea_norm', action='store_false', default=True, help='not normalize feature' )
parser.add_argument('--missing_rate', type=int, default=0, help='missing rate, from 0 to 100' )

args = parser.parse_args()

# logger
logging.basicConfig(format='%(message)s', level=getattr(logging, args.log.upper())) 

# load data
data = load_data(args.data, normalize_feature=args.no_fea_norm, missing_rate=args.missing_rate, cuda=True)
nfeat = data.x.size(1)
nclass = int(data.y.max()) + 1

best_acc = 0 
best_loss = 1e10
all_test_acc = []
for i in range(5):
    net = getattr(models, args.model)(args, nfeat, nclass)
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        cal_erank = False if epoch == args.epochs - 1 else False
        cal_metrics = False if epoch == args.epochs - 1 else False
        train_loss, train_acc, metrics = train(net, optimizer, criterion, data, cal_erank=cal_erank, cal_metrics=cal_metrics)
        val_loss, val_acc = val(net, criterion, data)
        if epoch == args.epochs - 1 and cal_metrics:
            with open(f"results/{args.data}-{args.norm_mode}-{args.hid}-{args.norm_scale}-{args.nlayer}.pkl", 'wb') as f:
                pickle.dump(metrics, f)
        # save model 
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-acc.pkl')
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-loss.pkl')

    # pick up the best model based on val_acc, then do test
    val_loss, val_acc = val(net, criterion, data)
    test_loss, test_acc = test(net, criterion, data)
    all_test_acc.append(test_acc.item())

import numpy as np

all_test_acc = np.array(all_test_acc)
print(all_test_acc.mean(), all_test_acc.std())
