def train(net, optimizer, criterion, data, cal_erank=False, cal_metrics=False):
    net.train()
    optimizer.zero_grad()
    output, metrics = net(data.x, data.adj, cal_erank, cal_metrics)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc, metrics

def val(net, criterion, data):
    net.eval()
    output, metrics = net(data.x, data.adj, cal_erank=False, cal_metrics=False)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val

def test(net, criterion, data):
    net.eval()
    output, metrics = net(data.x, data.adj, cal_erank=False, cal_metrics=False)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


if __name__ == '__main__':
    import torch
    import numpy as np
    import torch.nn.functional as F
    from math import exp as exp

    x = torch.Tensor([[-np.inf, 1, 2, 3]])
    y = F.softmax(x)