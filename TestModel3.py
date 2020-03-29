# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:23:28 2020

@author: zhoubo
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import DataProcess as DP
import myutils
import myplot
from torch.nn import init
import time

device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')

class FlattenLayer_1d(nn.Module):
    def __init__(self):
        super(FlattenLayer_1d, self).__init__()
        print('调用FLATTEN')
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)



class MLP(nn.Module):
    def  __init__(self):
        super(MLP,self).__init__()
        self.FC1 =nn.Sequential(

                                nn.Linear(1024,2048),
                                nn.BatchNorm1d(2048),
                                nn.ReLU(),
                                nn.Linear(2048,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,10),
                                nn.BatchNorm1d(10),
                                nn.ReLU(),
                                # nn.Softmax(dim=1)
                                )
    def forward(self,X):
        return self.FC1(X)


# X=torch.randn(10,1,1024)
net =nn.Sequential( FlattenLayer_1d(),MLP())
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

batch_size =256
weight_decay=0
lr=0.01
num_epochs =150



train_features,train_labels,test_features,test_labels =DP.get_features_and_labels(2000, 1600, 1024)

train_dataset =torch.utils.data.TensorDataset(train_features,train_labels)
train_iter =torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
test_dataset =torch.utils.data.TensorDataset(test_features,test_labels)
test_iter =torch.utils.data.DataLoader(test_dataset,batch_size ,shuffle =True)

train_ls, test_ls,train_acc,test_acc,net_record,best_conf_matrix=myutils.train(net,train_features,train_labels,test_features,test_labels,num_epochs,lr,weight_decay,device,batch_size)

train_loss= train_ls[-1]#对存放train_ls的列表最后一个数拿出来  最后一个epoch的loss
test_loss = test_ls[-1]

myplot.draw_loss(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                      range(1, num_epochs + 1), test_ls,
                      ['train', 'test'])
myplot.draw_accuracy(range(1, num_epochs + 1), train_acc, 'epochs', 'acc',
                      range(1, num_epochs + 1), test_acc,
                      ['train', 'test'])


print('train loss %f, valid loss %f' % ( train_loss, test_loss))
print(' train acc %f, valid acc %f' % ( train_acc[-1], test_acc[-1]))

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
# def evaluate_accuracy(data_iter, net, device=None):
#     if device is None and isinstance(net, torch.nn.Module):
#         # 如果没指定device就使用net的device
#         device = list(net.parameters())[0].device
#     acc_sum, n = 0.0, 0
#     with torch.no_grad():
#         for X, y in data_iter:
#             if isinstance(net, torch.nn.Module):
#                 net.eval() # 评估模式, 这会关闭dropout
#                 acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
#                 net.train() # 改回训练模式

#             n += y.shape[0]
#     return acc_sum / n


# # 本函数已保存在d2lzh_pytorch包中方便以后使用
# def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
#     net = net.to(device)
#     print("training on ", device)
#     loss = torch.nn.CrossEntropyLoss()
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
#         for X, y in train_iter:
#             X = X.to(device)
#             y = y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y.long())
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
#               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))






# optimizer =torch.optim.Adam(net.parameters(),lr)

# train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)



