# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:39:03 2020

@author: zhoubo
"""

#TestModel3
import DataProcess as DP
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import myplot
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import myutils


device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')
def train_model(net, train_iter,test_iter, batch_size, optimizer, device, num_epochs,conf_matrix):
    net = net.to(device)#将模型添加到设备上 可能是GPU可能是CPU
    print("training on ", device)#显示是在哪训练的
    loss = torch.nn.CrossEntropyLoss()#使用交叉熵损失函数

    train_yvalues=[]
    test_yvalues=[]

    train_acc =[]
    test_acc=[]

    for epoch in range(num_epochs):#几个迭代周期 开始迭代
        #定义 训练集损失和,训练集准确总数，总样本数n,几个batch,开始时间
        train_l_sum, train_acc_sum,n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        net.train()
        for X, y in train_iter:#一个批次中(比如一个batch_size是256)
            X = X.to(device)#先将需要计算的添加到 cuda
            y = y.to(device)#同上

            y_hat = net(X)#计算模型预测值y_hat---->

            l = loss(y_hat, y.long())#计算损失（利用前面定义的交叉熵损失函数）

            optimizer.zero_grad()#优化器梯度清0
            l.backward()#误差反向传播
            optimizer.step()#梯度更新
            train_l_sum += l.cpu().item()#计算得到的误差可能再GPU上先移动到CPU转成pyton数字
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()#训练集正确总数

            n += y.shape[0]
            batch_count += 1
        train_yvalues.append(train_l_sum/batch_count)
        train_acc.append(train_acc_sum / n)
        print('epoch %d, train loss %.4f, train acc %.3f, time %.1f sec'% (epoch + 1, train_l_sum/batch_count, train_acc_sum / n,time.time() - start),end='')

        test_acc_sum,test_l_sum,test_batch_count,test_n = 0.0,0.0,0,0#初始化准确的个数 ，样本数



        with torch.no_grad():
            for X,y in test_iter:
                net.eval()#关闭dropout 进入评估模式
                X= X.to(device)
                y= y.to(device)




                test_acc_sum +=  (net(X).argmax(dim =1)==y) .float().sum().cpu().item()
                test_l_sum += loss(net(X),y.long()).cpu().item()

                test_batch_count += 1
                test_n+=y.shape[0]


                if epoch ==num_epochs-1:
                    conf_matrix =myplot.confusion_matrix(net(X).argmax(dim =1),y,conf_matrix)



            test_yvalues.append(test_l_sum/test_batch_count)
            test_acc.append(test_acc_sum / test_n)
            print(' test loss %.4f, test acc %.3f'% ( test_l_sum/test_batch_count, test_acc_sum / test_n))



    #画loss图
    # myplot.semilogy(range(1, num_epochs + 1), train_yvalues, 'epochs', 'loss', range(1, num_epochs + 1), test_yvalues, ['train', 'test'])
    myplot.semilogy(range(1, num_epochs + 1), train_acc, 'epochs', 'acc', range(1, num_epochs + 1), test_acc, ['train', 'test'])
    #画混淆矩阵图
    print(conf_matrix)
#直接试试5层感知机的效果吧

class FlattenLayer_1d(nn.Module):
    def __init__(self):
        super(FlattenLayer_1d, self).__init__()
        print('调用FLATTEN')
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

net =nn.Sequential(FlattenLayer_1d(),
                   nn.Linear(1024,2048),
                   # nn.Dropout(0.5),
                   nn.Linear(2048,1024),
                   nn.ReLU(),
                   nn.Linear(1024,512),
                   nn.ReLU(),
                   nn.Linear(512,10),
                   nn.ReLU(),
                   nn.Softmax()

                   )

# print(net)
myutils.print_model_parm_nums(net)
# X = torch.rand(10, 1,1024)
# print(net(X))

batch_size=1024
train_iter,test_iter=DP.get_train_test_iter(2000,1600,1024,batch_size)

lr =0.0001
num_epochs=100
optimizer =torch.optim.Adam(net.parameters(),lr,weight_decay=0)
confusion_matrix =torch.zeros(10,10).int()


train_model(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs,confusion_matrix)