# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:14:29 2020

@author: zhoubo
"""

import DataProcess as DP
import torch
import torch.nn as nn
import time
import d2lzh_pytorch as d2l
import numpy as np
import random

device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子
# setup_seed(20)

#=========================================模型=================================

class Vgg1d_net(nn.Module):#自己瞎胡搭建的一个1d-CNN模仿的类型vgg 结果四不像。。准确率什么的都一堆bug要调  主要是测试后面的反向传播代码有没有问题
    def __init__(self):
        super(Vgg1d_net,self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(in_channels =1,out_channels =32,kernel_size=20),#第一层大卷积 1024-20+1=1005
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size =6,stride =3),#1005-6/4+1=334

                                            nn.Conv1d(in_channels =32,out_channels =64,kernel_size=10),#334-10+1=325
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size=5,stride =2 ), #325-5/2+1=161

                                            nn.Conv1d(in_channels =64,out_channels =128,kernel_size=4),#161-4+1=158
                                            nn.ReLU(),
                                            nn.Conv1d(in_channels =128,out_channels =256,kernel_size=4),#158-4+1=155
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size=3,stride=2),#155-4/2+1 =77

                                            nn.Conv1d(in_channels =256,out_channels =512,kernel_size=4),#77-4+1=74
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size=2,stride=2),#74-2/2+1 =37
                                            )

        self.fc =nn.Sequential(nn.Linear(37*512,4096),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(4096,1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024,256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,10)
                                )
    def forward(self,X):
        features =self.conv1(X)
        features=features.view(X.shape[0],-1)#全连接层要把卷积层展开
        output =self.fc(features)
        return output


    #模型太大了 光全连接层参数  18944+4194304+2622144+2560 ----差不多满6G了  显存肯定要爆掉 batch size一大就爆显存
#=============================================================================




def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,
             legend =None,figsize=(3.5,2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:#意思都不同时为0
        d2l.plt.semilogy(x2_vals,y2_vals,linestyle=':')
        d2l.plt.legend(legend)



def evaluate_accuracy_2(data_iter, net,device=None):#用于评估测试集准确率 目的是要实现评估的时候要自动关闭dropout
    if device is None and isinstance(net,torch.nn.Module):#如果设备是none且net是由nn.module生成的实例。则：
          # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0#初始化准确的个数 ，样本数

    with torch.no_grad():#不追踪操作 因为这个是测试集评估的 默认模型训练好了也就不需要记录梯度
        for X, y in data_iter:#从data_iter取出一个batch的X,y

          #先判断你这个net是怎么产生的是你自己手写的还是利用pytorch快速生成的
            if isinstance(net, torch.nn.Module):#判断net是不是用torch.nn.Module创建的实例(判断net是不是利用module模块搭建的)

                net.eval() # #如果是上面方法创建的 那么开启评估模式 dropout层全部关闭(因为我们要是通过module模块创建一个模型有可能添加了dropout层)
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()#判断正确的个数
                net.train() # 改回训练模式
            else: # 如果是我们自定义的模型    else下面的这段主要是用于3.13节我们自定义带dropout的模型，计算准确率的以后不会用到 不考虑GPU
                print('不是继承nn.Module创建的')


            n += y.shape[0]
            print('acc_sum=:',acc_sum)
            print('n=',n)#其实就是算了以下一个批次有多少样本 每次循环累加一下参加计算的样本数
    return acc_sum / n#在所有批次循环后  计算准确率 拿 准确的个数/总个数





def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)#将模型添加到设备上 可能是GPU可能是CPU
    print("training on ", device)#显示是在哪训练的
    loss = torch.nn.CrossEntropyLoss()#使用交叉熵损失函数

    train_ls=[]
    test_ls=[]
    for epoch in range(num_epochs):#几个迭代周期 开始迭代
        #定义 训练集损失和,训练集准确总数，总样本数n,几个batch,开始时间
        train_l_batch_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()




        for X, y in train_iter:#一个批次中(比如一个batch_size是256)
            X = X.to(device)#先将需要计算的添加到 设备
            y = y.to(device)#同上

            y_hat = net(X)#计算模型预测值y_hat---->

            l = loss(y_hat, y.long())#计算损失（利用前面定义的交叉熵损失函数）

            optimizer.zero_grad()#优化器梯度清0
            l.backward()#误差反向传播
            optimizer.step()#权值更新
            train_l_batch_sum += l.cpu().item()# train_l_sum计算的是一个batch上的总误差，最后累加的train_l_sum需要除以总的batch数     计算得到的误差可能再GPU上先移动到CPU转成pyton数字
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()#一个batch上训练集正确个数  一个batch一个batch不断累加得到最后的总个数 最后要除以训练集样本总数n

            n += y.shape[0]
            batch_count += 1


        #一个epoch后 整个训练集/测试集的 loss   ----------画图用 不画图要注释掉
        with torch.no_grad():

            # train_ls.append(loss(net(train_features.cuda()), train_labels.cuda().long()).cpu().item())
            train_ls.append(train_l_batch_sum/n)
            test_ls.append( loss(net(test_features.cuda()), test_labels.cuda().long()).cpu().item() )#计算整个测试集上的误差

        #一个epoch后 对测试集准确率进行模型评估
        test_acc = evaluate_accuracy_2(test_iter, net)
        #打印 一个epoch上 测试集的平均loss  测试集的准确率  训练集的准确率 计算所用时间
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_batch_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    # print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])#与train_l_batch_sum / batch_count等价

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',range(1, num_epochs + 1), test_ls, ['train', 'test'])#--------画图用 不画图可以注释





net =Vgg1d_net()
print(net)

X =torch.randn(5,1,1024)
print(net(X))




batch_size =10
lr =0.001
num_epochs=10
train_iter,test_iter,train_features,train_labels,test_features,test_labels =DP.get_dataset_iter(batch_size)
optimizer =torch.optim.Adam(net.parameters(),lr,weight_decay=0)

train_model(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)











