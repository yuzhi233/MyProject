# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:15:34 2020

@author: zhoubo
"""

import DataProcess as DP
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import myplot
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')


def train_model(net, train_iter,test_iter, batch_size, optimizer, device, num_epochs,conf_matrix):
    net.float()
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
    myplot.draw_loss(range(1, num_epochs + 1), train_yvalues, 'epochs', 'loss', range(1, num_epochs + 1), test_yvalues, ['train', 'test'])
    myplot.draw_accuracy(range(1, num_epochs + 1), train_acc, 'epochs', 'acc', range(1, num_epochs + 1), test_acc, ['train', 'test'])
    #画混淆矩阵图
    print(conf_matrix)



#==========================================创建模型==============================
class Inception_1d(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):#in_c是输入的通道 c1 c2 c3 c
        super(Inception_1d,self).__init__()
        #线路1 1x1 卷积层
        self.p1_1 =nn.Conv1d(in_c,c1,kernel_size=1)

        #线路2 1x1卷积层 跟1x3卷积
        self.p2_1 =nn.Conv1d(in_c,c2[0],kernel_size=1)
        self.p2_2 =nn.Conv1d(c2[0],c2[1],kernel_size=3,padding=1,stride=1)
        # #线路3 1x1卷积 跟1x5卷积层
        self.p3_1 =nn.Conv1d(in_c,c3[0],kernel_size=1)
        self.p3_2 =nn.Conv1d(c3[0],c3[1],kernel_size=5,padding=2,stride=1)
        # #线路4 1d最大池化 后跟1x1卷积
        self.p4_1 =nn.MaxPool1d(kernel_size=3,padding=1,stride=1)
        self.p4_2 =nn.Conv1d(in_c,c4,kernel_size=1)


    def forward(self,X):
        p1 =F.relu(self.p1_1(X))
        p2 =F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 =F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 =F.relu(self.p4_2(self.p4_1(X)))

        # return p4.shape
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出

class GlobalAvgPool1d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool1d,self).__init__()


    def forward(self,x):

        return F.avg_pool1d(x,kernel_size =(x.shape[2],))#这里！

class FlattenLayer_1d(nn.Module):
    def __init__(self):
        super(FlattenLayer_1d, self).__init__()
        print('调用FLATTEN')
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)



# 卷积层1
conv1 =nn.Sequential(nn.Conv1d(in_channels =1,out_channels =64,kernel_size=32,stride=8,padding=12),
                     nn.ReLU(),
                     nn.MaxPool1d(kernel_size=2,stride=2)
                     )

# 卷积层2
conv2 =nn.Sequential(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1),
                     nn.ReLU(),
                      nn.Conv1d(in_channels =128,out_channels=192,kernel_size=2,stride=2),
                      nn.ReLU(),
                      nn.MaxPool1d(kernel_size=2,stride =2)
                     )

# inception层
inception =nn.Sequential(Inception_1d(in_c =192,c1 =64,c2=(96,128),c3=(16,32),c4=32),
                         Inception_1d(in_c =256,c1 =128,c2=(128,192),c3=(32,96),c4=64)
                         )

# nin 层
nin =nn.Sequential(nn.Conv1d(in_channels=480,out_channels=256,kernel_size=1),
                   nn.ReLU(),
                   nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1),
                   nn.ReLU(),
                   GlobalAvgPool1d()
                  )
#flatten层
flatten= FlattenLayer_1d()


#全连接层
full_connect =nn.Sequential(
                            nn.Linear(128,64),
                            nn.ReLU(0.2),
                            # nn.Dropout(0.2),
                            nn.Linear(64,10),
                            nn.ReLU(),
                            # nn.Dropout(0.2)
                            )

#==============================================================================


#实例化一个模型 = =
net =nn.Sequential(conv1,conv2,inception,nin,flatten,full_connect)

#保存模型到文件夹
# X = torch.rand(10, 1,1024)
# print(net(X))
# with SummaryWriter('Net1')as w:
#     w.add_graph(net, (net(X),))


#测试通过
 # print(net)
# X = torch.rand(10, 1,1024)
# print(net(X).shape)



#设定batchsize
batch_size=64

train_iter,test_iter =DP.get_train_test_iter(2000,1600,1024,64)

#训练模型：
lr =0.001
num_epochs=100
optimizer =torch.optim.Adam(net.parameters(),lr,weight_decay=0)

confusion_matrix =torch.zeros(10,10).int()

train_model(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs,confusion_matrix)






