# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:41:30 2020

@author: zhoubo
"""
import torch

import time
import DataProcess as DP

import torch.nn as nn
import time
import torch.nn.functional as F
import myplot
import numpy as np
import myplot
from matplotlib import pyplot as plt


device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')





#将训练集再做K折交叉验证
# 下面实现了一个函数，它返回第i折交叉验证时所需要的训练和验证数据。
def get_k_fold_data(k, i, train_features, train_labels):#k:几折，i：第i个子集作为验证集 train:
    '''k:几折，i：第i个子集作为验证集 train:_features：最初训练集样本，trian_labels：训练集标签'''
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1#首先判断你要进行几折交叉验证 这个数字必须得是大于1的 才有意义 小于1 报错
    fold_size = train_features.shape[0] // k #如果进行K折交叉验证那么需要划分K个子集  这里是计算每个子集的大小 地板除
    X_train, y_train = None, None

    for j in range(k):#循环K次 j=0，1，2，3，4...K

        idx = slice(j * fold_size, (j + 1) * fold_size)# slice() 函数实现切片对象 inx =slice(1,4) a=[0,1,2,3,4,5,6,7,8,9,10] a[inx]：[1, 2, 3]
        X_part, y_part = train_features[idx, :], train_labels[idx]#截取一个fold_size的X数据和y数据

        if j == i:#判断当前循环次数是否是 我们要取出当测试集的那一次（第i次）如果是：
            X_valid, y_valid = X_part, y_part#那么把这次循环截取到的子集 作为验证集
        elif X_train is None:#如果X_train为空 （一般来说第一次循环的时候是None）就把截取到的数据 复制给X_train y_train
            X_train, y_train = X_part, y_part

        else:#如果X_train y_train 有值了或者 当前循环取出的数据不是我们要用作测试集的
            X_train = torch.cat((X_train, X_part), dim=0)#竖着拼接拼成一列
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid#返回将原始训练集划再分成 训练集-验证集的 第i个子集作为验证集时的  训练集  验证集 features 和labels






# 在K折交叉验证中我们训练K次并返回训练和验证的平均误差。
def k_fold(k, train_features, train_labels, num_epochs,lr,weight_decay, batch_size,device):



    train_l_sum, valid_l_sum = 0, 0#定义训练集loss和验证集 loss置为0

    for i in range(k):#相当于K个测试集-验证集 循环K次

        net =get_net()#必须保证每次模型都得重新创建
        print(id(net))
        data = get_k_fold_data(k, i, train_features, train_labels)# 获取K折交叉验证所用的测试集验证集数据     data此时应该是一个元组 的形式


        train_ls, valid_ls,train_acc,valid_acc = train(net, *data,num_epochs,lr,weight_decay,device,batch_size)#K折 i折验证的 训练集误差 和验证集误差

        train_l_sum += train_ls[-1]#对 把 计算第K折训练集上的误差  累加给 训练总误差
        valid_l_sum += valid_ls[-1]#第K折验证集上的 误差同理
        if i == 0:#画出第i==0次 训练集-rmse  验证集-rmse 对数图
            myplot.draw_loss(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                          range(1, num_epochs + 1), valid_ls,
                          ['train', 'valid'])

        print('fold %d, train loss %f, valid loss %f' % (i, train_ls[-1], valid_ls[-1]))#
        print('fold %d, train acc %f, valid acc %f' % (i, train_acc[-1], valid_acc[-1]))

    return train_l_sum / k, valid_l_sum / k #返回的是在K折交叉验证的平均误差

def train(net,X_train,y_train,X_,y_valid,num_epochs,lr,weight_decay,device,batch_size):
    net =net.to(device)
    print('Training on ',device)

    optimizer =torch.optim.Adam(net.parameters(),lr,weight_decay=0)


    #制作划分后的训练集dataloder
    train_dataset =torch.utils.data.TensorDataset(X_train,y_train)
    train_iter =torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    #制作验证集dataloader
    valid_dataset =torch.utils.data.TensorDataset(X_,y_valid)
    valid_iter =torch.utils.data.DataLoader(valid_dataset,batch_size,shuffle=True)

    train_ls =[]#存放每个epoch训练集loss
    valid_ls=[]#存放每个epoch验证集loss

    train_acc=[]#存放每个epoch的训 练集正确率
    valid_acc=[]#存放每个epoch 验证集正确率




    for epoch in range(num_epochs):

        train_l_sum,train_batch_count =0.0,0#train_l_sum是所有epoch中所有batch的loss和
        train_acc_sum ,train_n=0,0# train_acc_sum是 epoch结束后 总的正确个数



        valid_l_sum,valid_batch_count =0.0,0
        valid_acc_sum,valid_n =0,0

#只关注loss！
        for X,y in train_iter:
            X = X.to(device)#先将需要计算的添加到 cuda
            y = y.to(device)#同上

            y_hat = net(X)#计算模型预测值y_hat---->

            l = loss(y_hat, y.long())#计算损失（利用前面定义的交叉熵损失函数）

            optimizer.zero_grad()#优化器梯度清0
            l.backward()#误差反向传播
            optimizer.step()#梯度更新

            train_l_sum += l.cpu().item()#计算得到的误差可能再GPU上先移动到CPU转成pyton数字 #这里的train_l_sum是这个epoch内，每个batch的loss累加
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()#训练集正确总数

            train_batch_count += 1
            train_n +=y.shape[0]#累加，训练样本的个数（通过一个batch，一个batch把y有多少行进行累加的）

        train_ls.append(train_l_sum/train_batch_count)#对一个epoch里 所有batch的loss求平均放到train loss里
        train_acc.append(train_acc_sum/train_n)


        with torch.no_grad():
            for X,y in valid_iter:
                net.eval()#关闭dropout 进入评估模式

                X= X.to(device)
                y= y.to(device)

                valid_l_sum += loss(net(X),y.long()).cpu().item()
                valid_acc_sum +=  (net(X).argmax(dim =1)==y) .float().sum().cpu().item()

                valid_batch_count += 1
                valid_n += y.shape[0]


                net.train()

            valid_ls.append(valid_l_sum/valid_batch_count)
            valid_acc.append(valid_acc_sum/valid_n)

    return train_ls, valid_ls,train_acc,valid_acc







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

def get_net():
    net =nn.Sequential(conv1,conv2,inception,nin,flatten,full_connect)
    return net

loss =torch.nn.CrossEntropyLoss()#默认使用交叉熵损失函数

batch_size=128

train_features,train_labels,_,_ =DP.get_features_and_labels(2000, 1600, 1024)
k=5
num_epochs=50
lr=0.0001
weight_decay =0

K_train_loss,K_valid_loss=k_fold(k, train_features, train_labels, num_epochs,lr,weight_decay, batch_size,device)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, K_train_loss, K_valid_loss))