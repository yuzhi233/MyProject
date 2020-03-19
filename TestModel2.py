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







# write = SummaryWriter()








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
    myplot.semilogy(range(1, num_epochs + 1), train_yvalues, 'epochs', 'loss', range(1, num_epochs + 1), test_yvalues, ['train', 'test'])
    myplot.evl_accuracy(range(1, num_epochs + 1), train_acc, 'epochs', 'acc', range(1, num_epochs + 1), test_acc, ['train', 'test'])
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


#获取训练测试集
train_iter,test_iter=DP.get_train_test_iter(2000,1600,1024,batch_size)

def train(net, train_features, train_labels, test_features, test_labels, num_epochs,
          learning_rate, weight_decay, batch_size):

    loss =torch.nn.CrossEntropyLoss()

    train_ls, test_ls = [], []#老规矩存储训练集 和 测试集的loss

    dataset = torch.utils.data.TensorDataset(train_features, train_labels)#制作datase
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)#装载数据

    # 定义优化算法  这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay) #权值衰减开启


    for epoch in range(num_epochs):
        for X, y in train_iter:
            #计算loss
            l = loss(net(X.float()), y.float())
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            l.backward()
            #参数更新
            optimizer.step()
        train_ls.append(l)
        if test_labels is not None:#如果参数中传了 testlabels 就将testlabels的误差也计算存入列表
            test_ls.append(loss(net(test_features),test_labels))

    return train_ls, test_ls


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
def k_fold(k, train_features, train_labels, num_epochs,net,learning_rate, weight_decay, batch_size):

    train_l_sum, valid_l_sum = 0, 0#定义训练集loss和验证集 loss置为0

    for i in range(k):#相当于K个测试集-验证集 循环K次
        data = get_k_fold_data(k, i, train_features, train_labels)# 获取K折交叉验证所用的测试集验证集数据     data此时应该是一个元组 的形式


        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)#K折 i折验证的 训练集误差 和验证集误差

        train_l_sum += train_ls[-1]#对 把 计算第K折训练集上的误差  累加给 训练总误差
        valid_l_sum += valid_ls[-1]#第K折验证集上的 误差同理
        # if i == 0:#画出第i==0次 训练集-rmse  验证集-rmse 对数图
        #     d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
        #                  range(1, num_epochs + 1), valid_ls,
        #                  ['train', 'valid'])

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))#

    return train_l_sum / k, valid_l_sum / k #返回的是在K折交叉验证的平均误差



train_features,train_labels =DP.get_train_features_and_labels(2000,1600,1024)




#训练模型：
# lr =0.001
# num_epochs=10
# optimizer =torch.optim.Adam(net.parameters(),lr,weight_decay=0)

# confusion_matrix =torch.zeros(10,10).int()

# train_model(net,test_iter,train_iter,batch_size,optimizer,device,num_epochs,confusion_matrix)






