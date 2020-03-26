# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:03:22 2020

@author: zhoubo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#这里放模型
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
                     nn.BatchNorm1d(64),
                     nn.ReLU(),
                     nn.MaxPool1d(kernel_size=2,stride=2)
                     )

# 卷积层2
conv2 =nn.Sequential(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1),
                     nn.BatchNorm1d(128),
                     nn.ReLU(),

                     nn.Conv1d(in_channels =128,out_channels=192,kernel_size=2,stride=2),
                     nn.BatchNorm1d(192),
                     nn.ReLU(),
                     nn.MaxPool1d(kernel_size=2,stride =2)
                     )

# inception层
inception =nn.Sequential(Inception_1d(in_c =192,c1 =64,c2=(96,128),c3=(16,32),c4=32),
                         Inception_1d(in_c =256,c1 =128,c2=(128,192),c3=(32,96),c4=64)
                         )

# nin 层
nin =nn.Sequential(nn.Conv1d(in_channels=480,out_channels=256,kernel_size=1),
                   nn.BatchNorm1d(256),
                   nn.ReLU(),
                   nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1),
                   nn.BatchNorm1d(128),
                   nn.ReLU(),
                   GlobalAvgPool1d()
                  )
#flatten层
flatten= FlattenLayer_1d()

#全连接层
full_connect =nn.Sequential(
                            nn.Linear(128,64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(64,10),
                            nn.BatchNorm1d(10),
                            nn.ReLU()

                            )

def get_net():
    net =nn.Sequential(conv1,conv2,inception,nin,flatten,full_connect)
    return net



# if __name__ =='__main__':
#     import torch
#     net =get_net()
#     a= torch.randn(1,1,1024)
#     print(net(a))