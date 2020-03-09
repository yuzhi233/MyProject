# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:39:33 2020

@author: zhoubo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:30:56 2020

@author: zhoubo
"""

#%% 实现LeNet模型
import time 
import torch

from torch import nn,optim 
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)#将模型添加到设备上 可能是GPU可能是CPU
    print("training on ", device)#显示是在哪训练的
    loss = torch.nn.CrossEntropyLoss()#使用交叉熵损失函数
    for epoch in range(num_epochs):#几个迭代周期 开始迭代
        #定义 训练集损失和,训练集准确总数，总样本数n,几个batch,开始时间
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:#一个批次中(比如一个batch_size是256)
            X = X.to(device)#先将需要计算的添加到 设备
            y = y.to(device)#同上
            
            y_hat = net(X)#计算模型预测值y_hat---->
 
            l = loss(y_hat, y)#计算损失（利用前面定义的交叉熵损失函数）
            
            optimizer.zero_grad()#优化器梯度清0
            l.backward()#误差反向传播
            optimizer.step()#梯度更新
            train_l_sum += l.cpu().item()#计算得到的误差可能再GPU上先移动到CPU转成pyton数字
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()#训练集正确总数
            n += y.shape[0]
            batch_count += 1
            
        #在训练集迭代完用测试集进行模型评估
        test_acc = evaluate_accuracy_2(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
        
        
        
def evaluate_accuracy_2(data_iter, net,device=None):#用于评估测试集准确率 目的是要实现评估的时候要自动关闭dropout 
    if device is None and isinstance(net,torch.nn.Module):#如果设备是none且net是由nn.module生成的实例。则：
          # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():#不追踪操作
        for X, y in data_iter:#从data_iter取出一个batch的X,y
                  
          #先判断你这个net是怎么产生的是你自己手写的还是利用pytorch快速生成的
            if isinstance(net, torch.nn.Module):#判断net是不是用torch.nn.Module创建的实例(判断net是不是利用module模块搭建的)
            
                net.eval() # #如果是上面方法创建的 那么开启评估模式 dropout层全部关闭(因为我们要是通过module模块创建一个模型有可能添加了dropout层)
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()#判断正确的个数
                net.train() # 改回训练模式
            # else: # 如果是我们自定义的模型    else下面的这段主要是用于3.13节我们自定义带dropout的模型，计算准确率的以后不会用到 不考虑GPU
            #     if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
            #     # 将is_training设置成False
            #         acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() #先将is_training设置成 False 关闭dropout
            #     else:#(形参)没有is_training这个参数
            #         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            
            n += y.shape[0]#其实就是算了以下一个批次有多少样本 每次循环累加一下参加计算的样本数
            break
    return acc_sum / n#在所有批次循环后  计算准确率 拿 准确的个数/总个数
            

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv =nn.Sequential(
            #步长默认为1,padding默认是0 图像是32*328------>卷积后：  
            nn.Conv2d(1,6,5),## in_channels, out_channels, kernel_size   第一个卷积层  输入通道1(后面用Fashion_MNIST只有一通道)，输出通道 第一层设计了6个输出通道，卷积核size5*5  计算后得到卷积后图像为28*28  
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),#池化层 最大池化  kernel size =2*2 池化操作默认 stride=keren_size=2
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)#这步池化后得到的图像尺寸为4*4
        #至此 卷积部分结束
            )
        self.fc =nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
            )
    def forward(self,img):
        feature =self.conv(img)
        output =self.fc(feature.view(img.shape[0],-1))#数据扁平化 一个图像整成一行数据
        return output
        
        
net=LeNet()
print(net)    

#  获取数据和训练模型
# 下面我们来实验LeNet模型。实验中，我们仍然使用Fashion-MNIST作为训练数据集。

batch_size = 256#设置一个batch256个样本

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)#加载数据到迭代器
lr =0.001
num_epochs =5
optimizer =torch.optim.Adam(net.parameters(),lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)


        





    