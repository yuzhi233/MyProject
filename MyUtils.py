# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:59:04 2020

@author: zhoubo
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn as nn
import myplot
import TestModel2



#统计并打印模型总共有多少参数
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

#保存模型到文件夹
def save_model_graph(model,X):
    with SummaryWriter('Net1')as w:
        w.add_graph(model, (X,))#这里第二个输入值是元组


def weight_init(m):
    classname = m.__class__.__name__ # 得到网络层的名字，如ConvTranspose2d
    if classname.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def _initialize_weights(self):#这个是VGG的参数初始化策略
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)






#--------------------------------------------------------------------------K折交叉验证相关--------------------------------------

'''K折交叉验证时候要注意的地方:

1.loss函数要在程序里提前定义
2.程序中需要有个名为使用一个get_net()函数创建模型
3.混淆矩阵要提前初始化
4.k_fold中模型不同net =TestModel2.get_net()需要重写

默认是交叉熵损失函数  train
结果K_fold会返回回的是在K折交叉验证的平均loss，返回的net_record是K折，每一折 loss最低时候的net的states-dict,
best_conf_matix是每一折中loss最低时候的混淆矩阵
'''


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


#注意运行该 需要在程序里定义一个get_net()函数来获取模型！！！！！
def train(net,X_train,y_train,X_valid,y_valid,num_epochs,lr,weight_decay,device,batch_size):
    net =net.to(device)
    print('Training on ',device)

    loss =nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(net.parameters(),lr,weight_decay=0)


    #制作划分后的训练集dataloder
    train_dataset =torch.utils.data.TensorDataset(X_train,y_train)
    train_iter =torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    #制作验证集dataloader
    valid_dataset =torch.utils.data.TensorDataset(X_valid,y_valid)
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


        # if epoch ==0:
        #     net_record=net.state_dict()
        #     min_loss =train_ls[-1]
        # elif train_ls[-1]<min_loss:
        #     min_loss =train_ls[-1]
        #     net_record=net.state_dict()
        # else:
        #     pass

        with torch.no_grad():
            confuse_matrix =torch.zeros(10,10).int()#每一个epoch 给混淆矩阵置0
            for X,y in valid_iter:
                net.eval()#关闭dropout 进入评估模式

                X= X.to(device)
                y= y.to(device)

                valid_l_sum += loss(net(X),y.long()).cpu().item()
                valid_acc_sum +=  (net(X).argmax(dim =1)==y) .float().sum().cpu().item()

                valid_batch_count += 1
                valid_n += y.shape[0]

                confuse_matrix =myplot.confusion_matrix(net(X).argmax(dim =1),y,confuse_matrix)
                net.train()

            valid_ls.append(valid_l_sum/valid_batch_count)
            valid_acc.append(valid_acc_sum/valid_n)

            #针对
            if epoch ==0:#如果是epoch0 初始化一下 net_record记录loss最低的模型的参数 min_loss：最低loss，best_conf_matrix loss最低的混淆矩阵
                net_record=net.state_dict()
                min_loss =valid_ls[-1]
                best_conf_matrix =confuse_matrix

            elif valid_ls[-1]<min_loss:#epoch>=1时 如果这次算的loss比上次小
                min_loss =valid_ls[-1]#最低loss 换成这个epoch的
                net_record=net.state_dict()#记录这次epoch的 模型参数
                best_conf_matrix =confuse_matrix#记录这次的混淆矩阵
            else:#要是既不是epoch=0又不满足 这次loss比上次低
                pass#那就啥也不干


    return train_ls, valid_ls,train_acc,valid_acc,net_record,best_conf_matrix


# 在K折交叉验证中我们训练K次并返回训练和验证的平均误差。
def k_fold(k, train_features, train_labels, num_epochs,lr,weight_decay, batch_size,device):

    time_start =time.time()

    train_l_sum, valid_l_sum = 0, 0#定义训练集loss和验证集 loss置为0
    best_net=[]#存放每一折的best model参数
    K_best_conf_matrix=[]
    for i in range(k):#相当于K个测试集-验证集 循环K次

        net =TestModel2.get_net()#必须保证每次模型都得重新创建
        net.apply(_initialize_weights)

        print(id(net))
        data = get_k_fold_data(k, i, train_features, train_labels)# 获取K折交叉验证所用的测试集验证集数据     data此时应该是一个元组 的形式


        train_ls, valid_ls,train_acc,valid_acc,net_record,best_conf_matrix = train(net, *data,num_epochs,lr,weight_decay,device,batch_size)#K折 i折验证的 训练集误差 和验证集误差

        train_l_sum += train_ls[-1]#对 把 计算第K折训练集上的最终误差  累加给 训练总误差
        valid_l_sum += valid_ls[-1]#第K折验证集上的 误差同理

        myplot.draw_loss(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                      range(1, num_epochs + 1), valid_ls,
                      ['train', 'valid'])
        myplot.draw_accuracy(range(1, num_epochs + 1), train_acc, 'epochs', 'acc',
                      range(1, num_epochs + 1), valid_acc,
                      ['train', 'valid'])

        best_net.append(net_record)


        K_best_conf_matrix.append(best_conf_matrix)

        print('fold %d, train loss %f, valid loss %f' % (i, train_ls[-1], valid_ls[-1]))
        print('fold %d, train acc %f, valid acc %f' % (i, train_acc[-1], valid_acc[-1]))
    print('time:',time.time()-time_start)
    return train_l_sum / k, valid_l_sum / k,best_net,K_best_conf_matrix #返回的是在K折交叉验证的平均误差，#返回的net_record是K折 每一折 loss最低时候的net的states-dict,best_conf_matix是每一折中loss最低时候的混淆矩阵



#------------------------------------------------------------------------------------------------------------------------------------





