# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:18:35 2020

@author: zhoubo
"""



import torch

import DataProcess as DP

import TestModel2

import myutils
import myplot

device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')


train_features,train_labels,test_features,test_labels =DP.get_features_and_labels(2000, 1600, 1024)



batch_size=256
k=2
num_epochs=10
lr=0.0001
weight_decay =0

#初始化混淆矩阵
confuse_matrix =torch.zeros(10,10)

#--------------------------K折交叉验证----------------------------------------
# K_train_loss,K_valid_loss,best_net,conf_matrix=myutils.k_fold(k, train_features, train_labels, num_epochs,lr,weight_decay, batch_size,device)
# print('%d-fold validation: avg train loss %f, avg valid loss %f' % (k, K_train_loss, K_valid_loss))

#----------------------------------------------------------------------------

#-------------------在完整的数据集集上进行训练和测试----------------------------
net =TestModel2.get_net()

# train_dataset =torch.utils.data.TensorDataset(train_features,train_labels)
# train_iter =torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
# test_dataset =torch.utils.data.TensorDataset(test_features,test_labels)
# test_iter =torch.utils.data.DataLoader(test_dataset,batch_size ,shuffle =True)

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
