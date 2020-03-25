# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:18:35 2020

@author: zhoubo
"""



import torch

import DataProcess as DP

import TestModel2

import myutils



device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')


train_features,train_labels,test_features,test_labels =DP.get_features_and_labels(2000, 1600, 1024)



batch_size=256
k=5
num_epochs=100
lr=0.0001
weight_decay =0

#初始化混淆矩阵
confuse_matrix =torch.zeros(10,10)

#--------------------------K折交叉验证----------------------------------------
K_train_loss,K_valid_loss,best_net,conf_matrix=myutils.k_fold(k, train_features, train_labels, num_epochs,lr,weight_decay, batch_size,device)
print('%d-fold validation: avg train loss %f, avg valid loss %f' % (k, K_train_loss, K_valid_loss))

#----------------------------------------------------------------------------

net =TestModel2.get_net()


