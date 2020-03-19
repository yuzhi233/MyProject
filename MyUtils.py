# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:59:04 2020

@author: zhoubo
"""
import torch
from torch.utils.tensorboard import SummaryWriter

#统计并打印模型总共有多少参数
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

#保存模型到文件夹
def save_model_graph(model,X):
    with SummaryWriter('Net1')as w:
        w.add_graph(model, (X,))#这里第二个输入值是元组


