# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:01:54 2020

@author: zhoubo
"""
from matplotlib import pyplot as plt
import torch
from IPython import display

#------------------------画测试集训练集的loss -epoch 图----------------------------------
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

#这个可以调用
def draw_loss(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,legend=None, figsize=(3.5, 2.5)):
    plt.figure(1)
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals :
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)





def draw_accuracy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,legend=None, figsize=(3.5, 2.5)):
    plt.figure(2)
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals :
        plt.plot(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
#------------------------------------------------------------------------------


#----------------------------画淆矩阵图-----------------------------------------
def confusion_matrix(preds,labels,conf_matrix):#传入的prebs是argmax后的
    preds =preds.int()
    labels =labels.int()
    for p,t in zip(preds,labels):
        conf_matrix[p,t] += 1
    return conf_matrix

# if __name__ =='__main__':
#     preds =torch.randint(10,(10,10))
#     print(preds)
#     labels =torch.torch.linspace(0,9,steps =10).int()
#     print(labels)
#     conf_matrix =torch.zeros(10,10)
#     # print()
#     a =confusion_matrix(preds,labels,conf_matrix)
#     print(a)
#     # preds =torch.argmax(preds,1)
#     # for p,t in zip(preds,labels):

#     #     conf_matrix[p.int(),t.int()] += 1






