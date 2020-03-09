# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:09:17 2020

@author: zhoubo
"""

import torch
import numpy as np
import pandas as pd
#归一化函数 做max-min归一化
def max_min_scaler(x):
    return x-x.min()/(x.max()-x.min())

# 预处理函数
def data_normalize(dataFrame):
    dataFrame[['0']] =dataFrame[['0']].apply(max_min_scaler)
    return dataFrame#返回归一化后的dataFrame






def get_train_test_data(dataFrame,sample_length =1024,sample_nums=470,train_nums=370,slice_stride=None):#需要传入的dataFrame是n行1列的

    #数据类型转换
    dataFrame =dataFrame.values#取出dataFrame的这一列值 是个numpy数组
    dataFrame =dataFrame.astype(np.float32).reshape(1,-1)#数据类型转为float32再将一列数据转换成1行（2维度）

    strat_index =0#初始化索引
    sample_blk=[]#创建用来存放截取样本的list

    #如果没有输入步长 就按不重叠顺序采样
    if slice_stride is None:
        for i in range(sample_nums):#需要截出多少样本循环多少次
            slice_length =slice(strat_index,(i+1)*sample_length)

            data =dataFrame[:,slice_length]#因为上面已经把dataFrame转成一行n列的二维numpy数组 所以这里按列截取
            sample_blk.append(data)#将每次截取到的numpy数组存入列表
            strat_index+=sample_length#对起始索引累加更新

        # 循环截取直到指定次数
        # 创建测试集(用列表来生成tensor)
        train_data =torch.tensor(sample_blk[0:train_nums],dtype=torch.float32)#训练集数据为train_data个
        test_data = torch.tensor(sample_blk[train_nums:],dtype=torch.float32)#剩余的为测试集
        print('train_nums =',train_nums)
        print('test_nums= ',sample_nums -train_nums)
        return train_data,test_data

    #如果输入了步长，就按按步长顺序重叠采样
    else:
       for i in range(sample_nums):
           slice_length =slice(strat_index,strat_index+sample_length)
           data =dataFrame[:,slice_length]
           sample_blk.append(data)
           strat_index+=slice_stride


       train_data =torch.tensor(sample_blk[0:train_nums],dtype=torch.float32)#训练集数据为train_data个
       test_data = torch.tensor(sample_blk[train_nums:],dtype=torch.float32)#剩余的为测试集
       print('train_nums =',train_nums)
       print('test_nums= ',sample_nums -train_nums)
       return train_data,test_data


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetDataSetA(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, datafile, normalize =False,sample_length =1024,sample_nums=470,train_nums=370,slice_stride=None,data_label =0):
        self.dataFrame =pd.read_csv(datafile)

        self.sample_length =sample_length
        self.sample_nums =sample_nums
        self.train_nums =train_nums
        self.slice_stride=slice_stride


        self.strat_index =0#初始化索引
        self.sample_blk=[]#创建用来存放截取样本的list



        # self.data = data_root
        # self.label = data_label
        if normalize ==False:
            print('数据未进行归一化处理！')

        else:
            self.__data_normalize()#第一步归一化

            self.__split_train_test_data()#第二部对输入的数据划分测试集 和 训练集两部分



    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.train_data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)



    def __data_normalize(self):#私有方法  数据归一化
        self.dataFrame[['0']] =self.dataFrame[['0']].apply(lambda x: x-x.min()/(x.max()-x.min()))#将dataFrame ‘9’这一列归一化

    def __split_train_test_data(self):#私有方法 按要求截取归一化后的 训练集和测试集 保存到test_data 和 train_data属性中

        # #数据类型转换
        # self.dataFrame =self.dataFrame.values#取出dataFrame的这一列值 是个numpy数组
        self.dataFrame =self.dataFrame.values.astype(np.float32).reshape(1,-1)#数据类型转为float32再将一列数据转换成1行（2维度）


        #如果没有输入步长 就按不重叠顺序采样
        if self.slice_stride is None:
            for i in range(self.sample_nums):#需要截出多少样本循环多少次
                self.slice_length =slice(self.strat_index,(i+1)*self.sample_length)

                self.data =self.dataFrame[:,self.slice_length]#因为上面已经把dataFrame转成一行n列的二维numpy数组 所以这里按列截取
                self.sample_blk.append(self.data)#将每次截取到的numpy数组存入列表
                self.strat_index+=self.sample_length#对起始索引累加更新

            # 循环截取直到指定次数
            # 创建测试集(用列表来生成tensor)
            self.train_data =torch.tensor(self.sample_blk[0:self.train_nums],dtype=torch.float32)#训练集数据为train_data个
            self.test_data = torch.tensor(self.sample_blk[self.train_nums:],dtype=torch.float32)#剩余的为测试集
            print('train_nums =',self.train_nums)
            print('test_nums= ',self.sample_nums -self.train_nums)


        #如果输入了步长，就按按步长顺序重叠采样
        else:
            for i in range(self.sample_nums):
                self.slice_length =slice(self.strat_index,self.strat_index+self.sample_length)
                self.data =self.dataFrame[:,self.slice_length]
                self.sample_blk.append(self.data)
                self.strat_index+=self.slice_stride


            self.train_data =torch.tensor(self.sample_blk[0:self.train_nums],dtype=torch.float32)#训练集数据为train_data个
            self.test_data = torch.tensor(self.sample_blk[self.train_nums:],dtype=torch.float32)#剩余的为测试集
            print('train_nums =',self.train_nums)
            print('test_nums= ',self.sample_nums -self.train_nums)


    def __


a =GetDataSetA('D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv',normalize =True,slice_stride =200)