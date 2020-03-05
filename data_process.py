# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:44:19 2020

@author: zhoubo
"""
# #%%没有使用数据增强

# import pandas as pd
# import torch
# import numpy as np
# #============================制作正常数据集=====================================
# #读取1772转速下正常的轴承振动数据并制作数据集
# regular_data_1772 =pd.read_csv('D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv')

# #定义归一化函数
# max_min_scaler=lambda x: x-x.min()/(x.max()-x.min())
# #数据归一化处理
# regular_data_1772[['0']] = regular_data_1772[['0']].apply(max_min_scaler)

# #换个名字防止弄混
# regular_data_1772_normal =regular_data_1772



# regular_data_1772_normal =regular_data_1772_normal.values

# regular_data_1772_normal =regular_data_1772_normal.astype(np.float32).reshape(1,-1)


# print(regular_data_1772_normal.dtype,regular_data_1772_normal.shape)#二维数组

# slice_num=1024#1024个数据点作为一个样本
# sample_num=470
# sample_blk=[]#用来存放截取出来正常的数据
# start =0

# for i in range(sample_num):
    
#     myslice=slice(start,(i+1)*slice_num)
    
#     data =regular_data_1772_normal[:,myslice]
#     start+=slice_num
#     sample_blk.append(data)
#     # break

# test_data_regular =torch.tensor(sample_blk[0:370],dtype =torch.float32)#训练集数据为370
# train_data_regular =torch.tensor(sample_blk[370:],dtype=torch.float32)#测试集数据为100
# # 这样测试集：训练集8:2 因为后面打算K折交叉验证 6：2：2
# print(test_data_regular.shape)#制作3维张量    330个1行1024列的样本
   
# #=======================制作0.007损伤半径,内圈故障的轴承数据集===================
# inner_race_data:pd.read_csv('D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv')
 

#%%   
#定义归一化函数
import pandas as pd
import torch
import numpy as np




#归一化函数 做max-min归一化
def max_min_scaler(x):
    return x-x.min()/(x.max()-x.min())

# 预处理函数
def data_normalize(dataFrame):
    dataFrame[['0']] =dataFrame[['0']].apply(max_min_scaler)
    return dataFrame#返回归一化后的dataFrame
    

#获取分割后的数据集函数  要求传入的dataFrame必须是n行1列的
def get_train_test_data(dataFrame,sample_length =1024,sample_nums=470,train_nums=370):#需要传入的dataFrame是n行1列的
    #数据类型转换
    dataFrame =dataFrame.values#取出dataFrame的这一列值 是个numpy数组
    dataFrame =dataFrame.astype(np.float32).reshape(1,-1)#数据类型转为float32再将一列数据转换成1行（2维度）
    
    strat_index =0#初始化索引
    #创建用来存放截取样本的list   
    sample_blk=[]
    
    for i in range(sample_nums):#需要截出多少样本循环多少次
        slice_length =slice(strat_index,(i+1)*sample_length)
        
        data =dataFrame[:,slice_length]#因为上面已经把dataFrame转成一行n列的二维numpy数组 所以这里按列截取
        sample_blk.append(data)#将每次截取到的numpy数组存入列表
        strat_index+=sample_length#对起始索引累加更新
    
    # 循环截取直到指定次数
    # 创建测试集(用列表来生成tensor)
    train_data =torch.tensor(sample_blk[0:train_nums],dtype=torch.float32)#训练集数据为train_data个
    test_data = torch.tensor(sample_blk[train_nums:],dtype=torch.float32)#剩余的为测试集
    print('train_nums =:\n',train_nums)
    print('test_nums= :\n',sample_nums -train_nums)
    return train_data,test_data
        
        
    
    

#制作 =========================正常轴承数据 测试训练集=====================
    
regular_data_1772 =pd.read_csv('D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv')

#归一化
regular_data_1772 =data_normalize(regular_data_1772)

#得到测试训练集
test_data_regular,train_data_regular = get_train_test_data(regular_data_1772)







    