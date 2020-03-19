# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:44:19 2020
这个可以舍弃了...
@author: zhoubo
"""


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


#获取分割后的数据集函数  要求传入的dataFrame必须是n行1列的  数据集的截取方式是不重叠截取
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



# def cacul_stride(dataFrame,sample_length,sample_nums):

#     length = dataFrame.shape[0]#算出dataFrame(1列的dataFrame) 的长度
#     # (length-sample_length)//sample_nums =stride
#     # print(stride)









# =========================制作正常轴承数据 测试训练集===========================
def get_dataset_iter(batch_size):
    regular_data_1772 =pd.read_csv('D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv')

    #归一化
    regular_data_1772 =data_normalize(regular_data_1772)

    #得到测试训练集
    train_data_regular,test_data_regular = get_train_test_data(regular_data_1772)

    #制作正常测试数据 标签0
    train_data_regular_labels =torch.zeros(train_data_regular.shape[0])
    # 制作正常训练数据
    test_data_regular_labels =torch.zeros(test_data_regular.shape[0])


# =======================制作损伤半径0.007 内圈故障轴承训练测试集=================
    inner_race_data = pd.read_csv('D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv')

    # 归一化
    inner_race_data = data_normalize(inner_race_data)

    #获得测试集和训练集
    train_data_inner ,test_data_inner =get_train_test_data(inner_race_data,slice_stride =200)

    train_data_inner_labels =torch.ones(train_data_inner.shape[0])
    test_data_inner_labels =torch.ones(test_data_inner.shape[0])


#====================制作损伤半径0.007 滚动体故障轴承测试集训练集=================
    ball_data =pd.read_csv('D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X119_DE_time.csv')

    ball_data = data_normalize(ball_data)

    train_data_ball ,test_data_ball =get_train_test_data(ball_data,slice_stride =200)

    train_data_ball_labels =torch.ones(train_data_ball.shape[0])*2
    test_data_ball_labels =torch.ones(test_data_ball.shape[0])*2

#====================制作损伤半径0.007 外圈故障轴承训练测试集====================
    outer_race_data =pd.read_csv('D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X131_DE_time.csv')

    outer_race_data =data_normalize(outer_race_data)

    train_data_outer ,test_data_outer =get_train_test_data(ball_data,slice_stride =200)

    train_data_outer_labels =torch.ones(train_data_outer.shape[0])*3
    test_data_outer_labels =torch.ones(test_data_outer.shape[0])*3

#=====================================拼接=======================================


# 按行方向拼接训练集 （该测试集包含了0.007损伤半径的 内圈 外圈 滚动体 正常 4种  4分类
    train_features = torch.cat((train_data_regular,train_data_inner,train_data_ball,train_data_outer),dim =0)
    # torch.unsqueeze(index)可以为Tensor增加一个维度
    # train_features=train_features.unsqueeze(0)
# 按行方向拼接测试集
    test_features =torch.cat((test_data_regular,test_data_inner,test_data_ball,test_data_outer),dim=0)


# 制作训练集标签
    train_labels=torch.cat((train_data_regular_labels,train_data_inner_labels,train_data_ball_labels,train_data_outer_labels))
#制作测试集标签
    test_labels =torch.cat((test_data_regular_labels,test_data_inner_labels,test_data_ball_labels,test_data_outer_labels))


# 制作训练数据集
    train_dataset =torch.utils.data.TensorDataset(train_features,train_labels)
# 制作测试数据
    test_dataset =torch.utils.data.TensorDataset(test_features,test_labels)




    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle =True)
    test_iter = torch.utils.data.DataLoader(test_dataset,batch_size,shuffle =True)

    #测试
    # for X,y in data_iter:
    #     print(X.shape)
    #     print(y)
    #     break

    return train_iter,test_iter,train_features,train_labels,test_features,test_labels




if __name__ =='__main__':
    # train_iter,test_iter,_,_,_,_ =get_dataset_iter(batch_size=20)
    # for X,y in test_iter:
    #     print(X.shape)
    #     print(y)
    #     break

    regular_data_1772 =pd.read_csv('D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv')

    #归一化
    regular_data_1772 =data_normalize(regular_data_1772)
    train_data_regular,test_data_regular = get_train_test_data(regular_data_1772)


