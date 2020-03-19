# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:09:18 2020

@author: zhoubo
"""
import torch
import numpy as np
import pandas as pd



#归一化函数
def max_min_scaler(x):
    return x-x.min()/(x.max()-x.min())

#数据归一化函数
def data_normalize(df):
    '''DataFrame_data必须是n行一列 且这一列标签为‘0’的DataFrame类型的data  '''
    s = (df['0'] - df['0'].min())/(df['0'].max() - df['0'].min())
    df2 = df.drop(['0'],axis=1)
    df2.insert(0,'0',s)


    return df2#返回归一化后的DataFrame对象

#计算采样步长
def calculate_stride(data_length,sample_length,sample_nums):
    stride =(data_length-sample_length)//(sample_nums- 1)
    return stride

#随机采样  以后用这种方法试试
def Randomsampling():
    pass


# 1制作标签
def make_train_test_labels(train_data,test_data,labels=0):
    ''' 制作训练集和测试集的的标签
        要求输入为训练集和测试集划分好后的数据
        返回值为 制作做好的测试集 训练集标签'''
    train_labels =np.ones(train_data.shape[0])*labels
    test_labels =np.ones(test_data.shape[0])*labels
    return train_labels,test_labels


# 2拼接样本数据集 拼接3维数据 按上下方向拼接
def compose_sample_subset(*subset):#会将dim=1的维度压缩
    '''输入为若干个划分好的各种不同故障的数据集样本然后上下拼接他们'''
    for i in range(len(subset)):
        if i == 0:
            dataset = subset[0]
        else:
            dataset =np.concatenate((dataset,subset[i]),axis =0)
    return dataset

#拼接标签数据集 左右拼接
def compose_labels_subset(*subset):#会将dim=1的维度压缩
    '''输入为若干个划分好的各种不同故障的标签然后左右拼接他们'''
    for i in range(len(subset)):
        if i == 0:
            dataset = subset[0]
        else:
            dataset =np.concatenate((dataset,subset[i]))
    return dataset




#拼接成一个大的数据集
def combine_dataset(features,labels):
     #调整形状 以便于拼接
    features =features.reshape(features.shape[0],-1)
    labels =labels.reshape(labels.shape[0],1)
    complete_dataset =np.concatenate((features,labels),axis=1)

    return complete_dataset


# # 3打乱数据集  不能用，，，这个只能打乱列表形式的特征和标签💀
# def shuffle_data(train_data,train_labels,test_data,test_labels):
#     ''''打乱数据集,参数：训练集数据，训练集标签，测试集数据，测试集标签
#         返回打乱后的训练集数据，训练集标签，测试集数据，测试集标签'''

#     #打乱训练集的 样本 和标签
#     for i in range(train_data.shape[0]):
#         templist1 =list(zip(train_data,train_labels))
#         random.shuffle(templist1)
#         train_data,train_labels=zip(*templist1)
#         train_data =np.array(train_data)
#         train_labels=np.array(train_labels)

#     #打乱测试集的样本和对应的标签

#     templist2 =list(zip(test_data,test_labels))
#     random.shuffle(templist2)
#     test_data,test_labels=zip(*templist1)
#     test_data =np.array(test_data)
#     test_labels=np.array(test_labels)



#     return train_data,train_labels,test_data,test_labels





def split_train_test_data(file_path,sample_nums =2000,train_data_nums=1600,sample_length=1024):
    '''file_path:西储石油大学下载的每一种故障类型数据的根目录
       sample_nums: 样本个数(总共计划划分多少个样本数据出来，默认1000,训练800,测试200)
       sample_length:样本长度(单个样本的数据点数 默认1024)
       train_data_nums:训练集划分多少个(默认划分800,测试集=sample_nums-train_data_nums)
    '''
    #读取源数据
    file_data = pd.read_csv(file_path)

    #对源数据(此时读取到的filedata是DataFrame)进行归一化 此时filedata还是n行一列的DataFrame
    file_data = data_normalize(file_data)

    #数据类型转换
    file_data = file_data.values #将filedata的这一列取出来---->变成了numpy数组 的array类型
    file_data = file_data.astype(np.float32).reshape(1,-1)#设定数据类型为float32，转换成1行n列的numpy数组(2维的)

    #先分割再划分 ----------💀这里不太严谨 但是初期先试试这种划分方式效果---------------------
    #按照重叠取样来截取
        #计算一下这个filedata有多长
    file_data_length =file_data.shape[1]

    #计算一下截取需要的步长
    slice_stride =calculate_stride(file_data_length,sample_length,sample_nums)

    assert slice_stride > 0#只有步长大于0的时候才能正常运行


    strat_index =0#初始化索引
    sample_blk=[]#创建用来存放截取样本的list



    for i in range(sample_nums):
        slice_length =slice(strat_index,strat_index+sample_length)
        split_data =file_data[:,slice_length]
        sample_blk.append(split_data)
        strat_index+=slice_stride


    train_data_list =sample_blk[0:train_data_nums]#训练集数据为train_data个
    test_data_list =sample_blk[train_data_nums:]#剩余的为测试集
    train_data =np.array(train_data_list)
    test_data = np.array(test_data_list)
    print('train_nums =',train_data_nums)
    print('test_nums= ',sample_nums -train_data_nums)
    # print(train_data.shape)
    return train_data,test_data#返回的是三维numpy数组



#获取训练集和测试集的iter
def  get_train_test_iter(sample_nums ,train_data_nums,sample_length,batch_size):

    '''输入参数为train_data_nums,sample_length,batch_size返回训练测试集迭代器'''
    #--------------------------------------------------------------------------
    #制作正常轴承的 测试训练集
    file_path1 ='D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv'
    #划分数据集 -----这一步结束后是numpy数组
    normal_train_data,normal_test_data =split_train_test_data(file_path1,sample_nums ,train_data_nums,sample_length)
    #制作数据集标签
    normal_train_labels ,normal_test_labels =make_train_test_labels(normal_train_data,normal_test_data,0)#此时标签和数据都还是numpy数组
    #--------------------------------------------------------------------------
    #制作0.007 inch损伤 内圈故障轴承 测试训练集 ----轻度损伤
    file_path2 = 'D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv'
    inner7_train_data,inner7_test_data =split_train_test_data(file_path2,sample_nums,train_data_nums,sample_length)
    inner7_train_labels ,inner7_test_labels =make_train_test_labels(inner7_train_data,inner7_test_data,1)
    #制作0.007 inch损伤 滚动体故障轴承 测试训练集 -----轻度损伤
    file_path3 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X119_DE_time.csv'
    ball7_train_data,ball7_test_data =split_train_test_data(file_path3,sample_nums ,train_data_nums,sample_length)
    ball7_train_labels ,ball7_test_labels =make_train_test_labels(ball7_train_data,ball7_test_data,2)
    #制作0.007 inch损伤 外圈故障轴承 测试训练集 -----轻度损伤
    file_path4 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X131_DE_time.csv'
    outer7_train_data,outer7_test_data =split_train_test_data(file_path4,sample_nums,train_data_nums,sample_length)
    outer7_train_labels ,outer7_test_labels =make_train_test_labels(outer7_train_data,outer7_test_data,3)
    #--------------------------------------------------------------------------
    #制作0.014 inch 损伤 内圈故障 测试训练集 -----中度损伤
    file_path5 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X170_DE_time.csv'
    inner14_train_data,inner14_test_data =split_train_test_data(file_path5,sample_nums,train_data_nums,sample_length)
    inner14_train_labels ,inner14_test_labels =make_train_test_labels(inner14_train_data,inner14_test_data,4)
    #制作0.014 inch 损伤 滚动体故障 测试训练集 -----中度损伤
    file_path6 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X185_DE_time.csv'
    ball14_train_data,ball14_test_data =split_train_test_data(file_path6,sample_nums,train_data_nums,sample_length)
    ball14_train_labels ,ball14_test_labels =make_train_test_labels(ball14_train_data,ball14_test_data,5)
    #制作0.014 inch损伤 外圈故障轴承 测试训练集 -----中度损伤
    file_path7 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X198_DE_time.csv'
    outer14_train_data,outer14_test_data =split_train_test_data(file_path7,sample_nums ,train_data_nums,sample_length)
    outer14_train_labels ,outer14_test_labels =make_train_test_labels(outer14_train_data,outer14_test_data,6)
    #--------------------------------------------------------------------------
    #制作0.021 inch 损伤 内圈故障 测试训练集 -----重度损伤
    file_path8 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X210_DE_time.csv'
    inner21_train_data,inner21_test_data =split_train_test_data(file_path8,sample_nums ,train_data_nums,sample_length)
    inner21_train_labels ,inner21_test_labels =make_train_test_labels(inner21_train_data,inner21_test_data,7)
    #制作0.021 inch 损伤 滚动体故障 测试训练集 -----重度损伤
    file_path9 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X223_DE_time.csv'
    ball21_train_data,ball21_test_data =split_train_test_data(file_path9,sample_nums ,train_data_nums,sample_length)
    ball21_train_labels ,ball21_test_labels =make_train_test_labels(ball21_train_data,ball21_test_data,8)
    #制作0.021 inch损伤 外圈故障轴承 测试训练集 -----重度损伤
    file_path10 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X235_DE_time.csv'
    outer21_train_data,outer21_test_data =split_train_test_data(file_path10,sample_nums,train_data_nums,sample_length)
    outer21_train_labels ,outer21_test_labels =make_train_test_labels(outer21_train_data,outer21_test_data,9)
    #=======================================拼接成一个大的数据集==============================

    #将各种故障划分出的训练样本 上下拼接成一个大的训练样本
    train_features =compose_sample_subset(normal_train_data,
                                          inner7_train_data,ball7_train_data,outer7_train_data,#0.007
                                          inner14_train_data,ball14_train_data,outer14_train_data,#0.014
                                          inner21_train_data,ball21_train_data,outer21_train_data)#0.021
    #拼接样本 构成样本集
    train_labels =compose_labels_subset(normal_train_labels,
                                        inner7_train_labels,ball7_train_labels,outer7_train_labels,
                                        inner14_train_labels,ball14_train_labels,outer14_train_labels,
                                        inner21_train_labels,ball21_train_labels,outer21_train_labels,)


    test_features =compose_sample_subset(normal_test_data,
                                         inner7_test_data,ball7_test_data,outer7_test_data,
                                         inner14_test_data,ball14_test_data,outer14_test_data,
                                         inner21_test_data,ball21_test_data,outer21_test_data)

    test_labels =compose_labels_subset(normal_test_labels,
                                       inner7_test_labels,ball7_test_labels,outer7_test_labels,
                                       inner14_test_labels,ball14_test_labels,outer14_test_labels,
                                       inner21_test_labels,ball21_test_labels,outer21_test_labels)
    #拼接成一个完整的没有打乱的大数据集
    train_dataset =combine_dataset(train_features,train_labels)
    test_dataset =combine_dataset(test_features,test_labels)
    #======================================提前打乱数据集======================================
    #打乱训练集
    np.random.shuffle(train_dataset)
    train_dataset =np.array(train_dataset,dtype =np.float32)

    #打乱测试集
    np.random.shuffle(test_dataset)
    test_dataset =np.array(test_dataset,dtype =np.float32)
    #=================================升维度，转成tensor==================================
    #训练集：
    #先把标签特征打乱后没分开的训练集转成tensor
    tensor_train_dataset =torch.from_numpy(train_dataset)
    #样本升维，使其满足一维卷积神经网络输入要求
    tensor_train_features = tensor_train_dataset[:,:1024]
    tensor_train_features = tensor_train_features.view(tensor_train_features.shape[0],1,tensor_train_features.shape[1])

    #取出打乱后的labels
    tensor_train_labels =tensor_train_dataset[:,-1]


    #测试集：
    tensor_test_dataset =torch.from_numpy(test_dataset)
    #样本升维，使其满足一维卷积神经网络输入要求
    tensor_test_features = tensor_test_dataset[:,:1024]
    tensor_test_features = tensor_test_features.view(tensor_test_features.shape[0],1,tensor_test_features.shape[1])

    #取出打乱后的labels
    tensor_test_labels =tensor_test_dataset[:,-1]



    #===================================制作 pytorch数据集=============================
    train_dataset = torch.utils.data.TensorDataset(tensor_train_features,tensor_train_labels)
    test_dataset = torch.utils.data.TensorDataset(tensor_test_features,tensor_test_labels)

    #===================================制作train_iter和test_iter=========================
    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size ,shuffle =True)
    test_iter = torch.utils.data.DataLoader(test_dataset,batch_size ,shuffle =True)

    return train_iter,test_iter


#获取训练集 的样本 和 标签
def get_train_features_and_labels(sample_nums ,train_data_nums,sample_length):

    #--------------------------------------------------------------------------
    #制作正常轴承的 测试训练集
    file_path1 ='D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv'
    #划分数据集 -----这一步结束后是numpy数组
    normal_train_data,normal_test_data =split_train_test_data(file_path1,sample_nums ,train_data_nums,sample_length)
    #制作数据集标签
    normal_train_labels ,normal_test_labels =make_train_test_labels(normal_train_data,normal_test_data,0)#此时标签和数据都还是numpy数组
    #--------------------------------------------------------------------------
    #制作0.007 inch损伤 内圈故障轴承 测试训练集 ----轻度损伤
    file_path2 = 'D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv'
    inner7_train_data,inner7_test_data =split_train_test_data(file_path2,sample_nums,train_data_nums,sample_length)
    inner7_train_labels ,inner7_test_labels =make_train_test_labels(inner7_train_data,inner7_test_data,1)
    #制作0.007 inch损伤 滚动体故障轴承 测试训练集 -----轻度损伤
    file_path3 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X119_DE_time.csv'
    ball7_train_data,ball7_test_data =split_train_test_data(file_path3,sample_nums ,train_data_nums,sample_length)
    ball7_train_labels ,ball7_test_labels =make_train_test_labels(ball7_train_data,ball7_test_data,2)
    #制作0.007 inch损伤 外圈故障轴承 测试训练集 -----轻度损伤
    file_path4 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X131_DE_time.csv'
    outer7_train_data,outer7_test_data =split_train_test_data(file_path4,sample_nums,train_data_nums,sample_length)
    outer7_train_labels ,outer7_test_labels =make_train_test_labels(outer7_train_data,outer7_test_data,3)
    #--------------------------------------------------------------------------
    #制作0.014 inch 损伤 内圈故障 测试训练集 -----中度损伤
    file_path5 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X170_DE_time.csv'
    inner14_train_data,inner14_test_data =split_train_test_data(file_path5,sample_nums,train_data_nums,sample_length)
    inner14_train_labels ,inner14_test_labels =make_train_test_labels(inner14_train_data,inner14_test_data,4)
    #制作0.014 inch 损伤 滚动体故障 测试训练集 -----中度损伤
    file_path6 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X185_DE_time.csv'
    ball14_train_data,ball14_test_data =split_train_test_data(file_path6,sample_nums,train_data_nums,sample_length)
    ball14_train_labels ,ball14_test_labels =make_train_test_labels(ball14_train_data,ball14_test_data,5)
    #制作0.014 inch损伤 外圈故障轴承 测试训练集 -----中度损伤
    file_path7 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X198_DE_time.csv'
    outer14_train_data,outer14_test_data =split_train_test_data(file_path7,sample_nums ,train_data_nums,sample_length)
    outer14_train_labels ,outer14_test_labels =make_train_test_labels(outer14_train_data,outer14_test_data,6)
    #--------------------------------------------------------------------------
    #制作0.021 inch 损伤 内圈故障 测试训练集 -----重度损伤
    file_path8 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X210_DE_time.csv'
    inner21_train_data,inner21_test_data =split_train_test_data(file_path8,sample_nums ,train_data_nums,sample_length)
    inner21_train_labels ,inner21_test_labels =make_train_test_labels(inner21_train_data,inner21_test_data,7)
    #制作0.021 inch 损伤 滚动体故障 测试训练集 -----重度损伤
    file_path9 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X223_DE_time.csv'
    ball21_train_data,ball21_test_data =split_train_test_data(file_path9,sample_nums ,train_data_nums,sample_length)
    ball21_train_labels ,ball21_test_labels =make_train_test_labels(ball21_train_data,ball21_test_data,8)
    #制作0.021 inch损伤 外圈故障轴承 测试训练集 -----重度损伤
    file_path10 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X235_DE_time.csv'
    outer21_train_data,outer21_test_data =split_train_test_data(file_path10,sample_nums,train_data_nums,sample_length)
    outer21_train_labels ,outer21_test_labels =make_train_test_labels(outer21_train_data,outer21_test_data,9)
    #=======================================拼接成一个大的数据集==============================

    #将各种故障划分出的训练样本 上下拼接成一个大的训练样本
    train_features =compose_sample_subset(normal_train_data,
                                          inner7_train_data,ball7_train_data,outer7_train_data,#0.007
                                          inner14_train_data,ball14_train_data,outer14_train_data,#0.014
                                          inner21_train_data,ball21_train_data,outer21_train_data)#0.021
    #拼接样本 构成样本集
    train_labels =compose_labels_subset(normal_train_labels,
                                        inner7_train_labels,ball7_train_labels,outer7_train_labels,
                                        inner14_train_labels,ball14_train_labels,outer14_train_labels,
                                        inner21_train_labels,ball21_train_labels,outer21_train_labels,)
    #拼接成一个完整的没有打乱的大数据集
    train_dataset =combine_dataset(train_features,train_labels)
    #======================================提前打乱数据集======================================
    #打乱训练集
    np.random.shuffle(train_dataset)
    train_dataset =np.array(train_dataset,dtype =np.float32)


    #=================================升维度，转成tensor==================================
    #训练集：
    #先把标签特征打乱后没分开的训练集转成tensor
    tensor_train_dataset =torch.from_numpy(train_dataset)
    #样本升维，使其满足一维卷积神经网络输入要求
    tensor_train_features = tensor_train_dataset[:,:1024]
    tensor_train_features = tensor_train_features.view(tensor_train_features.shape[0],1,tensor_train_features.shape[1])

    #取出打乱后的labels
    tensor_train_labels =tensor_train_dataset[:,-1]


    return tensor_train_features,tensor_train_labels




if __name__ =='__main__':
    train_features,train_labels =get_train_features_and_labels(2000,1600,1024)




# if __name__ =='__main__':
    # # train_iter,test_iter =get_train_test_iter(64)



    # '''输入参数为batch_size，返回训练测试集迭代器'''
    # sample_nums ,train_data_nums,sample_length=50,30,5
    # #制作正常轴承的 测试训练集
    # file_path1 ='D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv'
    # # file_path1 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\1.csv'
    # #划分数据集 -----这一步结束后是numpy数组
    # normal_train_data,normal_test_data =split_train_test_data(file_path1,sample_nums,train_data_nums,sample_length)
    # #制作数据集标签
    # normal_train_labels ,normal_test_labels =make_train_test_labels(normal_train_data,normal_test_data,0)#此时标签和数据都还是numpy数组



    # #--------------------------------------------------------------------------
    # #制作0.007 inch损伤 内圈故障轴承 测试训练集 ----轻度损伤
    # file_path2 = 'D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv'
    # inner7_train_data,inner7_test_data =split_train_test_data(file_path2,sample_nums ,train_data_nums,sample_length)
    # inner7_train_labels ,inner7_test_labels =make_train_test_labels(inner7_train_data,inner7_test_data,1)


    # #制作0.007 inch损伤 滚动体故障轴承 测试训练集 -----轻度损伤
    # file_path3 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X119_DE_time.csv'
    # ball7_train_data,ball7_test_data =split_train_test_data(file_path3,sample_nums,train_data_nums,sample_length)
    # ball7_train_labels ,ball7_test_labels =make_train_test_labels(ball7_train_data,ball7_test_data,2)


    # #制作0.007 inch损伤 外圈故障轴承 测试训练集 -----轻度损伤
    # file_path4 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X131_DE_time.csv'
    # outer7_train_data,outer7_test_data =split_train_test_data(file_path4,sample_nums,train_data_nums,sample_length)
    # outer7_train_labels ,outer7_test_labels =make_train_test_labels(outer7_train_data,outer7_test_data,3)
    # #--------------------------------------------------------------------------
    # #制作0.014 inch 损伤 内圈故障 测试训练集 -----中度损伤
    # file_path5 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X170_DE_time.csv'
    # inner14_train_data,inner14_test_data =split_train_test_data(file_path5,sample_nums,train_data_nums,sample_length)
    # inner14_train_labels ,inner14_test_labels =make_train_test_labels(inner14_train_data,inner14_test_data,4)

    # #制作0.014 inch 损伤 滚动体故障 测试训练集 -----中度损伤
    # file_path6 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X185_DE_time.csv'
    # ball14_train_data,ball14_test_data =split_train_test_data(file_path6,sample_nums,train_data_nums,sample_length)
    # ball14_train_labels ,ball14_test_labels =make_train_test_labels(ball14_train_data,ball14_test_data,5)

    # #制作0.014 inch损伤 外圈故障轴承 测试训练集 -----中度损伤
    # file_path7 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X198_DE_time.csv'
    # outer14_train_data,outer14_test_data =split_train_test_data(file_path7,sample_nums ,train_data_nums,sample_length)
    # outer14_train_labels ,outer14_test_labels =make_train_test_labels(outer14_train_data,outer14_test_data,6)
    # #--------------------------------------------------------------------------
    # #制作0.021 inch 损伤 内圈故障 测试训练集 -----重度损伤
    # file_path8 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X210_DE_time.csv'
    # inner21_train_data,inner21_test_data =split_train_test_data(file_path8,sample_nums ,train_data_nums,sample_length)
    # inner21_train_labels ,inner21_test_labels =make_train_test_labels(inner21_train_data,inner21_test_data,7)

    # #制作0.021 inch 损伤 滚动体故障 测试训练集 -----重度损伤
    # file_path9 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X223_DE_time.csv'
    # ball21_train_data,ball21_test_data =split_train_test_data(file_path9,sample_nums ,train_data_nums,sample_length)
    # ball21_train_labels ,ball21_test_labels =make_train_test_labels(ball21_train_data,ball21_test_data,8)

    # #制作0.021 inch损伤 外圈故障轴承 测试训练集 -----重度损伤
    # file_path10 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X235_DE_time.csv'
    # outer21_train_data,outer21_test_data =split_train_test_data(file_path7,sample_nums,train_data_nums,sample_length)
    # outer21_train_labels ,outer21_test_labels =make_train_test_labels(outer21_train_data,outer21_test_data,9)










    # #=======================================拼接成一个大的数据集==============================

    # #将各种故障划分出的训练样本 上下拼接成一个大的训练样本
    # train_features =compose_sample_subset(normal_train_data,
    #                                       inner7_train_data,ball7_train_data,outer7_train_data,#0.007
    #                                       inner14_train_data,ball14_train_data,outer14_train_data,#0.014
    #                                       inner21_train_data,ball21_train_data,outer21_train_data)#0.021
    # #拼接样本 构成样本集
    # train_labels =compose_labels_subset(normal_train_labels,
    #                                     inner7_train_labels,ball7_train_labels,outer7_train_labels,
    #                                     inner14_train_labels,ball14_train_labels,outer14_train_labels,
    #                                     inner21_train_labels,ball21_train_labels,outer21_train_labels,)


    # test_features =compose_sample_subset(normal_test_data,
    #                                       inner7_test_data,ball7_test_data,outer7_test_data,
    #                                       inner14_test_data,ball14_test_data,outer14_test_data,
    #                                       inner21_test_data,ball21_test_data,outer21_test_data)

    # test_labels =compose_labels_subset(normal_test_labels,
    #                                     inner7_test_labels,ball7_test_labels,outer7_test_labels,
    #                                     inner14_test_labels,ball14_test_labels,outer14_test_labels,
    #                                     inner21_test_labels,ball21_test_labels,outer21_test_labels)



    # #拼接成一个完整的没有打乱的大数据集
    # train_dataset =combine_dataset(train_features,train_labels)

    # test_dataset =combine_dataset(test_features,test_labels)




    # #======================================提前打乱数据集======================================

    # #打乱训练集
    # np.random.shuffle(train_dataset)
    # train_dataset =np.array(train_dataset,dtype =np.float32)

    # #打乱测试集
    # np.random.shuffle(test_dataset)
    # test_dataset =np.array(test_dataset,dtype =np.float32)

    # #=================================升维度，转成tensor==================================
    # #训练集：
    # #先把标签特征打乱后没分开的训练集转成tensor
    # tensor_train_dataset =torch.from_numpy(train_dataset)
    # #样本升维，使其满足一维卷积神经网络输入要求
    # tensor_train_features = tensor_train_dataset[:,:1024]
    # tensor_train_features = tensor_train_features.view(tensor_train_features.shape[0],1,tensor_train_features.shape[1])

    # #取出打乱后的labels
    # tensor_train_labels =tensor_train_dataset[:,-1]


    # #测试集：
    # tensor_test_dataset =torch.from_numpy(test_dataset)
    # #样本升维，使其满足一维卷积神经网络输入要求
    # tensor_test_features = tensor_test_dataset[:,:1024]
    # tensor_test_features = tensor_test_features.view(tensor_test_features.shape[0],1,tensor_test_features.shape[1])

    # #取出打乱后的labels
    # tensor_test_labels =tensor_test_dataset[:,-1]



    # #===================================制作 pytorch数据集=============================
    # train_dataset = torch.utils.data.TensorDataset(tensor_train_features,tensor_train_labels)
    # test_dataset = torch.utils.data.TensorDataset(tensor_test_features,tensor_test_labels)

    # #===================================制作train_iter和test_iter=========================
    # train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=64 ,shuffle =True)
    # test_iter = torch.utils.data.DataLoader(test_dataset,batch_size=64 ,shuffle =True)





