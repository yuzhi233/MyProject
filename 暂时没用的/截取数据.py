# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:08:19 2020

@author: zhoubo
"""
#备份 
def get_train_test_data_1(dataFrame,sample_length =1024,sample_nums=470,train_nums=370,slice_stride=None):#需要传入的dataFrame是n行1列的
    #数据类型转换
    dataFrame =dataFrame.values#取出dataFrame的这一列值 是个numpy数组
    dataFrame =dataFrame.astype(np.float32).reshape(1,-1)#数据类型转为float32再将一列数据转换成1行（2维度）
    
    
    
    
    #如果没有指定切片步长 就按不重叠采样
    
        strat_index =0#初始化索引
    #创建用来存放截取样本的list   
        sample_blk=[]
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        