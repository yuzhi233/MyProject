# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:08:32 2020

@author: zhoubo
"""

from matplotlib import pyplot as plt
import matplotlib
import pandas as pd


font = {'family' : 'MicroSoft YaHei',
              'weight' : 'bold',
              'size'   : '8.'}

matplotlib.rc('font',**font)


#读取数据
def get_y(path):
    df =pd.read_csv(path)
    y=df.values
    y=y[:,0]#其实就相当于索引出了这一列 只不过摆放是横着的而且是一维的
    y =y[1000:3000]
    return y



#创建画布 注意是subplots不是subplot
fig,ax=plt.subplots(5,2,dpi=120)#5行2列

#画1772r/min正常轴承的振幅图
path1 ='D:\SPYDER_CODE\MyProject\DataSetA\X098_DE_time.csv'
y1 =get_y(path1)
ax[0,0].plot(y1,color ='g')
ax[0,0].set_title('正常轴承')
ax[0,0].set_xlabel('采样点数')
ax[0,0].set_ylabel('幅值(mm)')

#画0.007 内圈损伤
path2 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X106_DE_time.csv'
y2=get_y(path2)

ax[0,1].plot(y2,color='g')
ax[0,1].set_title('内圈轻度磨损')
ax[0,1].set_xlabel('采样点数')
ax[0,1].set_ylabel('幅值(mm)')

#画0.007 滚动体损伤
path3 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X119_DE_time.csv'
y3=get_y(path3)

ax[1,0].plot(y3,color='g')
ax[1,0].set_title('滚动体轻度磨损')
ax[1,0].set_xlabel('采样点数')
ax[1,0].set_ylabel('幅值(mm)')

#画0.007 外圈故障
path4 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.007-1-1772\\X131_DE_time.csv'
y4=get_y(path4)

ax[1,1].plot(y4,color='g')
ax[1,1].set_title('外圈轻度磨损')
ax[1,1].set_xlabel('采样点数')
ax[1,1].set_ylabel('幅值(mm)')

#画0.014 内圈故障
path5 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X170_DE_time.csv'
y5=get_y(path5)

ax[2,0].plot(y5,color='g')
ax[2,0].set_title('内圈中度磨损')
ax[2,0].set_xlabel('采样点数')
ax[2,0].set_ylabel('幅值(mm)')


#画0.014 滚动体故障
path6 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X185_DE_time.csv'
y6=get_y(path6)

ax[2,1].plot(y6,color='g')
ax[2,1].set_title('滚动体中度磨损')
ax[2,1].set_xlabel('采样点数')
ax[2,1].set_ylabel('幅值(mm)')


#画0.014 外圈故障
path7 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.014-1-1772\\X198_DE_time.csv'
y7=get_y(path7)

ax[3,0].plot(y7,color='g')
ax[3,0].set_title('外圈中度磨损')
ax[3,0].set_xlabel('采样点数')
ax[3,0].set_ylabel('幅值(mm)')

#画 0.021 内圈故障
path8 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X210_DE_time.csv'
y8=get_y(path8)

ax[3,1].plot(y8,color='g')
ax[3,1].set_title('内圈重度磨损')
ax[3,1].set_xlabel('采样点数')
ax[3,1].set_ylabel('幅值(mm)')

# 画0.021 滚动体故障
path9 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X223_DE_time.csv'
y9=get_y(path9)

ax[4,0].plot(y9,color='g')
ax[4,0].set_title('滚动体重度磨损')
ax[4,0].set_xlabel('采样点数')
ax[4,0].set_ylabel('幅值(mm)')

#画0.021 外圈故障
path10 ='D:\\SPYDER_CODE\\MyProject\\DataSetA\\12k-0.021-1-1772\\X235_DE_time.csv'
y10=get_y(path10)

ax[4,1].plot(y10,color='g')
ax[4,1].set_title('外圈重度磨损')
ax[4,1].set_xlabel('采样点数')
ax[4,1].set_ylabel('幅值(mm)')

plt.show()

