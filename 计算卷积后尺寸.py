# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:11:22 2020

@author: zhoubo
"""

size =int(input('size:\n'))
stride =int(input('stride\n'))
padding =int(input('padding:\n'))
kernel_size =int(input('kernel size:\n'))



if (size +2*padding-kernel_size)%stride != 0:
    print('没有被整除，卷积后地板除后尺寸为:\n',(size +2*padding-kernel_size)//stride +1)
else:
    print('卷积后尺寸为:',(size +2*padding-kernel_size)/stride +1)


0