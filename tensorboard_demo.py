# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:31:10 2020

@author: yuzhi233
"""

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log')
x = range(100)
for i in x:
    writer.add_scalar('y=3x', i * 3, i)
writer.close()
