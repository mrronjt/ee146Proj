#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:23:23 2019

@author: ron
"""

import numpy as np 
import glob , os

direc = "./test_images/CU/CU1"
file_list = sorted(glob.glob(os.path.join(direc,"*.txt")))
print(file_list[179])

openf = file_list[10]
f = open(openf)
with f as fp:
    for i, line in enumerate(fp):
        if i == 2:
            third = line.split(' ') 
        if i == 1:
            second = line.split(' ')
#            print(third)
#print(second)
for i in range(len(third)-1):
    third[i]=float(third[i])#make a list of float
for i in range(len(second)-1):
    second[i]=float(second[i])#make a list of float

    
third = third[:-1]
second = second[:-1]
rightx=[]
righty=[]
leftx=[]
lefty=[]
for i in range(0,len(third)-1,2):
    
    rightx.append(third[i])
    righty.append(third[i+1])
for i in range(0,len(second)-1,2):
    
    leftx.append(second[i])
    lefty.append(second[i+1])

print(leftx)


#print(third[:-1]) #get rid of last element "/n
#print(len(third))

#print(third[2]+1)


#lines = f.read().split(' ')"
#print(lines)
    