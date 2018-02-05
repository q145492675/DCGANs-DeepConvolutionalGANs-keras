# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:11:54 2017

@author: Chuhan Wu
"""

import cv2
import numpy as np
import skimage.io as io
import pandas as pd


###########################################################################################################
###########################################################################################################
def loading(data_dir):
    temp=data_dir+'/*.jpg'
    coll=io.ImageCollection(temp)
    return coll
########################################################################################################### 
def pre_process(img):
    BLACK = [0,0,0]
    top,bottom,left,right=(0,0,0,0)
    length=img.shape[0]
    width=img.shape[1]
    longest_edge=max(length,width)
    if length < longest_edge:
        temp=longest_edge-length
        top=temp//2
        bottom=temp-top
    elif width<longest_edge:
        temp=longest_edge-width
        left=temp//2
        right=temp-left
    else: 
        pass
    constant=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
    return constant

###########################################################################################################
  
def merge(data_1):
    length1=range(len(data_1))   
    temp=np.zeros([(len(data_1)),100,100,3]).astype(np.float32)
 #   temp=np.zeros([len(data_7)*7,350,350,3]).astype(np.float32)
    #temp=temp.astype(np.float32)
    for i in length1:
        img=(data_1[i]-127.5)/127.5
        img=pre_process(img)
        res=cv2.resize(img,(100,100))
        temp[i,:,:,:]=res
    return temp
###########################################################################################################
if __name__ == '__main__':
    coll=loading(data_dir)
    dataset=merge(coll)
