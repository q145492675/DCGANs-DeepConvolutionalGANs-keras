# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:12:06 2017

@author: Chuhan Wu

This runtine is based on the concept of DCGANs, it can decide and build the framework of Ganerator and Discriminator
based on the size of the input data automatically. 

The runing time:
    
/------------------------------------------------------------------------/
    GTX 1080Ti + CUDA8.0 +cuDNN6.1  :  25.5 min  (20000Epoch/size:100*100)
/------------------------------------------------------------------------/

                                                      Chuhan Wu
                                                      2/4/2018
"""


import time
import numpy as np
import sys
import os
from keras.models import Sequential
from keras.models import load_model

import load_image
import GANs_trainning

if __name__ == '__main__':
    Base_dir = os.path.dirname(__file__)
    data_dir=Base_dir+'/GANs_dataset'
    coll=load_image.loading(data_dir)
    dataset=load_image.merge(coll)
###########################################################################################################   
    #kernel=int(sys.argv[1])  
    kernel=3
    #depth =int(sys.argv[2])
    depth =40
    #Learning_rate=int(sys.argv[3])
    Learning_rate=0.0003
    #Decay_rate=int(sys.argv[4])
    Decay_rate=3e-8
    #epoch=int(sys.argv[5])
    epoch=25000
    #result_path=sys.argv[6]
    result_path=Base_dir+'/result/result/'
    #predict_path=sys.argv[7]
    predict_path=Base_dir+'/result/predict/'
###########################################################################################################  
    Rows=dataset.shape[1] 
    Cols=dataset.shape[2]
###########################################################################################################
    print('*******************************************************')
    print(' Now we generate two models we need in GANs    \n')
    print(' Please wait....\n')
    print('*******************************************************\n')
    time.sleep(4)   

    G1=Sequential()
    D1=Sequential()

    GANs_trainning.get_generative(G1,kernel,Rows,Cols,depth_base=depth,Lr=Learning_rate,Dc=Decay_rate)
    GANs_trainning.get_discriminative(D1,kernel,Rows,Cols,depth_base=depth,Lr=Learning_rate,Dc=Decay_rate)

    print('\n*******************************************************')
    print(' Now Two model have been created !!    \n')
    print(' Please wait....\n')
    print('*******************************************************\n')
    time.sleep(2)
###########################################################################################################
    GANs_trainning.pretrain(G1,D1)
######s#####################################################################################################   
    [D,L,Training_detail]=GANs_trainning.train(dataset,G1,D1,epochs=epoch,Noise_dim=100,Number=10,batch_size=10)
    GANs_trainning.print_example(G1,1,predict_path)
    G1.save('Generator.h5')
    d_loss=Training_detail[:,0]
    g_loss=Training_detail[:,1]
    GANs_trainning.print_result(d_loss,g_loss,1,result_path)  
     