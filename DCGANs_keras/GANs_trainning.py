3# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:33:01 2017

@author: Chuhan Wu
"""

import numpy as np
import time
import Data_writing
import load_image
#from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input, Reshape
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop,SGD
#from keras.optimizers import sparse
#from Data_loading import load_features
import matplotlib.pyplot as plt

def d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("The running time of this code: %s " % self.elapsed(time.time() - self.start_time) )

Base_dir = os.path.dirname(__file__)
data_dir=Base_dir+'/GANs_dataset'
coll=load_image.loading(data_dir)
Initial_data_all=load_image.merge(coll)

def Create_Initial_data(number=10):
  #  Initial_data=np.zeros((number,120,40))
  #  i=0
  #  while i<number:
  #      Initial_data[i,:,:]=i*10
  #      Initial_data[i,40:80,:]=i*10+20
  #      Initial_data[i,80:,:]=i*10+40
  #      i=i+1
    Initial_data=Initial_data_all[np.random.randint(0,Initial_data_all.shape[0],size=number),:,:,:]
    return Initial_data

def load_data_info():
    data=Create_Initial_data()
    Rows=data.shape[1]
    Cols=data.shape[2]
    return Rows,Cols

def get_generative(G_Model,kernel,Rows,Cols,input_dim=100,depth_base=32,dropout=0.4,Lr=0.0001,Dc=3e-8,trans_layer_num=1):  
    Rows_cp = Rows
    Cols_cp = Cols
    lower_dim = min(Rows, Cols)
    lyr_max=0
    ### determine the number of layers
    while ( lower_dim % 2 == 0):
        lyr_max +=1
        lower_dim /=2
        Rows_cp /=2
        Cols_cp /=2
        if  Rows_cp % 2 != 0 or Cols_cp % 2 != 0:
            break    
    rows = int(Rows/(2**lyr_max))
    cols = int(Cols/(2**lyr_max))
    depth = depth_base*(lyr_max+1)*2
    
    G_Model.add(Dense(rows*cols*depth, input_dim=input_dim,activation='relu',kernel_initializer='glorot_normal',name='G_Dense_0'))
    G_Model.add(BatchNormalization(momentum=0.9,name='G_BatchNor_0'))
  #  G_Model.add(LeakyReLU(alpha=0.2,name='G_LeakRelu_0'))
    G_Model.add(Reshape((rows, cols, depth)))
    G_Model.add(Dropout(dropout,name='G_Drop_0'))
   
    for lyr in range(1,lyr_max+1):      
        G_Model.add(UpSampling2D(name="G_UpSample_%i" % lyr))
        G_Model.add(Conv2DTranspose(int(depth/(2*lyr)), kernel, activation='relu',kernel_initializer='glorot_normal',padding='same',name="G_TranConv2d_%i" % lyr))
        G_Model.add(BatchNormalization(momentum=0.9,name="G_BatchNor_%i" % lyr))
        #G_Model.add(LeakyReLU(alpha=0.2,name="G_LeakRelu_%i" % lyr))
    
    for trans_layer in range(0,trans_layer_num):
        lyr = lyr_max+1 + trans_layer 
        G_Model.add(Conv2DTranspose(int(depth/(2*lyr)), kernel, activation='relu',padding='same',kernel_initializer='glorot_normal',name="G_TranConv2d_%i" % lyr))
        G_Model.add(BatchNormalization(momentum=0.9,name="G_BatchNor_%i" % lyr))
        #G_Model.add(LeakyReLU(alpha=0.2,name="G_LeakRelu_%i" % lyr))
           
    G_Model.add(Conv2DTranspose(3, kernel,init='glorot_normal', padding='same'))
    G_Model.add(Activation('tanh',name='G_Act_out'))
    #----------------------------------------------------------------------------- 
    G_optimizer=Adam(lr=Lr)
#    G_Model.compile(loss='binary_crossentropy', optimizer=G_optimizer,metrics=['accuracy'])
#    G_Model.compile(loss='categorical_crossentropy', optimizer=G_optimizer,metrics=['accuracy'])
    G_Model.compile(loss='mean_squared_error', optimizer=G_optimizer,metrics=['accuracy'])
    print('\n The Generator is shown like that :    \n')
    G_Model.summary()     
#----------------------------------------------------------------------------- 
#----------------------------------------------------------------------------- 

def get_discriminative(D_Model,kernel,Rows,Cols,depth_base=32,dropout=0.4,Lr=0.0001,Dc=3e-8):
    Rows_cp = Rows
    Cols_cp = Cols
    lower_dim = min(Rows, Cols)
    lyr_max=0
    ### determine the number of layers
    while ( lower_dim % 2 == 0):
        lyr_max +=1
        lower_dim /=2
        Rows_cp /=2
        Cols_cp /=2
        if  Rows_cp % 2 != 0 or Cols_cp % 2 != 0:
            break
    print(lyr_max+1)
    Input_shape =(Rows,Cols,3)
    
    for lyr in range(1,(lyr_max+1)):
        D_Model.add(Conv2D(depth_base*2*lyr,kernel, strides=2, input_shape=Input_shape,padding='same',kernel_initializer='glorot_normal',name="D_Conv2d_%i" % lyr))
        D_Model.add(LeakyReLU(alpha=0.2,name="D_LRelu%i" % lyr))
        D_Model.add(Dropout(dropout,name="D_Drop_%i" % lyr))

    D_Model.add(Conv2D(depth_base*2*(lyr_max+1), kernel, strides=1, padding='same',kernel_initializer='glorot_normal',name="D_Conv2d_%i" % (lyr_max+1)))
    D_Model.add(LeakyReLU(alpha=0.2,name="D_LRelu%i" % (lyr_max+1)))
    D_Model.add(Dropout(dropout,name="D_Drop_%i" % (lyr_max+1)))     
    ##################################################################################################################
    D_Model.add(Conv2D(depth_base*2*(lyr_max+1), kernel, strides=1, padding='same',kernel_initializer='glorot_normal',name="D_Conv2d_%i" % (lyr_max+2)))
    D_Model.add(LeakyReLU(alpha=0.2,name="D_LRelu%i" % (lyr_max+2)))
    D_Model.add(Dropout(dropout,name="D_Drop_%i" % (lyr_max+2)))     
    ##################################################################################################################
    # Out: 1-dim probability
    D_Model.add(Flatten())
    D_Model.add(Dense(1))
#-----------------------------------------------------------------------------        
    D_Model.add(Activation('sigmoid'))
    #D_Model.add(LeakyReLU(alpha=0.2))
#-----------------------------------------------------------------------------
    rate=Lr*1
    D_optimizer=SGD(lr=rate)#,decay=Dc)
    D_Model.compile(loss='binary_crossentropy', optimizer=D_optimizer,metrics=['accuracy'])
    print(' The Discriminator is shown like that :    \n')
    D_Model.summary()
#----------------------------------------------------------------------------- 
#----------------------------------------------------------------------------- 

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
        
def make_gan(i, G, D,noise_dim=100):
    set_trainability(D, False)
    GAN=Sequential()
    GAN.add(G)
    GAN.add(D)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer,metrics=['accuracy'])
    if (i==0):
        print(' The Adversarial Model is shown like that :    \n')
        GAN.summary()
    return GAN

def sample_data_and_gen(Generator,dataset=0,noise_dim=100,Fake_number=10,Number=10):
    X_Train = Create_Initial_data(number=Number)
    #X_Train = dataset
    #X_Train=X_Train.reshape(X_Train.shape[0],X_Train.shape[1],X_Train.shape[2],1)
    Truth_number=X_Train.shape[0]
    X_noise = np.random.normal(-1.0,1.0, size=[Fake_number, noise_dim])
    X_Fake = Generator.predict(X_noise)
    X = np.concatenate((X_Train, X_Fake))
    y = np.zeros((Truth_number+Fake_number, 1))
    y[0:Truth_number] = 1
    return X, y

def Batch_size(per,data):
    Truth_number=data.shape[0]
    temp1=int(Truth_number*per)
    temp2=temp1-Truth_number*per
    if (temp2 ==0):
        print('Error: The Batch_size should devided by the number of dataset!\n')
        exit(0)
    return temp1

def pretrain(G, D, noise_dim=100,per=1):
    X, y = sample_data_and_gen(G,noise_dim=noise_dim)
    set_trainability(D, True)
#    batch_size=Batch_size(per)
    print('/*******************************************************/\n')
    print(' The Pre_Training of Discriminator is working.......\n')
    print(' Please wait....')
    print('/*******************************************************/\n')
    time.sleep(2)
    D.fit(X, y, epochs=1, batch_size=10)
    print('/*******************************************************/\n')
    print(' The Pre_Training of Discriminator is sucessed!\n')
    print(' Please wait....')
    print('/*******************************************************/\n')
    time.sleep(5)
    
def sample_noise( noise_dim=100,number=5):
    X = np.random.normal(-1.0,1.0, size=[number, noise_dim])
    y = np.ones((number, 1))
    return X, y

def train(dataset,G, D, epochs=10000, Noise_dim=100,Number=5,batch_size=32):
    Training_detail=np.zeros(((epochs+1),2))
    reconstruction_error = []
    epoch=0
    timer = ElapsedTimer()
    GAN=make_gan(0,G,D)
    print('/*******************************************************/\n')
    print(' Now we begin to training DC_GANs model.\n')
    print('/*******************************************************/\n')    
    while epoch<epochs:
        X0, y0 = sample_data_and_gen(G,dataset=0,noise_dim=Noise_dim,Fake_number=Number,Number=Number)
        set_trainability(D, True)
        d_loss=D.train_on_batch(X0, y0)
#-----------------------------------------------------------------------------         
        X1, y1 = sample_noise(noise_dim=Noise_dim,number=Number)
        set_trainability(D, False)
      #  GAN=make_gan(epoch,G,D)
        g_loss=GAN.train_on_batch(X1, y1)
        log_mesg ="%d: [Discriminator loss : %f, Acc : %f]" % (epoch, d_loss[0], d_loss[1])
        log_mesg ="%s  [Generator    loss : %f, Acc :  %f]" % (log_mesg, g_loss[0], g_loss[1])
        Training_detail[epoch,0]=d_loss[0]
        Training_detail[epoch,1]=g_loss[0]
        print(log_mesg)
        epoch=epoch+1
    
   # test_dataset = dataset[0:20,:,:]
#-----------------------------------------------------------------------------         
    test_dataset=Create_Initial_data(number=20)
    test=np.random.normal(-1.0,1.0,size=[test_dataset.shape[0],Noise_dim])
    reconction_sets=G.predict(test)
     # calculate reconstruction error
    v = reconction_sets-test_dataset
    reconstruction_error.append((v*v).sum()/(test_dataset.shape[0])) 
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n') 
#-----------------------------------------------------------------------------         
    return d_loss, g_loss,Training_detail

def print_example(Generator,count,predict_path,number=10,Noise_dim=100):
        test=np.random.normal(-1.0,1.0,size=[number,Noise_dim])
        Example=Generator.predict(test)
        Example=Example.reshape(-1,Example.shape[1],Example.shape[2],3)        
        Num=Example.shape[0]
        for i in range(Num):
          data=Example[i,:,:,:]
          data=(data*255).astype(int)
          name=predict_path+'_'+str(i)+'.jpg'
          cv2.imwrite(name,data)
          
def print_result(d_loss,g_loss,count,result_path):
        data = np.concatenate((d_loss, g_loss))
        Data_writing.Write_data_CSV(data,result_path,count,Type='result')