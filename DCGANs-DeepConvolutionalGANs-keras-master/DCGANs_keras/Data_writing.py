# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:26:38 2017

@author: Chuhan Wu
"""

import keras
import numpy as np  
np.random.seed(1337)  # for reproducibility  
import pandas as pd
import os

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def Write_data_CSV(data,predict_result_path,count,Type='none') :
    ensure_dir(predict_result_path)
    temp=str(count)
    temp1=pd.DataFrame(data)
    temp1=temp1.rename(index=int,columns={2:-1})
    if (Type=='predict'):
        Feature_predict_path=predict_result_path+'predict'+temp+'.csv'
    elif (Type=='result'):
        Feature_predict_path=predict_result_path+'result'+temp+'.csv'
    else:
        print('Type is error, please use a right type of csv')
    temp1.to_csv(Feature_predict_path)


def Write_ModelData(Model,count,Model_path):    
    temp=str(count)
    Model_H5_path=Model_path+temp+'_.h5'
    Model.save(Model_H5_path)
    
def Write_data_singleUse(score,predict_result,predict_result_path,predict_condition_path,count) :
    temp=str(count)
    temp1=pd.DataFrame(predict_result)
    temp2=pd.DataFrame(score)
    temp1=temp1.rename(index=int,columns={2:-1})
    Feature_CV_path=predict_condition_path+'Label_'+temp+'_.csv'
    Feature_predict_path=predict_result_path+'Label_'+temp+'_.csv'
    temp1.to_csv(Feature_predict_path)
    temp2.to_csv(Feature_CV_path)
