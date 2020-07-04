#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 22:36:46 2020

@author: Denise
"""

import ApplyDeepHaz as dh
import pandas as pd
import numpy as np


''' EXAMPLE FOR A DATASET WITH TIME VARYING COVARIATES
'''
'''a)Import from csv train and test dataset as panda dataframe
          IMPORTANT:Time column needs to be called 'Time',Event indicator column needs to be called 'Event',
          covariates need to be called : 'Variable_ij' where i is the number of the variable
          j is the interval onto which the variable gets that value
'''    
train=pd.read_csv('sampletraintimevar.csv',delimiter=',')
test=pd.read_csv('sampletesttimevar.csv',delimiter=',')

'''b)Indicate the extremes of the intervals. 
             In this case the covariates are measured at time 0.2,0.4,0.6. Always start at 0 and use as last number a number that is bigger than the last event time in your data
'''
inter=np.array((0,0.2,0.4,0.6,100)) #set interval

'''c)define the structure of the NNs. In this case we have 2 layers with 10 nodes each. In each layer dropout is 0.2 and the activation function is Relu.
'''
structure = [{'activation': 'Relu','num_units':10,'dropout':0.2},{'activation': 'Relu','num_units':10,'dropout':0.2}] 

'''d)Train the model onto the training dataset and predict the survival onto the test data. Compute the C_index as metrics
     This function return a list of trained NNs, an array with the predicted survival (rows are observations, columns are times) and the time_varying C_index
     For an explanations of the arguments, see ApplyDeepHaz.py
'''
deepHazlis,Surv,C_index=dh.DeepHazTime(train=train,test=test,inter=inter,Ncol=3,l2c=1e-5,lrc=2e-1,structure=structure,init_method='he_uniform',optimizer='adam',num_epochs=1000,early_stopping=1e-5,penal='Ridge')



''' EXAMPLE FOR A DATASET WITH CONSTANT COVARIATES
'''
'''a)Import from csv train and test dataset as panda dataframe
          IMPORTANT:Time column needs to be called 'Time',Event indicator column needs to be called 'Event',
'''    
train=pd.read_csv('sampletrainconst.csv',delimiter=',')
test=pd.read_csv('sampletestconst.csv',delimiter=',')

'''b)define the structure of the NNs. In this case we have 2 layers with 10 nodes each. In each layer dropout is 0.2 and the activation function is Relu.
'''
structure = [{'activation': 'Relu','num_units':10,'dropout':0.2},{'activation': 'Relu','num_units':10,'dropout':0.2}] 

'''c)Train the model onto the training dataset and predict the survival onto the test data. Compute the C_index as metrics
     This function return the trained NN, an array with the predicted survival (rows are observations, columns are times) and the C_index
     For an explanations of the arguments, see ApplyDeepHaz.py
'''
deephaz,Surv,C_index=dh.DeepHazConst(train,test,l2c=1e-5,lrc=2e-1,structure=structure,init_method='he_uniform',optimizer='adam',num_epochs=1000,early_stopping=1e-5,penal='Ridge')

   