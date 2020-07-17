#!/usr/bin/env python2
# -*- coding: utf-8 -*-
 
#Created on Fri Jul  3 13:35:10 2020
import numpy as np
import bisect
import DeepHaz as dhn
import CumBaseandSurvival as cbs
import concordance_index_time as cit
from pysurvival.utils._metrics import _concordance_index



    

def Createtrainingsubset(inter,train,Ncol):
   #Creating all the subsets that are needed for training the model with time varying covariates
   
    #define the subsets
    M=inter.shape[0]-1
    
    T_train=train['Time']
    E_train=train['Event']
    
    Variable_Name=[]
    for x in range(1,Ncol+1):
        name='Variable_'+"{0}".format(x)
        Variable_Name.append(name)
    
    
    subset_list = []
    X_train_list = []
    T_train_list = []
    E_train_list = []
    X_train_final_list=[]
    subset1 = train.copy()
    subset1['Event'][subset1['Time']>inter[1]]=0
    subset1['Time'][subset1['Time']>inter[1]]=inter[1]
    b=["{0}".format(1)]*Ncol
    ColName=[''.join(i).strip() for i in zip(Variable_Name,b)]
    X_train1=subset1[ColName]
    X_train_final=train[ColName]
    X_train_final_list.append(X_train_final)
    T_train1=subset1['Time']
    E_train1=subset1['Event']
    subset_list.append(subset1)
    X_train_list.append(X_train1)
    T_train_list.append(T_train1)
    E_train_list.append(E_train1)
   
    for x in range(2,M+1):
        train_temp=train.copy()
        subset= train_temp[train_temp['Time']>inter[x-1]]
        subset['Event'][subset['Time']>inter[x]]=0
        subset['Time'][subset['Time']>inter[x]]=inter[x]
        b=["{0}".format(x)]*Ncol
        ColName=[''.join(i).strip() for i in zip(Variable_Name,b)]
        X_train=subset[ColName]
        T_train2=subset['Time']
        E_train2=subset['Event']
        X_train_final=train[ColName]
        X_train_final_list.append(X_train_final)
        X_train_int_list=[]
        X_train_int_list.append(X_train)
        T_train_list.append(T_train2)
        E_train_list.append(E_train2)
        for i in range(1,x):
          b=["{0}".format(i)]*Ncol
          ColName=[''.join(i).strip() for i in zip(Variable_Name,b)]
          X_train2=subset[ColName]
          X_train_int_list.append(X_train2)
        X_train_list.append(X_train_int_list)
    return(X_train_list,T_train_list,E_train_list,X_train_final_list,T_train,E_train)
       
def Createtestsubset(inter,test,Ncol):
    #Creating all the subsets that are needed for applying the model with time varying covariates
    
    #define the subsets
    M=inter.shape[0]-1
    
    T_test=test['Time']
    E_test=test['Event']
    
    Variable_Name=[]
    for x in range(1,Ncol+1):
        name='Variable_'+"{0}".format(x)
        Variable_Name.append(name)
    
    X_test_list=[]
    for x in range(1,M+1):
        b=["{0}".format(x)]*Ncol
        ColName=[''.join(i).strip() for i in zip(Variable_Name,b)]
        X_test=test[ColName]
        X_test_list.append(X_test)
    
    T_test=test['Time']
    E_test=test['Event']
    
    return(X_test_list,T_test,E_test)
    
    
def TrainDeepHazTime(train,inter,Ncol,l2c,lrc,structure,init_method,optimizer,num_epochs,early_stopping,penal):
   #Training the model with time_varying covariates
  
    # prepare the data by creating the datasets D
    X_train_list,T_train_list,E_train_list,X_train_final_list,T_train,E_train=Createtrainingsubset(inter,train,Ncol)
    M=inter.shape[0]-1
    deepHazlis=[]
    Ttemp=T_train_list[0]
    Etemp=E_train_list[0]
    Xtemp=X_train_list[0]
    
    # say what this function is doing ?
    deephaz1 = dhn.DeepHaz(structure=structure)
    
    # say what this function is doing ?
    deephaz1.fit(Xtemp, Ttemp, Etemp, lr=lrc, init_method=init_method,optimizer=optimizer,num_epochs=num_epochs,l2_reg=l2c,early_stopping=early_stopping,penal=penal)
    
    deepHazlis.append(deephaz1)
    score_list=[]
    score1=deephaz1.predict_risk(X_train_final_list[0])
    score1.shape=(score1.shape[0],1)
    score_list.append(score1)
    scoretemp=deephaz1.predict_risk(X_train_list[1][1])
    scoretemp.shape=(scoretemp.shape[0],1)
    X_train2n=np.concatenate((X_train_list[1][0],scoretemp),1)
    
    # running M neural-networks?
    for x in range(2,M):
        Ttemp=T_train_list[x-1]
        Etemp=E_train_list[x-1]
        #specify the network structure
        # we can easily here allow for some different learning rates or different structures, right ?
        deephaz2 = dhn.DeepHaz(structure=structure)
        
        #X_train2n seems to be the same for each itteration
        deephaz2.fit(X_train2n, Ttemp, Etemp, lr=lrc, t_start=inter[x-1],init_method='he_uniform',optimizer='adam',num_epochs=1000,l2_reg=l2c,early_stopping=1e-5,penal='Ridge')
        deepHazlis.append(deephaz2)
        trainscore=X_train_final_list[x-1]
        for i in range(x-2,-1,-1):
            trainscore=np.concatenate((trainscore,score_list[i]),1)
        score2=deephaz2.predict_risk(trainscore)
        score2.shape=(score2.shape[0],1)
        score_list.append(score2)
        score_temp=[]
        
        #can you explain for me what is each for loop doing and where does one end and another begin?
        for i in range(x):
         trainscore=X_train_list[x][i+1]
         for j in score_temp[::-1]:
            trainscore=np.concatenate((trainscore,j),1)
         score31=deepHazlis[i].predict_risk(trainscore)
         score31.shape=(score31.shape[0],1)
         score_temp.append(score31)
        X_train2n=X_train_list[x][0]
        
        for j in score_temp[::-1]:
            X_train2n=np.concatenate((X_train2n,j),1)
    Ttemp=T_train_list[M-1]
    Etemp=E_train_list[M-1]
    deephaz2 = dhn.DeepHaz(structure=structure)
    deephaz2.fit(X_train2n, Ttemp, Etemp, lr=lrc, t_start=inter[x-1],init_method='he_uniform',optimizer='adam',num_epochs=1000,l2_reg=l2c,early_stopping=1e-5,penal='Ridge')
    deepHazlis.append(deephaz2)
    trainscore=X_train_final_list[M-1]
    for i in range(M-2,-1,-1):
        trainscore=np.concatenate((trainscore,score_list[i]),1)
    score2=deephaz2.predict_risk(trainscore)
    score2.shape=(score2.shape[0],1)
    score_list.append(score2)
    
    score=score_list[0].reshape((-1,1))
    for j in range(1,M):
       score=np.concatenate((score,score_list[j].reshape((-1,1))), axis=1)
    
    def ind(t):
        return bisect.bisect_left(inter, t)
    indicator=map(ind, T_train)

    cumbase=cbs.predict_cumbase(score, T_train, E_train,inter,indicator)
    
    time=T_train
    
    # I didn't see where was one neetwork fed into the future one ?
    
    return(deepHazlis,score,cumbase,time)
  
def PredictDeepHazTime(inter,test,Ncol,deepHazlis,cumbase,time):
   #Use the model to predict survival function with time varying covariates
  
    M=inter.shape[0]-1
    score_list=[]
    X_test_list,T_test,E_test=Createtestsubset(inter,test,Ncol)
    score1=deepHazlis[0].predict_risk(X_test_list[0])
    score1.shape=(score1.shape[0],1)
    score_list.append(score1)
    for i in range(1,M):
        testscore=X_test_list[i]
        for j in score_list[::-1]:
            testscore=np.concatenate((testscore,j),1)
        score1=deepHazlis[i].predict_risk(testscore)
        score1.shape=(score1.shape[0],1)
        score_list.append(score1)
    score=score_list[0].reshape((-1,1))
    for j in range(1,M):
       score=np.concatenate((score,score_list[j].reshape((-1,1))), axis=1)
    
    def ind(t):
        return bisect.bisect_left(inter, t)

    
    indicator=map(ind, time)
    
    Surv=cbs.predict_surv(cumbase,score,time,inter,indicator,use_log = False)
    
    return(score,Surv)

   
def DeepHazTime(train,test,inter,Ncol,l2c,lrc,structure,init_method,optimizer,num_epochs,early_stopping,penal):
    """ Training DeepHazard on training data with time varying covariates and use the model to predict the Survival function onto a test dataset.
    Parameters:
        -----------
        * train : pandas dataframe that contains the training data. Time column needs to be called 'Time',Event indicator column needs to be called 'Event', 
                  Variables need to be called : 'Variable_ij' where i is the number of the variable
                  j is the interval onto which the variable gets that value
        * test : pandas dataframe that contains the test data. Time columns needs to be called 'Time',Event indicator columns needs to be called 'Event', 
                  Variables need to be called : 'Variable_ij' where i is the number of the variable
                  j is the interval onto which the variable gets that value
        * inter : np.array with the extremes of the intervals. 
        * Ncol : Dimesion of covariates
        * lrc : **float** *(default=1e-4)* -- 
            learning rate used in the optimization
        * l2c : **float** *(default=1e-4)* -- 
            regularization parameter for the model coefficients
        * structure: List of dictionaries
                ex: structure = [ {'activation': 'relu', 'num_units': 128,'dropout':0.2}, 
                                  {'activation': 'tanh', 'num_units': 128,'dropout':0.2}, ] 
        * init_method : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:
            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer 
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,
        * optimizer :  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:
            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`
        * num_epochs: **int** *(default=1000)* -- 
            The number of iterations in the optimization
        * early_stopping: early stopping tolerance
        * penal: 'Ridge' if we want to apply Ridge penalty to the loss
                 'Lasso' if we want to apply Lasso penalty to the loss        
    Outputs:
        -----------
        * deepHazlis : A list that contains the trained networks
        * Surv: np.array with the predicted survival, each rows correspond to a different observation in the test set
                each column correspond to different times (times in the training data)
        * Time dependent concorance index from
           Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
           index for survival data. Statistics in Medicine 24:3927–3944.
  
    """
    deepHazlis,score,cumbase,time=TrainDeepHazTime(train,inter,Ncol,l2c,lrc,structure,init_method,optimizer,num_epochs,early_stopping,penal)
    score,Surv=PredictDeepHazTime(inter,test,Ncol,deepHazlis,cumbase,time)
    X_test_list,T_test,E_test=Createtestsubset(inter,test,Ncol)
    T_test=np.array(T_test)
    E_test=np.array(E_test)
    C_index=cit.concordance_td(T_test, E_test, np.transpose(Surv), np.arange(T_test.shape[0]), method='antolini')
    return(deepHazlis,Surv,C_index)
          
    
def DeepHazConst(train,test,l2c,lrc,structure,init_method,optimizer,num_epochs,early_stopping,penal):
    """ Training DeepHazard on training data with time varying covariates and use the model to predict the Survival function onto a test dataset.
    Parameters:
        -----------
        * train : pandas dataframe that contains the training data. Time column needs to be called 'Time',Event indicator column needs to be called 'Event', 
        * test : pandas dataframe that contains the test data. Time columns needs to be called 'Time',Event indicator columns needs to be called 'Event', 
        * inter : np.array with the extremes of the intervals. 
        * lrc : **float** *(default=1e-4)* -- 
            learning rate used in the optimization
        * l2c : **float** *(default=1e-4)* -- 
            regularization parameter for the model coefficients
        * structure: List of dictionaries
                ex: structure = [ {'activation': 'relu', 'num_units': 128,'dropout':0.2}, 
                                  {'activation': 'tanh', 'num_units': 128,'dropout':0.2}, ] 
        * init_method : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:
            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer 
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,
        * optimizer :  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:
            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`
        * num_epochs: **int** *(default=1000)* -- 
            The number of iterations in the optimization
        * early_stopping: early stopping tolerance
        * penal: 'Ridge' if we want to apply Ridge penalty to the loss
                 'Lasso' if we want to apply Lasso penalty to the loss        
    Outputs:
        -----------
        * deepHaz : The trained Network
        * Surv: np.array with the predicted survival, each rows correspond to a different observation in the test set
                each column correspond to different times (times in the training data)
        * C_index
    """
  
    T_train=train['Time']
    E_train=train['Event']
    X_train=train.copy()
    X_train=X_train.drop(['Time','Event'],axis=1)
    T_test=test['Time']
    E_test=test['Event']
    X_test=test.copy()
    X_test=X_test.drop(['Time','Event'],axis=1)
    deephaz = dhn.DeepHaz(structure=structure)
    deephaz.fit(X_train, T_train, E_train, lr=lrc, init_method=init_method,optimizer=optimizer,num_epochs=num_epochs,l2_reg=l2c,early_stopping=early_stopping,penal=penal)
    Surv=deephaz.predict_surv(X_test,use_log = False)
    score=deephaz.predict_risk(X_test,use_log = False)
    order = np.argsort(-T_test)
    score = score[order]
    T_test = T_test[order]
    E_test = E_test[order]
    C_index = _concordance_index(score, T_test, E_test, True)[0]
    return(deephaz,Surv,C_index)
    

