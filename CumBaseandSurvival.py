#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:52:08 2019

@author: Denise
"""

import numpy as np
import torch


def risk_fail_matrix( T, E):
        """ Calculating the Risk, Fail matrices by vectorizing all the quantities at stake
        """

        N = T.shape[0]
        Risk = np.zeros( (N, N) )
        Risk_mod = np.zeros( (N, N) )
        Fail = np.zeros( (N, N) )
        
        order = np.argsort(T)
        T = T[order]
        E = E[order]
        
        for i in range(N):
            
            # At risk
            index_risk = np.argwhere( (T >= T[i]).values ).flatten()
            Risk[ i, index_risk] = 1.
            Risk_mod[ i, index_risk] = 1/float(len(index_risk))
            
            # Failed
            if E[i] == 1 :
                index_fail = np.argwhere( (T == T[i]).values )[0]
                Fail[index_fail, i] = 1.
        return torch.FloatTensor(Risk), torch.FloatTensor(Fail),torch.FloatTensor(Risk_mod)

        

def predict_cumbase( score, T, E,inter,indicator):
        """ Estimate the cumulative baseline
        
            Arguments:
               *score=risk score predicted from the networks for the observations onto which the network has been trained
               *T=censored event time onto which the network has been trained
               *inter=interval 
               *indicator=a vector that indicates at which interval any censored event times belong.
            
            Output: Predicted Cumulative Baseline Hazard \Lambda_0
        """
        M=inter.shape[0]-1 
        N=T.shape[0]
        
        Risk, Fail,Risk_mod = risk_fail_matrix(T, E)
        Risk = torch.FloatTensor(Risk) 
        Fail = torch.FloatTensor(Fail)
        Risk_mod = torch.FloatTensor(Risk_mod)
        
        

        # Calculating the score
        score = torch.FloatTensor(score)
        score = torch.reshape( score, (-1, M) )
        
        #calculating h_bar for each j
        def my_func(x):
            return torch.reshape(torch.mm(Risk_mod,torch.reshape(x,(-1,1))),(-1, 1))
        
        Hbar=torch.stack([my_func(score[:,i]) for i in np.arange(M) ],dim=0)
        
        #detect change of interval
        det=np.diff(indicator)==1
        det=np.insert(det,False,0)
        change=np.arange(N)[det]
        change=np.append(change,N)
        
       
        Hbar_mod=Hbar[0][np.arange(0,change[0])]
        
        for i in np.arange(1,M):
            temp=Hbar[i][np.arange(change[i-1],change[i])]
            Hbar_mod= torch.cat((Hbar_mod,temp))
        
        
        
        #calculating second part
        diff=np.concatenate((np.array(T[0]-0),np.diff(T)),axis=None)
        diff=torch.reshape(torch.FloatTensor(diff),(-1,1)).data.numpy().flatten()
        diff=torch.FloatTensor(diff)
       
        second_part_pre=torch.mul(diff,torch.FloatTensor(torch.reshape( Hbar_mod, (-1, 1) ).data.numpy().flatten()))
        
        second_part=torch.cumsum(second_part_pre,0)
       
        
        #calculating first part
        E=torch.FloatTensor(E)
        first_part=torch.mm(torch.reshape( E, (1, -1) ),Risk_mod)
        first_part=torch.reshape(first_part,(-1,1)).data.numpy().flatten()
        first_part=torch.FloatTensor(first_part)
        
        cum_base_pre=first_part-second_part
        
        cum_base_pre=torch.cat((torch.zeros(1),cum_base_pre))
        
        N=T.shape[0]
        
        cum_base=cum_base_pre[1:(N+1)]
    
        return cum_base

def predict_surv(cumbase,score,T,inter,indicator,use_log = False):
        """ 
        Predicting the survival function
        
        Arguments:
            * cumbase:predicted cumulative baseline
            * score:predicted risk score predicted from the network for the observation for which we want to predict the survival function
            * T:censored event time onto which the network has been trained
            *inter=interval 
            *indicator=a vector that indicates at which interval any censored event times belong.
        
        Output: np.array. Each columns corresponds to the times and each rows corresponds to a different observation
        
        """
                
        
        
        M=score.shape[1]
        N=cumbase.shape[0]
        
        

        #detect change of interval
        det=np.diff(indicator)==1
        det=np.insert(det,False,0)
        change=np.arange(N)[det]
        change=np.append(change,N)
       
      
            
        
        T=torch.FloatTensor(T)
        T_mod=torch.diag(T)
        for i in np.arange(1,M):
            for j in np.arange(change[i-1],change[i]):
                T_mod[j,j]=T_mod[j,j]-T[change[i-1]-1]
        
        N=T_mod.shape[0]
        score_mod=np.repeat(score[:,0],N)
        score_mod=torch.FloatTensor(score_mod)
        score_mod=torch.reshape(score_mod,(score.shape[0],N))
        for i in np.arange(1,M):
            temp=np.repeat(score[:,i],N)
            temp=torch.FloatTensor(temp)
            temp=torch.reshape(temp,(score.shape[0],N))
            score_mod[:,np.arange(change[i-1],change[i])]=temp[:,np.arange(change[i-1],change[i])]
        
        aux=torch.mm(score_mod,T_mod)
        temp=torch.reshape(torch.Tensor(score[:,0])*T[change[0]-1],(-1,1))
        aux[:,np.arange(change[0],change[1])]=aux[:,np.arange(change[0],change[1])]+torch.reshape(torch.Tensor(score[:,0])*T[change[0]-1],(-1,1))
        
        for i in np.arange(2,M):
            temp=temp+torch.reshape(torch.Tensor(score[:,i-1])*T[change[i-1]-1],(-1,1))
            aux[:,np.arange(change[i-1],change[i])]=aux[:,np.arange(change[i-1],change[i])]+temp
        
       
        baseline=np.exp(-cumbase.data.numpy().flatten())
        
        second_part=np.exp(-aux.data.numpy())
        
        survival = second_part*baseline
        
        survival=np.insert(survival, 0, 1, axis=1)
        
        survival=np.minimum.accumulate(survival,1) #adjusted survival
        
        return survival
