#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:12:30 2019

@author: Denise
"""

"""
Created on Tue Jun  4 09:50:53 2019

@author: Denise
"""

import torch 
import numpy as np
import copy
from pysurvival.models import BaseModel
from pysurvival import utils

import progressbar
import NNdifDrop as nn



class DeepHaz(BaseModel):
    """    * structure: None or list of dictionaries
                ex: structure = [ {'activation': 'relu', 'num_units': 128}, 
                                  {'activation': 'tanh', 'num_units': 128}, ] 
                Here are the possible activation functions:
                    * Atan
                    * BentIdentity
                    * BipolarSigmoid
                    * CosReLU
                    * ELU
                    * Gaussian
                    * Hardtanh
                    * Identity
                    * InverseSqrt
                    * LeakyReLU
                    * LeCunTanh
                    * LogLog
                    * LogSigmoid
                    * ReLU
                    * SELU
                    * Sigmoid
                    * Sinc
                    * SinReLU
                    * Softmax
                    * Softplus
                    * Softsign
                    * Swish
                    * Tanh
            * auto_scaler: boolean (default=True)
                Determines whether a sklearn scaler should be automatically 
                applied             
    """
    def optimize(self,loss_function, model, optimizer_str, lr=1e-4, nb_epochs=1000, 
               verbose = True, num_workers = 0,early_stopping=None, **kargs):
        W = model.parameters()
        if optimizer_str.lower() == 'adadelta':
                optimizer = torch.optim.Adadelta(W, lr=lr) 
        
        elif optimizer_str.lower() == 'adagrad':
                optimizer = torch.optim.Adagrad(W, lr=lr) 
    
        elif optimizer_str.lower() == 'adam':
                optimizer = torch.optim.Adam(W, lr=lr) 
    
        elif optimizer_str.lower() == 'adamax':
                optimizer = torch.optim.Adamax(W, lr=lr)     
    
        elif optimizer_str.lower() == 'rmsprop':
                optimizer = torch.optim.RMSprop(W, lr=lr)  
    
        elif optimizer_str.lower() == 'sparseadam':
                optimizer = torch.optim.SparseAdam(W, lr=lr)  
    
        elif optimizer_str.lower() == 'sgd':
                optimizer = torch.optim.SGD(W, lr=lr)  

        elif optimizer_str.lower() == 'lbfgs':
                optimizer = torch.optim.LBFGS(W, lr=lr)
    
        elif optimizer_str.lower() == 'rprop':
                optimizer = torch.optim.Rprop(W, lr=lr)

        else:
                error = "{} optimizer isn't implemented".format(optimizer_str)
                raise NotImplementedError(error)
    
    # Initializing the Progress Bar
        loss_values = []
        if verbose:
                widgets = [ '% Completion: ', progressbar.Percentage(), 
                   progressbar.Bar('*'), ''] 
                bar = progressbar.ProgressBar(maxval=nb_epochs, widgets=widgets)
                bar.start()

    # Updating the weights at each training epoch
        temp_model = None
        for epoch in range(nb_epochs):

        # Backward pass and optimization
            def closure():
                 optimizer.zero_grad()
                 loss = loss_function(model, **kargs)
                 loss.backward()
                 return loss

            if 'lbfgs' in optimizer_str.lower() :
                 optimizer.step(closure)
            else:
                 optimizer.step()
            loss = closure()
            loss_value = loss.item()

        # Printing error message if the gradient didn't explode
            if np.isnan(loss_value) or np.isinf(loss_value):
                error = "The gradient exploded... "
                error += "You should reduce the learning"
                error += "rate (lr) of your optimizer"
                if verbose:
                    widgets[-1] = error
                else:
                    print(error)
                break
            
        # Otherwise, printing value of loss function
            else:
                temp_model = copy.deepcopy(model)
                loss_values.append( loss_value )
            if verbose:
                widgets[-1] = "Loss: {:6.2f}".format( loss_value )

        # Updating the progressbar
            if verbose:
                bar.update( epoch + 1 )
            
            if early_stopping is not None and epoch > 1:
                assert isinstance(early_stopping, float), "early_stopping should be either None or float value. Got {}".format(early_stopping)
                eval_loss_diff = np.abs(loss_values[-2] - loss_values[-1])
                if eval_loss_diff < early_stopping:
                   print("Evaluation loss stopped decreased less than {}. Early stopping at epoch {}.".format(early_stopping, epoch))
                   break
    
    # Terminating the progressbar
        if verbose:
            bar.finish()
        
    # Finilazing the model
        if temp_model is not None:
            temp_model = temp_model.eval()
            model = copy.deepcopy(temp_model)
        else:
            raise ValueError(error)

        return model, loss_values


    def __init__(self, structure=None, auto_scaler = True):
        
        # Saving attributes
        self.structure = structure
        self.loss_values = []
        
        # Initializing the elements from BaseModel
        super(DeepHaz, self).__init__(auto_scaler)



    def risk_fail_matrix(self, T, E):
        """ Calculating the Risk, Fail matrices to calculate the loss 
            function by vectorizing all the quantities at stake
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
            index_risk = np.argwhere( T >= T[i] ).flatten()
            Risk[ i, index_risk] = 1.
            Risk_mod[ i, index_risk] = 1/float(len(index_risk))
            
            # Failed
            if E[i] == 1 :
                index_fail = np.argwhere( T == T[i] )[0]
                Fail[index_fail, i] = 1.

        self.nb_fail_per_time = np.sum( Fail, axis = 1 ).astype(int)
        return torch.FloatTensor(Risk), torch.FloatTensor(Fail),torch.FloatTensor(Risk_mod)



    def loss_function(self, model, X,E,T, Risk,Risk_mod, Fail, 
                       l2_reg,t_start,penal):
        """ Calculating the Loss function
        """
        
        N, self.num_vars = X.shape

        # Calculating the score
        score = model(X)
        score = torch.reshape( score, (-1, 1) )
        
        #calculating h_bar
        Hbar=torch.mm(Risk_mod,score)
        Hbar=torch.reshape( Hbar, (-1, 1) )

        # second part calculation
        second_part_pre = score-Hbar
        second_part =torch.mm(torch.reshape( torch.FloatTensor(2*E/N), (1, -1) ),second_part_pre)  

        # First part calculation
        diff=np.concatenate((np.array(T[0]-t_start),np.diff(T)),axis=None)
        diff=torch.reshape(torch.FloatTensor(diff),(-1,1))
        diff_fin=torch.reshape(diff.repeat(1,N),(-1,1))
        
        score_fin=score.repeat((N,1))
        
        mod_Hbar=torch.reshape(Hbar.repeat(1,N),(-1,1))
        
        first_part_pre=(score_fin-mod_Hbar)**2
        first_part_pre=torch.mul(diff_fin,first_part_pre)/N
        
        aux=torch.reshape(Risk,(1,-1))
    
        first_part=torch.mm(aux,first_part_pre)
        
        
        # Adding regularization
        loss =  first_part - second_part
        
        if (penal=='Ridge'):
            for w in model.parameters():
               loss += l2_reg*torch.sum(w*w)/2.
        if (penal=='Lasso'):
            for w in model.parameters():
                loss += l2_reg*torch.sum(torch.abs(w))
                
            
        return loss


    def fit(self, X, T, E, init_method = 'glorot_uniform',
            optimizer ='adam', lr = 1e-4, num_epochs = 1000,batch_normalization=False, bn_and_dropout=False,
            l2_reg=1e-5, verbose=True, early_stopping=None,t_start=0,penal='Ridge'):
        """ 
        Fit the estimator based on the given parameters.
        Parameters:
        -----------
        * `X` : **array-like**, *shape=(n_samples, n_features)* --
            The input samples.
        * `T` : **array-like** -- 
            The target values describing when the event of interest or censoring
            occurred.
        * `E` : **array-like** --
            The values that indicate if the event of interest occurred i.e.: 
            E[i]=1 corresponds to an event, and E[i] = 0 means censoring, 
            for all i.
        * `init_method` : **str** *(default = 'glorot_uniform')* -- 
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
        * `optimizer`:  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:
            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`
        * `lr`: **float** *(default=1e-4)* -- 
            learning rate used in the optimization
        * `num_epochs`: **int** *(default=1000)* -- 
            The number of iterations in the optimization
        * `dropout`: **float** *(default=0.5)* -- 
            Randomly sets a fraction rate of input units to 0 
            at each update during training time, which helps prevent overfitting.
        * `l2_reg`: **float** *(default=1e-4)* -- 
            L2 regularization parameter for the model coefficients
        * `batch_normalization`: **bool** *(default=True)* -- 
            Applying Batch Normalization or not
        * `bn_and_dropout`: **bool** *(default=False)* -- 
            Applying Batch Normalization and Dropout at the same time
        * `verbose`: **bool** *(default=True)* -- 
            Whether or not producing detailed logging about the modeling
                
        
        """

        # Checking data format (i.e.: transforming into numpy array)
        X, T, E = utils.check_data(X, T, E)

        # Extracting data parameters
        N, self.num_vars = X.shape
        input_shape = self.num_vars
        
        # Scaling data 
        if self.auto_scaler:
           X_original = self.scaler.fit_transform( X ) 
        else:
           X_original=X# Ensuring x has 2 dimensions
        if X.ndim == 1:
           X_original = np.reshape(X_original, (1, -1))
        # Sorting X, T, E in ascending order according to T
        order = np.argsort(T)
        T = T[order]
        E = E[order]
        X_original = X_original[order, :]
        #X_original = X[order]
        self.times = np.unique(T[E.astype(bool)])
        self.nb_times = len(self.times)
        self.get_time_buckets()

        # Initializing the model
        model = nn.NeuralNet(input_shape, 1, self.structure, 
                             init_method, batch_normalization, 
                             bn_and_dropout )

        # Looping through the data to calculate the loss
        X = torch.reshape(torch.FloatTensor(X_original),(-1,self.num_vars))

        # Computing the Risk and Fail tensors
        Risk, Fail,Risk_mod = self.risk_fail_matrix(T, E)
        Risk = torch.FloatTensor(Risk) 
        Fail = torch.FloatTensor(Fail)
        Risk_mod = torch.FloatTensor(Risk_mod)
        # Performing order 1 optimization
        model, loss_values = self.optimize(self.loss_function, model, optimizer, 
            lr, num_epochs, verbose,early_stopping=early_stopping, X=X,E=E,T=T, Risk=Risk,Risk_mod=Risk_mod, Fail=Fail, l2_reg=l2_reg,t_start=t_start,penal=penal)

        # Saving attributes
        self.model = model.eval()
        self.loss_values = loss_values
        
        score=self.model(torch.FloatTensor(X)).data.numpy().flatten()
        self.cumbase=self.predict_cumbase(score,T ,E)
        
        T=torch.FloatTensor(T)
        T_mod=torch.diag(T)
        self.T_mod=T_mod

        

        
        return self


    def predict_surv(self, x,use_log = False):
        """ 
        Predicting the survival function
        
        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
            * t: float (default=None)
                Time at which hazard, density and survival functions
                should be calculated. If None, the method returns 
                the functions for all times t. 
        """
                
        cumbase=self.cumbase
      
            
        # Calculating risk_score, hazard, density and survival 
        score    = self.predict_risk(x, use_log = False)
        #auxiliary quantity
        
        T_mod=self.T_mod
        N=T_mod.shape[0]
        score_mod=np.repeat(score,N)
        score_mod=torch.FloatTensor(score_mod)
        score_mod=torch.reshape(score_mod,(score.shape[0],N))
        aux=torch.mm(score_mod,T_mod)
        
        
        baseline=np.exp(-cumbase.data.numpy().flatten())
        
        second_part=np.exp(-aux.data.numpy())
        
        survival = second_part*baseline
        
        return survival

    def predict_risk(self, x, use_log = False):
        """
        Predicting the risk score functions
        
        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
        """        

        # Convert x into the right format
        x = utils.check_data(x)
        
        # Scaling the data
        if self.auto_scaler:
            if x.ndim == 1:
                x = self.scaler.transform( x.reshape(1, -1) )
            elif x.ndim == 2:
                x = self.scaler.transform( x )
        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))
                
        

        # Transforming into pytorch objects
        x = torch.FloatTensor(x)
        

        # Calculating risk_score
        score = self.model(x).data.numpy().flatten()


        return score
    
     
       
    def predict_cumbase(self, score, T, E):
        """
        Predicting the cumulative baseline function
        
        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
        """        

        
        # Calculating risk_score
        score=torch.FloatTensor(score)
        score = torch.reshape( score, (-1, 1) )
        
        
        Risk, Fail,Risk_mod = self.risk_fail_matrix(T, E)
        Risk = torch.FloatTensor(Risk) 
        Fail = torch.FloatTensor(Fail)
        Risk_mod = torch.FloatTensor(Risk_mod)
        
        #calculating h_bar
        Hbar=torch.mm(Risk_mod,score)
        Hbar=torch.reshape( Hbar, (-1, 1) ).data.numpy().flatten()
        Hbar=torch.FloatTensor(Hbar)
        
        #calculating second part
        diff=np.concatenate((np.array(T[0]-0),np.diff(T)),axis=None)
        diff=torch.reshape(torch.FloatTensor(diff),(-1,1)).data.numpy().flatten()
        diff=torch.FloatTensor(diff)
       
        second_part_pre=torch.mul(diff,Hbar)
        
        second_part=torch.cumsum(second_part_pre,0)
       
        
        #calculating first part
        E=torch.FloatTensor(E)
        first_part=torch.mm(torch.reshape( E, (1, -1) ),Risk_mod)
        first_part=torch.reshape(first_part,(-1,1)).data.numpy().flatten()
        first_part=torch.FloatTensor(first_part)
        
        cum_base_pre=first_part-second_part
        
        cum_base_pre=torch.cat((torch.zeros(1),cum_base_pre))
        
        cum_base=np.maximum.accumulate(cum_base_pre)
        
        N=T.shape[0]
        
        cum_base=cum_base[1:(N+1)]
    
        return cum_base


    def __repr__(self):
        """ Representing the class object """

        if self.structure is None:
            super(DeepHaz, self).__repr__()
            return self.name
            
        else:
            S = len(self.structure)
            self.name = self.__class__.__name__
            empty = len(self.name)
            self.name += '( '
            for i, s in enumerate(self.structure):
                n = 'Layer({}): '.format(i+1)
                activation = nn.activation_function(s['activation'], 
                    return_text=True)
                n += 'activation = {}, '.format( s['activation'] )
                n += 'num_units = {} '.format( s['num_units'] )
                
                if i != S-1:
                    self.name += n + '; \n'
                    self.name += empty*' ' + '  '
                else:
                    self.name += n
            self.name += ')'
            return self.name
        
    

