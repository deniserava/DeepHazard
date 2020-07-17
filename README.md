# DeepHazard
DeepHazard: Neural Network for Time Varying Risks

Title: "DeepHazard: Neural Network for Time Varying Risks"

Authors: Denise Rava, Jelena Bradic

Paper: 

This code implements DeepHazard: a Neural Network method for survival data for survival function estimation. DeepHazard allows time-varying covariates. Details on the method can be found in 'paper'.

### Details
* `DeepHazConst()`: fits DeepHazard on survival Data with constant covariates returning predicted survival functions for any observations in the test set and concordance index as measure of performance.
* ` DeepHazTime(()`: fits DeepHazard on survival Data with time-varying covariates returning predicted survival functions for any observations in the test set and time dependent concorance index from Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. 'A timedependent discrimination index for survival data. Statistics in Medicine 24:3927–3944' as measure of performance.

### Example
sample.py in the Sample folder show two examples, one applied on survival data with time constant covariates and one with time varying covariates. Datasets for both examples are provided in the Sample folder. In both examples DeepHazard is trained on the train dataset and used for predicting survival function on the test dataset. Concordance index is then computed as measure of performance. 
```python
train=pd.read_csv('sampletrainconst.csv',delimiter=',') #import data
test=pd.read_csv('sampletestconst.csv',delimiter=',')
structure = [{'activation': 'Relu','num_units':10,'dropout':0.2},{'activation': 'Relu','num_units':10,'dropout':0.2}] #define structure
deephaz,Surv,C_index=dh.DeepHazConst(train,test,l2c=1e-5,lrc=2e-1,structure=structure,init_method='he_uniform',optimizer='adam',num_epochs=1000,early_stopping=1e-5,penal='Ridge') #apply method
'''
```python
train=pd.read_csv('sampletraintimevar.csv',delimiter=',') #import data
test=pd.read_csv('sampletesttimevar.csv',delimiter=',')
inter=np.array((0,0.2,0.4,0.6,100)) #set interval
structure = [{'activation': 'Relu','num_units':10,'dropout':0.2},{'activation': 'Relu','num_units':10,'dropout':0.2}] #define structure of network
deepHazlis,Surv,C_index=dh.DeepHazTime(train=train,test=test,inter=inter,Ncol=3,l2c=1e-5,lrc=2e-1,structure=structure,init_method='he_uniform',optimizer='adam',num_epochs=1000,early_stopping=1e-5,penal='Ridge') #apply DeepHazard
'''
Additional details, comments and explanations can be found in sample.py. 

