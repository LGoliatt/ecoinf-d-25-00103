#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import glob as gl
import pylab as pl
import pandas as pd
import os
from scipy.optimize import differential_evolution as de
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


from sklearn.linear_model import Ridge, TweedieRegressor, PoissonRegressor, RANSACRegressor, Lasso, ARDRegression
from sklearn.svm import LinearSVR, SVR

from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error, max_error
from scipy.stats import pearsonr

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


basename='ankara__'
#-------------------------------------------------------------------------------

#%%
def lhsu(xmin,xmax,nsample):
   nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
   for j in range(nvar):
       idx=np.random.permutation(nsample)
       P =(idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
       
   return s
#%%
def read_data_ankara(variation=12, station='Ankara', plot=False):
     
    filename='./data/SPI'+str(variation)+'.xlsx'
    data = pd.read_excel(filename, index_col=None, )
    data.columns = [ ' '.join([i.capitalize() for i in d.split(' ')]) for d in data.columns]
    var_to_plot=data.columns
    df=data[var_to_plot]
    k=int(data.shape[0]*.75)
    id0=df.index <= k
    id1=df.index > k
    if plot:
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif',  serif='Times')
        pl.figure(figsize=(6,4))
        for i,group in enumerate(df.columns):
            pl.subplot(len(df.columns), 1, i+1)
            df[group].iloc[id0].plot(marker='', label='Training')#,fontsize=16,)#pyplot.plot(dataset[group].values)
            df[group].iloc[id1].plot(marker='', label='Test')#,fontsize=16,)#pyplot.plot(dataset[group].values)
            pl.axvline(k, color='grey', ls='-.')
            pl.xlabel('Year')
            #pl.legend()#(loc=(1.01,0.5))
            pl.ylabel(group)
            
        pl.show()
        
    target_names=[station]
    variable_names=data.columns.drop(target_names)  
    
    X=data[variable_names]
    y=data[target_names]
    
    n=k; 
    X_train, X_test = X.iloc[:n].values, X.iloc[n:].values    
    y_train, y_test = y.iloc[:n].values, y.iloc[n:].values    
    n_samples, n_features = X_train.shape 
         
    regression_data =  {
      'task'            : 'forecast',
      'name'            : station+' SPI'+str(variation),
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.reshape(1,-1),
      'y_test'          : y_test.reshape(1,-1),      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://doi.org/10.1016/j.ins.2021.02.011",
      'normalize'       : 'None',
      'date_range'      : None,
      }
    return regression_data
    #%%
#-------------------------------------------------------------------------------
datasets = [
            read_data_ankara(variation= 3,station='Ankara'),
            read_data_ankara(variation= 6,station='Ankara'),
            read_data_ankara(variation=12,station='Ankara'),
           ]     

#%%----------------------------------------------------------------------------   
pd.options.display.float_format = '{:.3f}'.format

strategy_list=[
             'best1bin',
             'best1exp',
             'rand1exp',
             'randtobest1exp',
             'currenttobest1exp',
             'best2exp',
             'rand2exp',
             'randtobest1bin',
             'currenttobest1bin',
             'best2bin',
             'rand2bin',
             'rand1bin', 
         ]

n_runs=30
for run in range(0, n_runs):
    random_seed=run+1000
    
    for dataset in datasets:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        os.system('mkdir  '+path)
        
        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            dataset_name = dataset['name']
            target                          = dataset['target_names'][tk]
            y_train, y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
            dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            n_samples_test                  = len(y_test)
            
            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Normalization              : '+str(normalize)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            s+='Reference                  : '+str(dataset['reference'])+'\n'
            s+='='*80
            s+='\n'            
            
            print(s)
            #------------------------------------------------------------------
            lb  = [0.0]*n_features  + [1e-6,  1e-6,    0,  ]
            ub  = [1.0]*n_features  + [1e+2,  2e+3,    1,  ]
            #------------------------------------------------------------------ 
            feature_names = dataset['feature_names']

            for strategy in strategy_list[:1]:
                args=(X_train, y_train, random_seed)
                
                def objective_function(x,*args):
                    X,y,random_seed = args
                    n_samples, n_features=X.shape
                    ft = [ i>0.5 for i in  x[:n_features] ]
                    if sum(ft)==0:
                        return 1e12                              
                    model=SVR(C=x[-2], epsilon=x[-1], gamma=x[-3],
                              tol=1e-6, kernel='poly',
                              max_iter=5000)                    
                    cv=TimeSeriesSplit(n_splits=5,)
                    r=cross_val_score(model,X[:,ft], y, cv=cv, n_jobs=1, 
                                      scoring='neg_root_mean_squared_error')
                    r=np.abs(r).mean()
                    return r
    
                np.random.seed(random_seed)
                init=lhsu(lb,ub,20)            
                res = de(objective_function, tuple(zip(lb,ub)), args=args,
                             strategy=strategy,
                             init=init, maxiter=30, tol=1e-8,  
                             mutation=0.8,  recombination=0.9, 
                             disp=True, 
                             seed=random_seed)
                
                z=res['x']
                ft = [ i>0.5 for i in  z[:n_features] ]
                model=ElasticNet(l1_ratio=z[-1], alpha=z[-2],random_state=random_seed)
                
                y_pred=model.fit(X_train, y_train).predict(X_test)
                print(feature_names[ft], strategy)
                #y_pred=X_test[:,ft].dot(model.coef_[ft])+model.intercept_
                l={
                'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':model.predict(X_train), 
                'Y_TEST_TRUE':y_test, 'Y_TEST_PRED':y_pred, 'RUN':run,            
                'EST_PARAMS':{'l1_ratio':z[-1], 'alpha':z[-2]}, 
                'PARAMS':z, 'ESTIMATOR':model, 'FEATURE_NAMES':feature_names,
                'SEED':random_seed, 'DATASET_NAME':dataset_name,
                'ALGO':'DE', 'ALGO_STRATEGY':strategy,
                'ACTIVE_VAR':ft, 'ACTIVE_VAR_NAMES':feature_names[ft],
                'MODEL_COEF':model.coef_, 'MODEL_INTERCEPT':model.intercept_,
                }
                
                pk=(path+basename+'_'+("%15s"% dataset_name).rjust(25)+
                    '_run_'+str("{:02d}".format(run))+'_'+
                    ("%15s"%target).rjust(25)+'.pkl')
                pk=pk.replace(' ','_').replace("'","").lower()
                pk=pk.replace('(','_').replace(")","_").lower()
                pk=pk.replace('[','_').replace("]","_").lower()
                pk=pk.replace('-','_').replace("_","_").lower()
                pd.DataFrame([l]).to_pickle(pk)
                #%%
                from hydroeval import  kge, nse
                y_pred = np.array(y_pred)
                rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
                r=pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
                nse_=nse(y_test.ravel(), y_pred.ravel())
                print(rmse, r2,r, nse_, kge_)
                            
                fig = pl.figure(figsize=[8,4])
                pl.plot(y_test, 'r-', y_pred,'b-')
                pl.title(dataset_name+' - '+strategy+'\n'+'RMSE = '+str(rmse)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                pl.show()
                #
                fig = pl.figure(figsize=[8,8])
                pl.plot(y_test,y_test, 'r-', y_test,y_pred,'bo')
                pl.title(dataset_name+' - '+strategy+'\n'+'RMSE = '+str(rmse)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                pl.show()
