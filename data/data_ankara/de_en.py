#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import glob as gl
import pylab as pl
import pandas as pd
import os
from scipy.optimize import differential_evolution as de
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error, max_error
from scipy.stats import pearsonr

basename='ankara__'
#-------------------------------------------------------------------------------

#%%
def read_data_ankara(variation=12, station='Ankara', multivariate=True,
                      kind='ml', plot=False):
     
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

n_runs      = 1
for run in range(0, n_runs):
    random_seed=run+1
    
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
            lb  = [0.0]*n_features  + [1e-6,    0,  ]
            ub  = [1.0]*n_features  + [2e+1,    1,  ]
            #------------------------------------------------------------------ 
            
            args=(X_train, y_train, random_seed)
            
            def objective_function(x,*args):
                X,y,random_seed = args
                n_samples, n_features=X.shape
                ft = [ i>0.5 for i in  x[:n_features] ]
                if sum(ft)==0:
                    return 1e12
                model=ElasticNet(l1_ratio=x[-1], alpha=x[-2],random_state=random_seed)
                cv=TimeSeriesSplit(n_splits=3,)
                r=cross_val_score(model,X[:,ft], y, cv=cv, n_jobs=1, 
                                  scoring='neg_root_mean_squared_error')
                r=np.abs(r).mean()
                return r
            
            res = de(objective_function, tuple(zip(lb,ub)), args=args,
                         strategy='randtobest1exp',
                         popsize=10, maxiter=100, tol=1e-8,  
                         mutation=(0.5, 1),  recombination=0.7,
                         disp=True,
                         init='latinhypercube',
                         seed=random_seed)
            
            z=res['x']
            ft = [ i>0.5 for i in  z[:n_features] ]
            model=ElasticNet(l1_ratio=z[-1], alpha=z[-2],random_state=random_seed)
            
            y_pred=model.fit(X_train, y_train).predict(X_test)
            #y_pred=(model.intercept_ + X_test*model.coef_).sum(axis=1)
            #%%
            from hydroeval import  kge, nse
            y_pred = np.array(y_pred)
            rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
            r=pearsonr(y_test.ravel(), y_pred.ravel())[0] 
            kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
            nse_=nse(y_test.ravel(), y_pred.ravel())
            print(rmse, r2,r)
                        
            fig = pl.figure(figsize=[8,4])
            pl.plot(y_test, 'r-', y_pred,'b-')
            pl.title(dataset_name+'\n'+'RMSE = '+str(rmse)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
            pl.show()
            #
            fig = pl.figure(figsize=[8,8])
            pl.plot(y_test,y_test, 'r-', y_test,y_pred,'bo')
            pl.title(dataset_name+'\n'+'RMSE = '+str(rmse)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
            pl.show()
