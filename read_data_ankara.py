#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import glob as gl
import pylab as pl
import pandas as pd
import os


def read_data_ankara(variation=12, station='Ankara', test=0.25, plot=False, 
                    expand_features=False, 
                     ):
     
    filename='./data/data_ankara/SPI'+str(variation)+'.xlsx'
    data = pd.read_excel(filename, index_col=None, )
    data.columns = [ ' '.join([i.capitalize() for i in d.split(' ')]) for d in data.columns]
    var_to_plot=data.columns
    df=data[var_to_plot]
    k=int(data.shape[0]*(1-test))
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

    if expand_features:
        for i in range(len(variable_names)):
            for j in range(i,len(variable_names)):
                a, b = variable_names[i], variable_names[j]
                if a!=b:
                    s=str(a)+str(' x ')+str(b)
                    X[s] = X[a]*X[b]
                    s=str(a)+str(' / ')+str(b)
                    X[s] = X[a]/X[b]
                    s=str(b)+str(' / ')+str(a)
                    X[s] = X[b]/X[a]
                
        variable_names = X.columns
       
        
    n=k; 
    X_train, X_test = X.iloc[:n].values, X.iloc[n:].values    
    y_train, y_test = y.iloc[:n].values, y.iloc[n:].values    
    n_samples, n_features = X_train.shape 
         
    regression_data =  {
      'task'            : 'forecast',
      'name'            : station+' SPI-'+str(variation),
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
      'reference'       : "",
      'normalize'       : 'None',
      'date_range'      : None,
      'reference'       : 'https://doi.org/10.1016/j.cageo.2020.104622',
      }
    return regression_data
    #%%
    
    
if __name__ == "__main__":
    datasets = [                 
            read_data_ankara(variation= 3,station='Ankara', test=0.25, expand_features=True, ),
            read_data_ankara(variation= 6,station='Ankara', test=0.25, expand_features=True, ),
            read_data_ankara(variation=12,station='Ankara', test=0.25, expand_features=True, ),
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
    
