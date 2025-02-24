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
     
    """
Reads and processes data from an Excel file for a specified station and variation.

    This method reads data from an Excel file, processes it by renaming columns, and optionally plots the training and test data. 
    It also allows for feature expansion by creating interaction terms between features. The processed data is then split into training 
    and test sets, which are returned in a structured format.

    Args:
        variation (int, optional): The variation number to specify which dataset to read. Defaults to 12.
        station (str, optional): The name of the station for which data is being read. Defaults to 'Ankara'.
        test (float, optional): The proportion of data to be used for testing. Defaults to 0.25.
        plot (bool, optional): If True, generates plots for training and test data. Defaults to False.
        expand_features (bool, optional): If True, creates additional features by multiplying and dividing existing features. Defaults to False.

    Returns:
        dict: A dictionary containing the processed regression data, including training and test sets, feature names, and other metadata.
    """
     
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
        pl.figure(figsize=(7,9))
        for i,group in enumerate(df.columns):
            pl.subplot(len(df.columns), 1, i+1)
            df[group].iloc[id0].plot(marker='', label='Training')#,fontsize=16,)#pyplot.plot(dataset[group].values)
            df[group].iloc[id1].plot(marker='', label='Test')#,fontsize=16,)#pyplot.plot(dataset[group].values)
            pl.axvline(k, color='grey', ls='-.')
            pl.xlabel('Year')
            #pl.legend()#(loc=(1.01,0.5))
            pl.ylabel(group)

        pl.savefig('dataset_'+str(variation)+'.png', transparent=True,  bbox_inches='tight', dpi=300)
        pl.show()

    target_names=[station]
    variable_names=data.columns.drop(target_names)  
    
    X=data[variable_names]
    y=data[target_names]

    var_names=variable_names
    if expand_features:
            for i in range(len(var_names)):
                a = var_names[i]
               
                for j in range(i,len(var_names)):
                    b = var_names[j]
                    if a!=b:
                        s=str(a)+str(' * ')+str(b)
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
            read_data_ankara(variation= 3,station='Ankara', test=0.25, expand_features=False, plot=True, ),
            read_data_ankara(variation= 6,station='Ankara', test=0.25, expand_features=False, plot=True, ),
            read_data_ankara(variation=12,station='Ankara', test=0.25, expand_features=False, plot=True, ),
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
    
