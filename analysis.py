#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import pandas as pd
from pandas.core.base import DataError
import matplotlib.pyplot as pl
import scipy as sp
import seaborn as sns
import re, os, sys, glob, math, itertools
import hydroeval as he
from scipy import stats

pd.options.display.float_format = '{:.3f}'.format
palette_color="Set1"#"Blues_r"

paths = glob.glob('./pkl_ankara*')

pkl_list  = []
for path in paths:    
  for (k,p) in enumerate(glob.glob(path)):
    pkl_list += glob.glob(p+'/'+'*.pkl')

#%%
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error
from util.metrics import rrmse, rmse, agreementindex,  lognashsutcliffe,  nashsutcliffe, vaf, kge


def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e0):
        return '%1.3f' % x   
      else:
        return '%1.2f' % x #return '%1.3f' % x
  
def fstat(x):
  #m,s= '{:1.4g}'.format(np.mean(x)), '{:1.4g}'.format(np.std(x))
  #m,s, md= fmt(np.mean(x)), fmt(np.std(x)), fmt(np.median(x)) 
  m,s, md= np.mean(x), np.std(x), np.median(x) 
  #text=str(m)+'$\pm$'+str(s)
  s = '--' if s<1e-8 else s
  text=fmt(m)+' ('+fmt(s)+')'#+' ['+str(md)+']'
  return text
  
#%%

pkl_list.sort()
A=[]
for pkl in pkl_list:
    df = pd.read_pickle(pkl)       
    A.append(df)

A = pd.concat(A, sort=False)
#%%
steps=['TRAIN', 'TEST'] if 'Y_TEST_PRED' in A.columns else ['TRAIN']

C = []
for step in steps:
    for k in range(len(A)):
        df=A.iloc[k]
        y_true = df['Y_'+step+'_TRUE']
        y_pred = df['Y_'+step+'_PRED']
        run = df['RUN']
        av = df['ACTIVE_VAR']
        ds_name = df['DATASET_NAME']
        coeff = df['MODEL_COEF']
        intercept = df['MODEL_INTERCEPT']
        active_features = df['ACTIVE_VAR']
        active_stations = df['ACTIVE_VAR_NAMES']
        idx_var = df['ACTIVE_VAR']
        feature_names=df['FEATURE_NAMES']
        estimator_name=type(df['ESTIMATOR']).__name__
        
        if len(y_true)>0:
            _mape    = abs((y_true - y_pred)/y_true).mean()*100
            _vaf     = vaf(y_true, y_pred)
            _r2      = r2_score(y_true, y_pred)
            _mae     = mean_absolute_error(y_true, y_pred)
            _mse     = mean_squared_error(y_true, y_pred)
            _rrmse   = rrmse(y_true, y_pred)
            _wi      = agreementindex(y_true, y_pred)
            _r       = stats.pearsonr(y_true, y_pred)[0]
            _nse     = nashsutcliffe(y_true, y_pred)
            _rmse    = he.rmse(y_true, y_pred)
            #_kge     = he.kge(y_true, y_pred)[0][0]
            _kge     = kge(y_true, y_pred)
            _mare    = he.mare(y_true, y_pred)
            n_var    = sum(df['ACTIVE_VAR'])
            n_feat   = len(df['FEATURE_NAMES'])
            #mxl      = 
            dic      = {'Run':run, 'Output':ds_name.split(' ')[1], 'MAPE':_mape,
                      'R$^2$':_r2, 'MSE':_mse,'Seed':df['SEED'], 
                      'Active Features':df['ACTIVE_VAR_NAMES'], 
                      'Dataset':ds_name, 'Phase':step, 'SI':None,
                      'NSE': _nse, 'MARE': _mare, 'MAE': _mae, 'VAF': _vaf, 
                      'Active Variables': ', '.join(df['ACTIVE_VAR_NAMES']),
                      'KGE': _kge,
                      'n_var':n_var, 'Estimator':estimator_name,
                      'Feature Names':feature_names,
                      'n_features':n_feat,
                      'RMSE':_rmse, 'R':_r, 'Parameters':df['EST_PARAMS'],
                      'NDEI':_rmse/np.std(y_true),
                      'WI':_wi, 'RRMSE':_rrmse,
                      'y_true':y_true.ravel(), 
                      'y_pred':y_pred.ravel(),
                      'Optimizer':df['ALGO'].split(':')[0],
                      'Estimator':None,
                      'Coefficients':coeff, 'Intercept':intercept,
                      'Active Indices':idx_var,
                      }
            C.append(dic)
     
C = pd.DataFrame(C)
C = C.reindex(sorted(C.columns), axis=1)
#%%   
metrics =  ['MAE', 'MAPE', 'RMSE', 'RRMSE', 'NSE', 'VAF', 'R', 'KGE', 'WI']    
metrics_max =  ['NSE', 'VAF', 'R', 'KGE', 'WI']    
   
aux=[]
for (f,d,o,), df in C.groupby(['Phase', 'Dataset', 'Output',]):
    print(d,f,o,len(df))
    dic={}
    dic['Dataset']=d
    dic['Phase']=f
    dic['Output']=o
    for f in metrics:
        dic[f]= fstat(df[f])
    
    aux.append(dic)
    
tbl = pd.DataFrame(aux)
print(tbl[['Dataset', 'Phase', 'NSE', 'RMSE', 'WI']])

#%%
S=[]   
for (i,j), df in C.groupby(['Dataset','Active Variables']): 
    #S.append({' Dataset':i, 'Active Variables':j})
    #print('\t',i,'\t','\t',j)
    
    B=[]
    for i in range(len(df)):
        aux=df.iloc[i]
        coeff = aux['Coefficients'] [aux['Active Indices'] ]
        cols = aux['Active Features']
        d=dict(zip(cols, coeff))
        d['Intercept'] = aux['Intercept']
        B.append(d)
        
    B=pd.DataFrame(B)
    print('\n\n', i, j,'\n\n', B.mean(), B.std())
    
#%%