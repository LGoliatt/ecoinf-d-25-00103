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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge, LinearRegression, TweedieRegressor, PoissonRegressor, RANSACRegressor, Lasso, ARDRegression, TheilSenRegressor
from sklearn.svm import LinearSVR, SVR

from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error, max_error, make_scorer
from scipy.stats import pearsonr
from hydroeval import  kge, nse

from read_data_ankara import *

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


basename='ankara__'
#-------------------------------------------------------------------------------

def lhsu(xmin,xmax,nsample):
   nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
   for j in range(nvar):
       idx=np.random.permutation(nsample)
       P =(idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
       
   return s

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#-------------------------------------------------------------------------------
datasets = [           
            read_data_ankara(variation= 3,station='Ankara', test=0.25, expand_features=True, ),
            read_data_ankara(variation= 6,station='Ankara', test=0.25, expand_features=True, ),
            read_data_ankara(variation=12,station='Ankara', test=0.25, expand_features=True, ),           
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

plot=True
n_runs=50
for run in range(0, n_runs):
    random_seed=run+10
    
    for dataset in datasets:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        os.system('mkdir '+path.replace("-","_").lower())
        
        for tk, tn in enumerate(dataset['target_names']):
            #print (tk, tn)
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
            
            #print(s)
            #------------------------------------------------------------------
            lb  = [0.0]*n_features  + [   1e-6,    0,]
            ub  = [1.0]*n_features  + [   2e+0,    1,]           
            #------------------------------------------------------------------ 
            feature_names = dataset['feature_names']
            samples = str(n_samples_train)+'-'+str(n_samples_test)

            for strategy in strategy_list[:1]:
              for beta in [0.0,0.1, 0.5, 1.0, 1.5, 2.0]:
                args=(X_train, y_train, random_seed, beta)
                
                def objective_function(x,*args):
                    X,y,random_seed,beta = args
                    n_samples, n_features=X.shape
                    ft = [ i>0.5 for i in  x[:n_features] ]
                    k=0
                    if sum(ft)==0:
                        return 1e12
                    
                    model=Lasso(alpha=x[-2],
                                     random_state=random_seed, max_iter=5000)
                    cv=TimeSeriesSplit(n_splits=10,)
                    r=cross_val_score(model,X[k:,ft], y[k:], cv=cv, n_jobs=1,
                                      scoring=make_scorer(rmse, greater_is_better=False),                                     
                                     )
                   r=-np.mean(r) * (1 + beta*sum(ft)/n_features) # modulating model complexity
                    return r
    
                np.random.seed(random_seed)
                init=lhsu(lb,ub,25)            
                res = de(objective_function, tuple(zip(lb,ub)), args=args,
                             strategy=strategy,
                             init=init, maxiter=200, tol=1e-5,  
                             mutation=0.8,  recombination=0.9, 
                             disp=False, polish=False,
                             seed=random_seed)
                
                z=res['x']
                ft = [ i>0.5 for i in  z[:n_features] ]
                model=ElasticNet(l1_ratio=z[-1], alpha=z[-2],random_state=random_seed)
                model=Lasso(alpha=z[-2],random_state=random_seed)
                
                y_pred=model.fit(X_train[:,ft], y_train).predict(X_test[:,ft])
                #%%
                y_pred = np.array(y_pred)
                rmse_, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
                r=pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
                nse_=nse(y_test.ravel(), y_pred.ravel())
                #print(rmse_, r2,r, nse_, kge_)
                          
                pl.rc('text', usetex=True)
                pl.rc('font', family='serif',  serif='Times')
                if plot:
                    fig = pl.figure(figsize=[10,4])
                    pl.plot(y_test, 'r-o', y_pred,'b-.o', ms=4); pl.legend(['Observed', 'Predicted'])
                    pl.title(dataset_name+' - '+samples+' - '+strategy+'\n'+'RMSE = '+str(rmse_)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                    pl.show()
                #
                s1 = "%3d: "%run+dataset_name.ljust(15)+' - '+"%0.3f"%rmse_+' - '+"%0.3f"%nse_
                s1+= ' >> '+"%0.3f"%beta
                s1+= ' | '+ ', '.join(feature_names[ft])+' -- '
                s1+= ' '.join(["%1.6f"%i for i in model.coef_])+" | %1.3f"%model.intercept_
                print(s1)
#%%
                l={
                'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':model.predict(X_train[:,ft]), 
                'Y_TEST_TRUE':y_test, 'Y_TEST_PRED':y_pred, 'RUN':run,            
                'EST_PARAMS':{'l1_ratio':z[-1], 'alpha':z[-2]}, 
                'PARAMS':z, 'ESTIMATOR':model, 'FEATURE_NAMES':feature_names,
                'SEED':random_seed, 'DATASET_NAME':dataset_name,
                'ALGO':'DE', 'ALGO_STRATEGY':strategy,
                'ACTIVE_VAR':ft, 'ACTIVE_VAR_NAMES':feature_names[ft],
                'MODEL_COEF':model.coef_, 'MODEL_INTERCEPT':model.intercept_,
                'BETA':beta,
                }
                
                pk=(path+basename+'_'+("%15s"% dataset_name).rjust(15)+
                    '_run_'+str("{:02d}".format(run))+'_'+
                    '_'+samples+'_'+
                    '_'+'beta'+str("%1.2f"%beta).replace('.','p')+'_'+
                    ("%15s"%target).rjust(15)+'.pkl')
                pk=pk.replace(' ','_').replace("'","").lower()
                pk=pk.replace('(','_').replace(")","_").lower()
                pk=pk.replace('[','_').replace("]","_").lower()
                pk=pk.replace('-','_').replace("-","_").lower()
                pd.DataFrame([l]).to_pickle(pk)
#%%
