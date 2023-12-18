#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as pl
import os, sys, getopt
from hydroeval import  kge, nse
from sklearn import metrics  
from sklearn.model_selection import ParameterGrid
import gmdhpy
from gmdhpy.gmdh import MultilayerGMDH
from pathlib import Path
import gzip
import pickle


#%%
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

#print ("This is the name of the script: ", program_name)
#print ("Number of arguments: ", len(arguments))
#print ("The arguments are: " , arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0, 5

#%%
from read_data_ankara import *
basename='spi_gmdh_'
datasets = [
            #read_cahora_bassa(look_back=7, look_forward=1),
            read_data_ankara(variation= 3,station='Ankara', test=0.25, expand_features=True, ),
            read_data_ankara(variation= 6,station='Ankara', test=0.25, expand_features=True, ),
            read_data_ankara(variation=12,station='Ankara', test=0.25, expand_features=True, ),            
           ]
#%%
param_grid ={
    #'ref_functions'  : ('linear_cov', 'quadratic', 'cubic', 'linear'),
    'ref_functions'  : (('linear_cov', 'quadratic', 'cubic', 'linear'),),
    'admix_features' : (False,),
    'normalize'      : (False,),
    'alpha'             : (0.001,),# 0.01, 0.05, 0.1, 0.5, 0.9, 1.0),
    #'seq_type'       : ('random',),
    }

pg = ParameterGrid(param_grid)

df_param=pd.DataFrame(pg)
#%%
pd.options.display.float_format = '{:.3f}'.format
for run in range(run0, n_runs):
    random_seed=run+10
    
    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+basename+dr+'/'
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        #os.system('mkdir  '+path)
        
        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            target                          = dataset['target_names'][tk]
            y_train_, y_test_               = dataset['y_train'][tk], dataset['y_test'][tk]
            dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            y_train                         = y_train_
            y_test                          = y_test_
            n_samples_test                  = len(y_test)
            np.random.seed(random_seed)

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
            feature_names=dataset['feature_names']
            feature_names = [f.replace('_','-') for f in feature_names]
                        
            params = {
                'ref_functions': ('linear_cov', 'quadratic', 'cubic', 'linear'),
                #'ref_functions': ('quadratic', ),
                #'criterion_type': 'test_bias',
                'seq_type': 'random',
                'feature_names': feature_names,
                #'min_best_neurons_count':1, 
                'criterion_minimum_width':1,
                'admix_features': True,
                'max_layer_count':30,
                'stop_train_epsilon_condition': 0.0001,
                'layer_err_criterion': 'top',
                #'alpha': 0.5,
                'normalize':False,
                #'l2': 0.1,
                'n_jobs': 1
            }
            
            data_pkl=[]
            for p in pg:
                clf = MultilayerGMDH(**params)
                clf = MultilayerGMDH(**p)
                clf = MultilayerGMDH(ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'), admix_features=True)
                clf.fit(X_train, y_train,)
                #%%
                # y_pred   = clf.predict(X_test)
                # rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
                # r        = sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                # tit      = dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r)
                # print(rmse, r2,r)
            
                # pl.figure(figsize=(16,4)); 
                # #pl.plot([a for a in y_train]+[None for a in y_test]); 
                # pl.plot([None for a in y_train]+[a for a in y_test], 'r-', label='Real data');
                # pl.plot([None for a in y_train]+[a for a in y_pred], 'b-', label='Predicted');
                # pl.legend(); pl.title(tit); pl.xlabel(p)
                # pl.show()
            
                # fig = pl.figure(figsize=[12,4])
                # pl.plot(y_test, 'r-', y_pred,'b-'); pl.legend(['Observed', 'Predicted'])
                # pl.legend(); pl.title(tit); pl.xlabel(p)
                # pl.show()
                    
                # pl.figure(figsize=(6,6)); 
                # pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
                # pl.legend(); pl.title(tit); pl.xlabel(p)
                # pl.show()
                #%%
                y_pred   = clf.predict(X_test)
                y_pred = np.array(y_pred)
                rmse_, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
                r=sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
                nse_=nse(y_test.ravel(), y_pred.ravel())
                #print(rmse_, r2,r, nse_, kge_)
                          
                pl.rc('text', usetex=True)
                pl.rc('font', family='serif',  serif='Times')
                plot=True
                if plot:
                    fig = pl.figure(figsize=[10,4])
                    pl.plot(y_test, 'r-o', y_pred,'b-.o', ms=4); pl.legend(['Observed', 'Predicted'])
                    pl.title(dataset_name+' - '+'\n'+'RMSE = '+str(rmse_)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                    pl.show()
                #
                #fig = pl.figure(figsize=[8,8])
                #pl.plot(y_test,y_test, 'r-', y_test,y_pred,'bo')
                #pl.title(dataset_name+' - '+samples+' - '+strategy+'\n'+'RMSE = '+str(rmse_)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                #pl.show()
                s1 = "%3d: "%run+dataset_name.ljust(15)+' - '+"%0.3f"%rmse_+' - '+"%0.3f"%nse_
                #s1+= ' - '+"%0.3f"%beta
                #s1+= ' - '+ ', '.join(feature_names[ft])+' -- '
                #s1+= ' '.join(["%1.6f"%i for i in model.coef_])+" | %1.3f"%model.intercept_
                #s1+= ' '.join(["%1.6f"%i for i in model[model.steps[-1][0]].coef_])+" | %1.3f"%model[model.steps[-1][0]].intercept_
                print(s1)
                
                #%%
                est_name='gmdh'
                data_pkl.append({'run':run, 'params':p, 'seed':random_seed,
                                 'feature_names':feature_names, 
                                 'estimator':est_name,
                                 'output':target, 'dataset_name':dataset_name,
                                 'selected_indices':clf.get_selected_features(),
                                  'y_test_true':y_test, 'y_test_pred':y_pred,})
                #%%
            ds_name = dataset_name.replace('/','_').replace("'","").lower()
            tg_name = target.replace('/','_').replace("'","").lower()
            #pk=(path+#'_'+
            #pk=(str(path)+#'_'+
            pk=(str(path)+"/"+#'_'+
                basename+'_'+
                '_run_'+str("{:02d}".format(run))+'_'+
                ("%10s"%ds_name         ).rjust(10).replace(' ','_')+#'_'+
                ("%6s"%est_name  ).rjust( 6).replace(' ','_')+#'_'+
                ("%10s"%tg_name         ).rjust(10).replace(' ','_')+#'_'+
                #("%15s"%os.uname()[1]   ).rjust(25).replace(' ','_')+#'_'+
                #time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                '.pkl') 
            pk=pk.replace(' ','_').replace("'","").lower()
            pk=pk.replace('(','_').replace(")","_").lower()
            pk=pk.replace('[','_').replace("]","_").lower()
            #pk=pk.replace('-','_').replace("_","_").lower()
            print(pk)
            pd.DataFrame(data_pkl).to_pickle(pk)
#%%
#with open('cb_gmdh____run_00_____cb_7_1__gmdh_________q.pkl', 'rb') as f:
#    data = pickle.load(f)