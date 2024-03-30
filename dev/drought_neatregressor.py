#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
import os
import sys, getopt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pysr

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    

import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from neuro_evolution import NEATRegressor, WANNRegressor
import visualize
    
#%%

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)


if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0, 4


#%%
basename='neat_ankara__'
pd.options.display.float_format = '{:.3f}'.format

from read_data_ankara import *

for run in range(run0, n_runs):
    random_seed=run+37

    datasets = [
            read_data_ankara(variation= 3,station='Ankara', test=0.25, expand_features=False, ),
            read_data_ankara(variation= 6,station='Ankara', test=0.25, expand_features=False, ),
            read_data_ankara(variation=12,station='Ankara', test=0.25, expand_features=False, ),
           ]     

    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        os.system('mkdir  '+path)
        
        #for (target,y_train,y_test) in zip(dataset['target_names'], dataset['y_train'], dataset['y_test']):                        
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
            #feature_names = [i.replace('_{-0}', '') for i in feature_names]

            
            
            regr = NEATRegressor(
                    number_of_generations=200,
                    fitness_criterion='min',
                    fitness_threshold=0.999,
                    num_hidden=0,
                    pop_size=200,
                    activation_options='sigmoid relu tanh gauss hat clamped identity',
                    activation_mutate_rate=0.1,
                    activation_default='relu',
                    elitism=1,
                    )
            
            
            regr.fit(X_train, y_train)            
            config,winner,stats = (regr.config_, regr.p_.best_genome,regr.stats_)
            
            node_names={(-j-1):i for j,i in enumerate(feature_names)}
            node_names[0]=target
            visualize.draw_net(config, winner, view=True, node_names=node_names, show_disabled=False)
            #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
            visualize.plot_stats(stats, ylog=False, view=True)
            visualize.plot_species(stats, view=True)
#%%
            from hydroeval import  kge
            y_pred = regr.predict(X_test)
            rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
            r=pearsonr(y_test.ravel(), y_pred.ravel())[0] 
            kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
            print(rmse, r2,r)
                        
            fig = pl.figure(figsize=[8,4])
            #pl.plot(y_test, 'r.-', y_pred,'b.-')
            pl.plot(y_test, 'r-', y_pred,'b-')
            pl.title(dataset_name+'\n'+'RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'KGE = '+str(kge_))
            pl.show()
            #
            fig = pl.figure(figsize=[5,5])
            pl.plot(y_test,y_test, 'k-', y_test,y_pred,'b.')
            pl.title(dataset_name+'\n'+'RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
            pl.show()

#%%
