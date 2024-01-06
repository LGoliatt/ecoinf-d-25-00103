#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
import os
import sys, getopt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    

import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
    

#%%

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)


if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0, 1

#%%
        
from gmdhpy.gmdh import MultilayerGMDH

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GMDH(BaseEstimator, RegressorMixin):

    def __init__(self,
                ref_functions= 'linear',
                #criterion_type= 'test_bias',
                seq_type= 'random',
                feature_names= None,
                min_best_neurons_count=1, 
                criterion_minimum_width=1,
                admix_features= False,
                max_layer_count=5,
                stop_train_epsilon_condition= 0.0001,
                layer_err_criterion= 'top',
                alpha= 0.5,
                normalize=False,
                l2= 0.1,
                n_jobs= 1,
                 ):

                self.ref_functions                   = ref_functions                
                #self.criterion_type                  = criterion_type               
                self.seq_type                        = seq_type                     
                self.feature_names                   = feature_names               
                self.min_best_neurons_count          = min_best_neurons_count       
                self.criterion_minimum_width         = criterion_minimum_width      
                self.admix_features                  = admix_features               
                self.max_layer_count                 = max_layer_count              
                self.stop_train_epsilon_condition    = stop_train_epsilon_condition 
                self.layer_err_criterion             = layer_err_criterion          
                self.alpha                           = alpha                        
                self.normalize                       = normalize                    
                self.l2                              = l2                           
                self.n_jobs                          = n_jobs                       
        
                self.params = {
                    'ref_functions'                   : self.ref_functions               ,               
                    #'criterion_type'                  : self.criterion_type              ,               
                    'seq_type'                        : self.seq_type                    ,               
                    'feature_names'                   : self.feature_names               ,               
                    #'min_best_neurons_count'          : self.min_best_neurons_count      ,               
                    'criterion_minimum_width'         : self.criterion_minimum_width     ,               
                    'admix_features'                  : self.admix_features              ,               
                    'max_layer_count'                 : self.max_layer_count             ,               
                    'stop_train_epsilon_condition'    : self.stop_train_epsilon_condition,               
                    'layer_err_criterion'             : self.layer_err_criterion         ,               
                    'alpha'                           : self.alpha                       ,               
                    'normalize'                       : self.normalize                   ,               
                    'alpha'                              : self.l2                          ,               
                    #'n_jobs'                          : self.n_jobs                      ,               
                    }
                self.set_params(**self.params)
                self.estimator = MultilayerGMDH(**self.params)
                

#    def describe(self):
#        """Describe the model"""
#        s = ['*' * 50,
#             'Model',
#             '*' * 50,
#            'Number of layers: {0}'.format(len(self.layers)),
#            'Max possible number of layers: {0}'.format(self.param.max_layer_count),
#            'Model selection criterion: {0}'.format(CriterionType.get_name(self.param.criterion_type)),
#            'Number of features: {0}'.format(self.n_features),
#            'Include features to inputs list for each layer: {0}'.format(self.param.admix_features),
#            'Data size: {0}'.format(self.n_train + self.n_validate),
#            'Train data size: {0}'.format(self.n_train),
#            'Test data size: {0}'.format(self.n_validate),
#            'Selected features by index: {0}'.format(self.get_selected_features_indices()),
#            'Selected features by name: {0}'.format(self.get_selected_features()),
#            'Unselected features by index: {0}'.format(self.get_unselected_features_indices()),
#            'Unselected features by name: {0}'.format(self.get_unselected_features()),
#        ]
#        for layer in self.layers:
#            s.append('\n' + layer.describe(self.feature_names, self.layers))
#        return '\n'.join(s)
        
    
    def fit(self, X, y, verbose=False):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        self.estimator.fit(X, y,)
                
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        self.estimator.feature_names= ['x'+str(i) for i in range(X.shape[1])]
        y = self.estimator.predict(X)
            
        return y

    #def get_params(self, deep=True):
    #    # suppose this estimator has parameters "alpha" and "recursive"
    #    return {"param1": self.param1, "param2": self.param2}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
#%%
basename='fbhp_gmdh'
pd.options.display.float_format = '{:.3f}'.format


from read_data_ankara import *

pd.options.display.float_format = '{:.3f}'.format
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
            feature_names = [i.replace('-','').replace(' ','') for i in feature_names]
            #feature_names = [i.replace('_{-0}', '') for i in feature_names]

            from sklearn.ensemble import AdaBoostRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import Ridge
            from sklearn.neural_network import MLPRegressor
            from sklearn.neighbors import KNeighborsRegressor
                    

            regr = GMDH(
                ref_functions=('linear','linear_cov', 'quadratic', 'cubic'), 
                feature_names=feature_names,
                admix_features=True,
                max_layer_count=20,
                #criterion_type='validate',
                normalize=False,
                #alpha=0,
                )
            regr.fit(X_train, y_train)
            
            from gmdhpy.plot_gmdh import PlotGMDH as PlotModel
            model=regr.estimator
            #PlotModel(model, filename='model_house_model', plot_neuron_name=True, view=True).plot()
            PlotModel(model, filename=basename, view=True)#.plot()           
            #%%
        
            expression=[]
            import sympy as sp
            sp.init_printing(forecolor='Yellow')
        
            print('\n'*10+'='*80+'\n GMDH Expression\n'+'='*80+'\n'*2)
            s=''
            sp.var(list(feature_names))
            for layer in model.layers:
                p=layer
                print(p)
                print('-'*80,'\n')
                s+='# '+str(p.layer_index)+'\n'
                for neuron in layer: 
                    print(
                          neuron,"|",
                          #neuron.neuron_index, "|",
                          neuron.fw_size,"|",
                          neuron.ftype, "|",
                          neuron.u1_index, "|",
                          neuron.u2_index,"|",
                          )
                    inp=[None,None]
                    inp[0]=neuron.get_features_name(neuron.u1_index,)# feature_names, model.layers)
                    inp[1]=neuron.get_features_name(neuron.u2_index,)# feature_names, model.layers)
                    x=inp
                    #x=[None,None]
                    #x[0]=get_features_name(neuron,neuron.u1_index,)# feature_names, model.layers)
                    #x[1]=get_features_name(neuron,neuron.u2_index,)# feature_names, model.layers)
            
                    x=[i.split('=')[-1].split(',')[-1].replace(' ','') for i in inp]
                    print(x)
                    w=neuron.w
                    sp.var(['w'+str(i) for i in range(len(w))])
                    coeff=';'.join(['w'+str(i)+'='+str(w[i]) for i in range(len(w))])
                    coeff_dict={'w'+str(i):w[i] for i in range(len(w))}
                    print(w)
                    expr=neuron.get_short_name()
                    print(expr)
                    prev='prev_layer_model_'+str(neuron.model_index)
                    print('-'*4,end='\n')
                    
                    sp.var('u1,u2')
                    
                    ftype=neuron.__str__().split(' - ')[1]
                    s+='u1='+str(x[0])+';'+'u2='+str(x[1])+';\n'
                    prev=neuron.transfer(u1,u2,w)
                    #prev=neuron.transfer(sp.var(x[0]),sp.var(x[1]),w)
                    s+='prev_layer_model_'+str(neuron.model_index)+'='+str(prev)+'\n'
                   
                   
            s+='\noutput = prev_layer_model_0;\n'
            s+='#display(output); \n'
            print(s)
            with open('output.py', 'w') as file:
                file.write(s)    
            #%%
            sp.var(list(feature_names))
            exec(open("output.py").read())
            print(output)
            #sp.parsing.sympy_parser.parse_expr(s)
            y_pred = model.predict(X_test)
            for i in range(len(X_test)):
                d=dict(zip(feature_names,X_test[i]))
                print(y_test[i],y_pred[i], output.subs(d))
            #%%
            from hydroeval import  kge
            y_pred = regr.predict(X_test)
            rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
            r=pearsonr(y_test.ravel(), y_pred.ravel())[0] 
            kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
            print(rmse, r2,r)
                        
            # fig = pl.figure(figsize=[8,4])
            # #pl.plot(y_test, 'r.-', y_pred,'b.-')
            # pl.plot(y_test, 'r-', y_pred,'b-')
            # pl.title(dataset_name+'\n'+'RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'KGE = '+str(kge_))
            # pl.show()
            #
            fig = pl.figure(figsize=[5,5])
            pl.plot(y_test,y_test, 'k-', y_test,y_pred,'b.')
            pl.title(dataset_name+'\n'+'RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
            pl.show()

#%%
