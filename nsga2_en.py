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


from sklearn.linear_model import Ridge, LinearRegression, TweedieRegressor, PoissonRegressor, RANSACRegressor, Lasso, ARDRegression, TheilSenRegressor
from sklearn.svm import LinearSVR, SVR

from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error, max_error, make_scorer
from scipy.stats import pearsonr
from hydroeval import nse, kge_c2m
#from util.metrics import kge, agreementindex, rmse, rrmse, kge_non_parametric, mape, mse

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


from platypus import NSGAII, Problem, Real, Integer, Binary, DTLZ2, nondominated


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
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
def split_sequences_multivariate(sequences, n_steps_in, n_steps_out):
    # split a multivariate sequence into samples
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix+1 > len(sequences):
			break
		# gather input and output parts of the pattern
		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)    
#-------------------------------------------------------------------------------
def read_data_xingu(plot=False, kind='lstm', n_steps_in=12):
     
    filename='./data/xingu.csv'
    df= pd.read_csv(filename,  delimiter=';')
    df.index = pd.DatetimeIndex(data=df['data'].values, yearfirst=True)
    df.drop('data', axis=1, inplace=True)
    df.sort_index(inplace=True)  
    df.columns = [i.replace('_','-') for i in df.columns]
  
    if plot:
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif',  serif='Times')
        #df=df[df.index<'2012-12-31']
        pl.figure(figsize=(10,15))
        for i,group in enumerate(df.columns):
            pl.subplot(len(df.columns), 1, i+1)
            df[group].plot(marker='.',fontsize=16,)#pyplot.plot(dataset[group].values)
            pl.title(group, y=0.5, loc='right')
            pl.axvline('2012-12-31', color='k')
            
        pl.show()
        
    target_names=['maxima']
    variable_names=df.columns.drop(target_names)  
    
    X=df[variable_names]
    y=df[target_names]
    
    A,a = split_sequences_multivariate(X.values,n_steps_in,0)
    b=y[n_steps_in-1::].values
    
    n=len(A)-60; 
    X_train, X_test = A[:n], A[n:]    
    y_train, y_test = b[:n], b[n:]
    n_samples, _, n_features = X_train.shape 
         
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train, y_test
        n_samples, n_features = X_train.shape
        variable_names = ['x_'+str("%3.3d"%(i+1)) for i in range(n_features) ]
 

    regression_data =  {
      'task'            : 'forecast',
      'name'            : 'Xingu',
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://sci-hub.se/https://link.springer.com/article/10.1007/s12145-020-00528-8",
      'normalize'       : 'None',
      'date_range'      : None,
      }
    return regression_data
    #%%
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

n_runs=1
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
            
            #print(s)
            #------------------------------------------------------------------
            lb  = [0.0]*n_features  + [1e-6,    0,  ]
            ub  = [1.0]*n_features  + [2e+0,    1,  ]
            #------------------------------------------------------------------ 
            feature_names = dataset['feature_names']
            samples = str(n_samples_train)+'-'+str(n_samples_test)

            for strategy in strategy_list[:1]:
                args=(X_train, y_train, random_seed)
                
                def objective_function(x,*args):
                    X,y,random_seed = args
                    n_samples, n_features=X.shape
                    #print(x)
                    ft = [ i>0.5 for i in  x[:n_features] ]
                    k=0
                    if sum(ft)==0:
                        return 1e12
                    
                    model=ElasticNet(l1_ratio=x[-1], alpha=x[-2],
                                     random_state=random_seed, max_iter=5000)
                    
                    model=Lasso(alpha=x[-2],
                                     random_state=random_seed, max_iter=5000)
                    cv=TimeSeriesSplit(n_splits=10,)
                    r=cross_val_score(model,X[k:,ft], y[k:], cv=cv, n_jobs=1,
                                      scoring=make_scorer(rmse, greater_is_better=False),
                                      #scoring=make_scorer(kge_non_parametric, greater_is_better=True),
                                      #scoring='r2',
                                     )
                    r=-np.mean(r)
                    return r
    
    
                def callback_function(algorithm):
                    print(algorithm.nfe,algorithm.population_size,algorithm.problem.nvars)
    
                class SPI(Problem):
                
                    def __init__(self, lb, ub, *args):
                        super(SPI, self).__init__(len(lb), 2, 0)
                        self.n_vars=len(lb)
                        self.types[:] = [Real(lb[i],ub[i]) for i in range(len(lb))]
                    
                    def evaluate(self, solution):
                        x = [ solution.variables[i] for i in range(self.n_vars)]
                        X,y,random_seed = args
                        n_samples, n_features=X.shape
                        #print(x)
                        ft = [ i>0.5 for i in  x[:n_features] ]
                        k=0
                        sumft=sum(ft)
                        if sumft==0:
                            sumft=1e12
                        
                        model=ElasticNet(l1_ratio=x[-1], alpha=x[-2],
                                         random_state=random_seed, max_iter=5000)
                        
                        model=Lasso(alpha=x[-2],
                                         random_state=random_seed, max_iter=5000)
                        cv=TimeSeriesSplit(n_splits=10,)
                        #print(x)
                        #print(ft)
                        if (sum(ft))==0:
                            return [1e2,1e12]
                        
                        r=cross_val_score(model,X[k:,ft], y[k:], cv=cv, n_jobs=1,
                                          scoring=make_scorer(rmse, greater_is_better=False),
                                          #scoring=make_scorer(kge_non_parametric, greater_is_better=True),
                                          #scoring='r2',
                                         )
                        r=-np.mean(r)
                        solution.objectives[:] = [sumft, r, ]
                
                        
                algorithm = NSGAII(SPI(lb, ub, args))
                pop_size=300; n_gen=50
                algorithm.population_size=pop_size
                algorithm.run(pop_size*n_gen, callback = callback_function)
                
                
                pl.scatter([s.objectives[0] for s in algorithm.result],
                            [s.objectives[1] for s in algorithm.result])
                pl.xlabel("$f_1(x)$")
                pl.ylabel("$f_2(x)$")
                pl.show()

                nondominated_solutions = nondominated(algorithm.result)                
                sol=[]
                for s in nondominated_solutions:                                        
                    z=s.variables                    
                    ft = [ i>0.5 for i in  z[:n_features] ]
                    model=ElasticNet(l1_ratio=z[-1], alpha=z[-2],random_state=random_seed)
                    model=Lasso(alpha=z[-2],random_state=random_seed)
                    y_pred=model.fit(X_train[:,ft], y_train).predict(X_test[:,ft])
                    sol.append({'variables':z, 
                                'ft':ft, 'coef':model.coef_,
                                'intercept': model.intercept_,
                                'mask':''.join([str(int(i)) for i in ft]),
                                'alpha':z[-2], 'y_pred':y_pred,
                                'n':s.objectives[0], 'RMSE':s.objectives[1]})

                sol=pd.DataFrame(sol)
                for i, df in sol.groupby(['n','mask']): 
                    print(i,len(df),df.iloc[0][['mask', 'n', 'alpha','RMSE']])
                    
                    z=df.iloc[0]['variables']
                    ft = [ i>0.5 for i in  z[:n_features] ]
                    model=ElasticNet(l1_ratio=z[-1], alpha=z[-2],random_state=random_seed)
                    model=Lasso(alpha=z[-2],random_state=random_seed)
                    y_pred=model.fit(X_train[:,ft], y_train).predict(X_test[:,ft])

                    
                # for s in nondominated_solutions:                                        
                #     z=s.variables                    
                #     ft = [ i>0.5 for i in  z[:n_features] ]
                #     model=ElasticNet(l1_ratio=z[-1], alpha=z[-2],random_state=random_seed)
                #     model=Lasso(alpha=z[-2],random_state=random_seed)
                #     y_pred=model.fit(X_train[:,ft], y_train).predict(X_test[:,ft])

                    print(feature_names[ft], strategy, samples)
                    print(model.coef_, model.intercept_)
                    #y_pred = df.iloc['y_pred']
                    #y_pred=X_test[:,ft].dot(model.coef_)+model.intercept_
                    l={
                    'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':model.predict(X_train[:,ft]), 
                    'Y_TEST_TRUE':y_test, 'Y_TEST_PRED':y_pred, 'RUN':run,            
                    'EST_PARAMS':{'l1_ratio':z[-1], 'alpha':z[-2]}, 
                    'PARAMS':z, 'ESTIMATOR':model, 'FEATURE_NAMES':feature_names,
                    'SEED':random_seed, 'DATASET_NAME':dataset_name,
                    'ALGO':'DE', 'ALGO_STRATEGY':strategy,
                    'ACTIVE_VAR':ft, 'ACTIVE_VAR_NAMES':feature_names[ft],
                    'MODEL_COEF':model.coef_, 'MODEL_INTERCEPT':model.intercept_,
                    }
                    
                    pk=(path+basename+'_'+("%15s"% dataset_name).rjust(15)+
                        '_run_'+str("{:02d}".format(run))+'_'+
                        '_'+samples+'_'+
                        ("%15s"%target).rjust(15)+'.pkl')
                    pk=pk.replace(' ','_').replace("'","").lower()
                    pk=pk.replace('(','_').replace(")","_").lower()
                    pk=pk.replace('[','_').replace("]","_").lower()
                    pk=pk.replace('-','_').replace("_","_").lower()
                    pd.DataFrame([l]).to_pickle(pk)
                    #%%
                    from hydroeval import  kge, nse
                    y_pred = np.array(y_pred)
                    rmse_, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
                    r=pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                    kge_=kge(y_test.ravel(), y_pred.ravel())[0][0]
                    nse_=nse(y_test.ravel(), y_pred.ravel())
                    print(rmse_, r2,r, nse_, kge_)
                              
                    pl.rc('text', usetex=True)
                    pl.rc('font', family='serif',  serif='Times')
        
                    fig = pl.figure(figsize=[10,4])
                    pl.plot(y_test, 'r-o', y_pred,'b-.o', ms=4); pl.legend(['Observed', 'Predicted'])
                    pl.title(dataset_name+' - '+samples+' - '+strategy+'\n'+'RMSE = '+str(rmse_)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                    pl.show()
                    #
                    #fig = pl.figure(figsize=[8,8])
                    #pl.plot(y_test,y_test, 'r-', y_test,y_pred,'bo')
                    #pl.title(dataset_name+' - '+samples+' - '+strategy+'\n'+'RMSE = '+str(rmse_)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
                    #pl.show()
