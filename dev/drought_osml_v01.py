#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
import os
import sys, getopt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns; sns.set()

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from gmdhpy.gmdh import MultilayerGMDH as GMDHRegressor


import pysr
from neuro_evolution import NEATRegressor
    
from skopt import BayesSearchCV

#%%


def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e0):
        return '%1.3f' % x   
      else:
        return '%1.3f' % x #return '%1.3f' % x
  
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
    run0, n_runs = 0, 30

#%%    
basename='drought_osml_'

look_back=1
look_forward=48
  

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import  XGBRegressor
from sklearn.linear_model import ElasticNetCV, Ridge, PassiveAggressiveRegressor, LinearRegression, BayesianRidge, OrthogonalMatchingPursuit
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVR
from pyearth import Earth as MARS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


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
        path='./json_'+dr+'/'
        path=path.replace('-','_').replace(" ","_").lower()
        os.system('mkdir -p '+path)
        
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
            
                        
            numLags = 1
            predictionStep = 1
            nDimInput = X_train.shape[1]
            nDimOutput = 1
            numNeurons = 25
            numCols = n_features
            LN=True
            InWeightFF=0.97
            OutWeightFF=0.92
            HiddenWeightFF=1.0
            AE=True
            lamb=0.0001

            # net = OSELM(inputs=numLags*numCols, outputs=nDimOutput, 
            #     numHiddenNeurons=20, 
            #     forgettingFactor=0.95,
            #     )
            # net.initializePhase(lamb=1e5)
            

            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            from sklearn.linear_model import PassiveAggressiveRegressor, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, ElasticNetCV, RANSACRegressor, GammaRegressor, RidgeCV
            reg_list=[
                #('LR'  , LinearRegression()),
                #('LS'  , LassoCV()),
                #('LL'  , LassoLarsCV()),
                #('RD'  , RidgeCV()),
                #('EN'  , ElasticNetCV()),
                #('NLR' , Pipeline([('poly', PolynomialFeatures(degree=2)), ('lr', LinearRegression())])), 
                #('NLR' , Pipeline([('poly', PolynomialFeatures(degree=1)), ('gb', GradientBoostingRegressor(verbose=0))])), 
                #('OMP' , OrthogonalMatchingPursuit()), 
                #('RF' , RandomForestRegressor(n_estimators=100)), 
                #('GB'  , GradientBoostingRegressor(verbose=0, n_estimators=100)),
                #('MARS', MARS(max_degree=2, penalty=1e4)),
                #('GMDH', GMDHRegressor(ref_functions=('linear','linear_cov', 'quadratic', 'cubic'), admix_features=True, max_layer_count=10)),
            ]

            reg_sr = pysr.PySRRegressor(
                niterations=20,  # < Increase me for better results
                binary_operators=["+", "*"],
                unary_operators=[
                    'neg', 
                    'square', 'cube', 
                    #'exp', 'abs', 
                    #'sqrt', 
                    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 
                    #'atan', 'asinh', 'acosh', 
                    #'erf', 'erfc', 'gamma', 
                    #'relu',
                    #"inv(x) = 1/x",
                    #"sigmoid(x) = 1/(1+exp(-x))",
                    #"swish(x)   = x/(1+exp(-x))",
                    # ^ Custom operator (julia syntax)
                ],
                extra_sympy_mappings={"inv": lambda x: 1 / x},
                #select_k_features=5,
                # ^ Define operator for SymPy as well
                loss="loss(prediction, target) = (prediction - target)^2",
                # ^ Custom loss function (julia syntax)
            )
            #reg_list.append(('SR',reg_sr))
            
            # reg = NEATRegressor(
            #         number_of_generations=50,
            #         fitness_criterion='min',
            #         fitness_threshold=0.95,
            #         pop_size=150,
            #         activation_options='sigmoid relu tanh gauss inv hat clamped sin square abs exp identity',
            #         activation_mutate_rate=0.01,
            #         activation_default='relu',
            #         )
            
            # net = ORELM(inputs=nDimInput,outputs=nDimOutput, 
            #             #inputs=numLags*numCols, outputs=nDimOutput,
            #             numHiddenNeurons=10, #outputWeightForgettingFactor=0.92,
            #             inputWeightForgettingFactor=0.97,#InWeightFF,
            #             outputWeightForgettingFactor=0.97,#OutWeightFF,
            #             hiddenWeightForgettingFactor=.9,#HiddenWeightFF,
            #             )
            #net.initializePhase(lamb=0.01)
            
            reg_bayes = BayesSearchCV(
                SVR(max_iter=500, verbose=True),
                {
                    'C': (1e-6, 1e+4, 'log-uniform'),
                    'gamma': (1e-6, 1e+1, 'log-uniform'),
                    'kernel': ['linear',  'rbf'],  # categorical parameter
                    #'degree': (1, 3),  # integer valued parameter
                    #'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
                },
                n_iter=32,
                cv=3
            )
            reg_list.append(('SVR-BAYES',reg_bayes))
            
            #https://pypi.org/project/tsmoothie/
            from tsmoothie.smoother import LowessSmoother
            for reg_name, reg in reg_list:
                list_results=[]
                
                X_data, y_data = X_train.copy(), y_train.copy()
                y_pred=[]
                dataset_size=100
                smooth_input=False
              
                batch=1
                n_batch=batch             
                for j in range(len(y_test)):    
                    #--
                    #from statsmodels.tsa.seasonal import STL                
                    #model = STL(y_data, period=90)
                    #res = model.fit()
                    #res_ = np.c_[res.trend,res.seasonal,res.resid]
                    #import emd
                    #imf = emd.sift.sift(y_data)
                    #print(imf.shape, y_data.shape)
                    #emd.plotting.plot_imfs(imf)
                    #imf=np.r_[imf, imf[-1:,:]]
                    #--
                    
                    X_data_=X_data.copy()
                    y_data_ = y_data.copy()
                    if j > dataset_size:
                            X_data_ = X_data[-dataset_size:,:]
                            y_data_ = y_data[-dataset_size:]
                    
                    if smooth_input:
                      for z in range( X_data_.shape[1]):
                        smoother = LowessSmoother(smooth_fraction=0.1, iterations=6)
                        smoother.smooth(X_data_[:,z])
                        X_data_[:,z]=smoother.smooth_data.ravel()
                        #pl.plot(X_data[:,z], 'r-', X_data_[:,z]); pl.title(feature_names[z]);pl.show()
                    
                    if n_batch>=batch:
                        n_batch=0
                        ## OSELM
                        #net.train(X_data, y_data.reshape(-1,1))                                
                        #net.train(X_data_, y_data_.reshape(-1,1))
                        ## XGB
                        #print(X_data_.shape)
                        #reg.fit(X_data_, y_data_.reshape(-1,1))
                        reg.fit(X_data_, y_data_)
                        #print('-'*40, '\n', reg.sympy())
                        #display(reg.sympy().simplify())
    
                        
                    n_batch+=1
                
                    #output = net.predict(X_test[[j], :])
                    #output=[reg.predict(X_test[[j], :])]
                    output=[[reg.predict(X_test[[j], :]).sum()]]
                    
                             
                    y_pred.append(output[0][0])            
                    X_data = np.r_[X_data, X_test[j].reshape(1,-1)]
                    y_data = np.r_[y_data, y_test[j]]
                    if j%2==0 and j>1:
                            smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                            smoother.smooth(y_pred)
                            low, up = smoother.get_intervals('prediction_interval')
                    
                            pl.plot(y_test[1:(j+1)],'r-', label='Observado');
                            pl.plot(smoother.smooth_data.ravel()[1::],'b-', label='Tendência'); 
                            pl.fill_between(range(len(smoother.data.ravel()[1::])), low.ravel()[1::], up.ravel()[1::], alpha=0.2    )
                            pl.plot(y_pred[1::],'g-', label='Previsto');
                            pl.legend()
                            pl.ylabel('Vazão')
                            pl.title(reg_name+' - Previsão '+str(look_forward)+' passos à frente')
                            pl.show()
                            #pl.plot(y_test[1:j],'r-',); pl.show()
                            print(j, end=' ')
                   
                #%%
                y_pred = np.array(y_pred)
                y_true=y_test
                rmse, r2 = mean_squared_error(y_true, y_pred)**.5, r2_score(y_true, y_pred)
                r=pearsonr(y_true.ravel(), y_pred.ravel())[0] 
                print(rmse, r2,r)
                            
                fig = pl.figure(figsize=[8,4])
                pl.plot(y_true, 'r-', y_pred,'b-')
                pl.title(reg_name+' - '+target+'\n'+'RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
                pl.show()
                #
                fig = pl.figure(figsize=[5,5])
                pl.plot(y_true,y_true, 'r-', y_test,y_pred,'b.')
                pl.title(reg_name+'\n'+target+'\n'+'RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
                pl.show()
                #pl.plot(y_true[2:], 'r-', y_pred[2:],'b-')
                #%%
                X_test_scaled=X_test
                n_outcomes=250000
                data=np.random.uniform( low=X_test_scaled.min(axis=0), high=X_test_scaled.max(axis=0), size=(n_outcomes, X_test_scaled.shape[1]) )
                predict = reg.predict(data)
                median = np.median(predict)
                mad=np.abs(predict - median).mean()
                uncertainty = 100*mad/median
                print(reg_name, median, mad, n_features, uncertainty/n_features, uncertainty)
                
                ix=y_pred>-1e12; unc_p,unc_t = y_pred[ix], y_true[ix]
                unc_e = ((unc_p) - (unc_t))
                unc_m=unc_e.mean()
                unc_s = np.sqrt(sum((unc_e - unc_m)**2)/(len(unc_e)-1))
                #pei95=fmt(10**(-unc_m-1.96*unc_s))+' to '+fmt(10**(-unc_m+1.96*unc_s))
                pei95=fmt((-unc_m-1.96*unc_s))+' to '+fmt((-unc_m+1.96*unc_s))
                
                print(reg_name, fmt(unc_m), fmt(unc_s), pei95 )
                sig = '+' if unc_m > 0 else ''
                #%%
                ds_name  = dataset_name.replace('/','_').replace("'","").lower()
                tg_name  = target.replace('/','_').replace("'","").lower()
                est_name = reg_name
                list_results.append({
                    'Y_TEST_TRUE':y_pred, 'Y_TEST_PRED':y_test, 'RUN':run, 
                    'EST_NAME':est_name, 'SEED':random_seed,'OUTPUT':tg_name,
                    'EST_PARAMS':None, 
                    #
                    'run':run, 'seed':random_seed, 'estimator':reg_name,
                    'y_true':y_test,'y_pred':y_pred,
                    'dataset_name':dataset_name,
                    'feature_names':feature_names,'target':target,                
                    'estimator_name':reg.__str__().split('(')[0],
                    'coef':None, 'final_estimator':None,
                     #'X_train':X_train_scaled,'y_train':y_train,
                     #'X_test':X_test_scaled,'y_test':y_test,
                    'EST_PARAMS':reg.get_params(),
                    'final_estimator':reg,                  
                    #
                    'feature_names':feature_names,'target':target,                
                    'Model':est_name,  'No. features':n_features, 'Median':median, 
                    'MAD':mad, 'Uncertainty':uncertainty,
                    'MPE':sig+fmt(unc_m), 'WUB':'$\pm$'+fmt(unc_s), 'PEI95':pei95,
                    #'SHAP':abs(shap_values).mean(axis=0),
                    #
                    'batch':batch, 'dataset_size':dataset_size,
                })
                #%%
                data     = pd.DataFrame(list_results)
                pk=(path+#'_'+
                    basename+'_'+
                    '_run_'+str("{:02d}".format(run))+'_'+
                    ("%15s"%ds_name         ).rjust(15).replace(' ','_')+#'_'+
                    ("%9s"%est_name         ).rjust( 9).replace(' ','_')+#'_'+
                    #("%10s"%alg_name        ).rjust(10).replace(' ','_')+#'_'+
                    ("%15s"%tg_name         ).rjust(25).replace(' ','_')+#'_'+
                    #("%15s"%os.uname()[1]   ).rjust(25).replace(' ','_')+#'_'+
                    #time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                    '.json') 
                pk=pk.replace(' ','_').replace("'","").lower()
                pk=pk.replace('(','_').replace(")","_").lower()
                pk=pk.replace('[','_').replace("]","_").lower()
                pk=pk.replace('-','_').replace("_","_").lower()
                pk=pk.replace('{','').replace("}","").lower()
                pk=pk.replace('$','').replace("}","").lower()
                #print(pk)
                data.to_json(pk)
                #%%

