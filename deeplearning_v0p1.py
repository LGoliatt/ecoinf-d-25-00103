#!/usr/bin/python
# -*- coding: utf-8 -*-    


# https://www.tensorflow.org/tutorials/structured_data/time_series
# https://keras.io/api/layers/core_layers/lambda/
# https://blog.paperspace.com/working-with-the-lambda-layer-in-keras/

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
#from util.metrics import agreementindex, rmse, rrmse, kge_non_parametric, mape, mse

from read_data_ankara import *

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


basename='dl_ankara__'
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

#-------------------------------------------------------------------------------
datasets = [
            #read_data_ankara(variation= 3,station='Ankara', test=0.25, expand_features=False, ),
            read_data_ankara(variation= 6,station='Ankara', test=0.25, expand_features=False, ),
            #read_data_ankara(variation=12,station='Ankara', test=0.25, expand_features=False, ),
           ]     

#%%
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.regularizers import l1, l2
import numpy as np


def norm2(x):
    #x -= K.mean(x, axis=1, keepdims=True)
    #x = K.l2_normalize(x, axis=1)
    n=x.shape[1]
    
    t=[
       K.identity(x),       
       #K.sigmoid(x),
       #K.prod(x[:,0:1], axis=1,  keepdims=True),
       #K.pow(x, -1),       
       ]
    # for i in range(n):
    #     j=i+2 if i< n else -1
    #     t+=[
    #         K.prod(x[:,i:j], axis=1,  keepdims=True),
    #         ]
    #for i in range(n):
    #    j=i+1 if i< n else -1
    #    t+=[
    #        K.pow(K.prod(x[:,i:j], axis=1,  keepdims=True),-1),
    #        ]
        
    return K.concatenate(t, axis=1)


 
def antirectifier(x):
    #x -= K.mean(x, axis=1, keepdims=True)
    #x = K.l2_normalize(x, axis=1)
    y=K.identity(x)
    n=x.shape[1]
    t=[
       K.identity(x),       
       K.sigmoid(x),       
       #K.elu(x),
       #K.relu(x),
       #K.exp(x),
       K.sin(x),
       K.cos(x),
       K.tanh(x),
       #K.prod(x, axis=1,  keepdims=True),
       #K.pow(x, +2),
       #K.pow(x, -1),
       #K.pow(K.identity(K.prod(x, axis=1,  keepdims=True)), -1),
       #K.relu(-x),
       #K.batch_dot(x, y),
       ]
    return K.concatenate(t, axis=1)


def lamb(x):
    #x -= K.mean(x, axis=1, keepdims=True)
    #x = K.l2_normalize(x, axis=1)
    print(x.shape)
    z=x
    print(x[:,0])
    t1 = x
    t2 = K.sigmoid(x)
    t3 = x**2
    t4 =[]
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i>=j:
                t4.append(x[:,i]*x[:,j])
    
    return K.concatenate([t1,t2,t3,t4], axis=1)


def build_model(shape, mdl='linear', patience=3):
  
  model = keras.Sequential([
      keras.layers.Input(shape=shape),
      keras.layers.Lambda(norm2),
      keras.layers.Dense(1,kernel_regularizer = l1(0.1))
  ])

  if mdl=='linear':
      model = keras.experimental.LinearModel()
  
  if mdl=='fe01':
      model = keras.Sequential([
          keras.layers.Input(shape=shape),
          keras.layers.Lambda(antirectifier),
          keras.experimental.LinearModel(),
      ])

  if mdl=='fe02':
      model = keras.Sequential([
          keras.layers.Input(shape=shape),
          keras.layers.Lambda(antirectifier),
          keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
      ])
      
  if mdl=='fe03':
      model = keras.Sequential([
          keras.layers.Input(shape=shape),
          keras.layers.Lambda(antirectifier),
          keras.layers.Dense(64, activation='linear',),
          #keras.layers.Dropout(0.1),
          keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
      ])

  if mdl=='fe04':
    m2_input_layer = keras.layers.Input(shape=shape),
    m2_dense_layer_1 = keras.layers.Dense(32, activation='relu')(m2_input_layer)
    m2_dense_layer_2 = keras.layers.Dense(16, activation='relu')(m2_input_layer)
    m2_merged_layer = keras.layers.concatenate([m2_dense_layer_1, m2_dense_layer_2], name='Concatenate')
    m2_final_layer = keras.layers.Dense(1, activation='linear')(m2_merged_layer)
    model2 = keras.Model(inputs=m2_input_layer, outputs=m2_final_layer, name="Model_2")
    model2.save_weights("model2_initial_weights.h5")
    model2.summary()
    keras.utils.plot_model(model2, 'model2.png', show_shapes=True)




  if mdl=='fe05':
    left_branch_input = keras.layers.Input(shape=shape, name='Left_input')
    left_branch_output = keras.layers.Dense(10, activation='linear')(left_branch_input)
    
    right_branch_input = keras.layers.Input(shape=shape, name='Right_input')
    right_branch_output = keras.layers.Dense(10, activation='relu')(right_branch_input)
    
    concat = keras.layers.concatenate([left_branch_output, right_branch_output], name='Concatenate')
    final_model_output = keras.layers.Dense(1, activation='sigmoid')(concat)
    final_model = keras.Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                        name='Final_output')
    
    model=final_model
  
  if mdl=='fe06':
      model1 = keras.Sequential([
          keras.layers.Input(shape=shape),
          keras.layers.Lambda(antirectifier),
          keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
      ])
      model2 = keras.Sequential([
          keras.layers.Input(shape=shape),
          keras.layers.Dense(64, activation='linear',),
          #keras.layers.Dropout(0.1),
          keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
      ])
      model3 = keras.Sequential([
          keras.layers.Input(shape=shape),
          keras.layers.Dense(64, activation='relu',),
          #keras.layers.Dropout(0.1),
          keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
      ])
      # model4 = keras.Sequential([
      #     keras.layers.Input(shape=shape),
      #     keras.layers.Conv1D(kernel_size=5,input_shape=shape),
      #     keras.layers.Dense(64, activation='linear',),
      #     #keras.layers.Dropout(0.1),
      #     keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
      # ])
      
      model = keras.Sequential([
            keras.layers.Input(shape=shape),
            keras.layers.concatenate([model1, model2, model3], ),
            keras.layers.Dense(64, activation='linear'),
            keras.layers.Dense(1,kernel_regularizer = l1(0.1)),
        ])
  
  if mdl=='nn':
    model = keras.Sequential([
      keras.layers.Input(shape=shape),
      keras.layers.Dense(64, activation='linear', input_shape=shape),
      keras.layers.Dropout(0.05),
      keras.layers.Dense(64, activation='linear'),
      keras.layers.Dropout(0.05),
      keras.layers.Dense(64, activation='linear'),
      keras.layers.Dropout(0.05),
      #keras.layers.Dense(8, activation='linear'),
      #keras.layers.Dense(4, activation='linear'),
      keras.layers.Dense(1),
    ])
  
    

  optimizer = keras.optimizers.Adam()
  model.compile(loss=['mse',],
                optimizer=optimizer,
                metrics=['mse', #'mse'
                         ])
  
  dot_img_file = '/tmp/model_1.png'
  keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=False)
  
  return model


#%%----------------------------------------------------------------------------   
pd.options.display.float_format = '{:.3f}'.format

plot=True
n_runs=10
for run in range(0, n_runs):
    random_seed=run+10
    
    for dataset in datasets:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+basename+'_'+dr+'/'
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
            
            shape=(X_train.shape[1],)
            
            MAX_EPOCHS = 50000
            
            patience=3
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=patience,
                                                                mode='min')
            
            model = build_model((X_train.shape[1],), mdl='--')
            
            history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                                  validation_split=0.15,
                                  verbose=0,
                                  callbacks=[early_stopping])
            
            #pd.DataFrame(history.history).plot()
            #%%
            y_pred = model.predict(X_test)
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
                pl.title(dataset_name+' - '+'\n'+'RMSE = '+str(rmse_)+'\n'+'R$^2$ = '+str(r2)+'\n'+'KGE = '+str(kge_))
                pl.show()
            #
            #fig = pl.figure(figsize=[8,8])
            #pl.plot(y_test,y_test, 'r-', y_test,y_pred,'bo')
            #pl.title(dataset_name+' - '+samples+' - '+strategy+'\n'+'RMSE = '+str(rmse_)+'\n'+'NSE = '+str(nse_)+'\n'+'KGE = '+str(kge_))
            #pl.show()
            #s1 = "%3d: "%run+dataset_name.ljust(15)+' - '+"%0.3f"%rmse_+' - '+"%0.3f"%nse_
            #s1+= ' >> '+"%0.3f"%beta
            #s1+= ' | '+ ', '.join(feature_names[ft])+' -- '
            #s1+= ' '.join(["%1.6f"%i for i in model.coef_])+" | %1.3f"%model.intercept_
            #s1+= ' '.join(["%1.6f"%i for i in model[model.steps[-1][0]].coef_])+" | %1.3f"%model[model.steps[-1][0]].intercept_
            #print(s1)
# #%%
#                 l={
#                 'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':model.predict(X_train[:,ft]), 
#                 'Y_TEST_TRUE':y_test, 'Y_TEST_PRED':y_pred, 'RUN':run,            
#                 'EST_PARAMS':{'l1_ratio':z[-1], 'alpha':z[-2]}, 
#                 'PARAMS':z, 'ESTIMATOR':model, 'FEATURE_NAMES':feature_names,
#                 'SEED':random_seed, 'DATASET_NAME':dataset_name,
#                 'ALGO':'DE', 'ALGO_STRATEGY':strategy,
#                 'ACTIVE_VAR':ft, 'ACTIVE_VAR_NAMES':feature_names[ft],
#                 'MODEL_COEF':model.coef_, 'MODEL_INTERCEPT':model.intercept_,
#                  #'MODEL_COEF':model[model.steps[-1][0]].coef_, 'MODEL_INTERCEPT':model[model.steps[-1][0]].intercept_,
#                 'BETA':beta,
#                 }
                
#                 pk=(path+basename+'_'+("%15s"% dataset_name).rjust(15)+
#                     '_run_'+str("{:02d}".format(run))+'_'+
#                     '_'+samples+'_'+
#                     '_'+'beta'+str("%1.2f"%beta).replace('.','p')+'_'+
#                     ("%15s"%target).rjust(15)+'.pkl')
#                 pk=pk.replace(' ','_').replace("'","").lower()
#                 pk=pk.replace('(','_').replace(")","_").lower()
#                 pk=pk.replace('[','_').replace("]","_").lower()
#                 pk=pk.replace('-','_').replace("-","_").lower()
#                 pd.DataFrame([l]).to_pickle(pk)
# #%%
