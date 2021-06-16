import seaborn as sns
import pylab as pl
from io import BytesIO
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error,r2_score

#
# Leitura dos dados
#
link='https://github.com/alidanandeh/ENN-SA/blob/master/SPI3.xlsx?raw=true'

def read_ankara(spi=3, test=0.25):
  link='https://github.com/alidanandeh/ENN-SA/blob/master/SPI'+str(spi)+'.xlsx?raw=true'
  data = pd.read_excel(link)

  data.dropna(inplace=True)
  target_names=['ANKARA']
  variable_names = data.columns.drop(target_names)
  
  X=data[variable_names]
  y=data[target_names]
  k=int(data.shape[0]*(1-test))

  n=k; 
  X_train, X_test = X.iloc[:n].values, X.iloc[n:].values    
  y_train, y_test = y.iloc[:n].values, y.iloc[n:].values    
  n_samples, n_features = X_train.shape 
    
  n_samples, n_features = X_train.shape
  dataset =  {
      'task'            : 'forecast',
      'name'            : 'Drought SPI-'+str(spi),
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'y_train'         : y_train.T,
      'X_test'          : X_test,
      'y_test'          : y_test,
      'targets'         : target_names,
      'descriptions'    : 'None',
      'reference'       : "https://doi.org/10.1016/j.cageo.2020.104622",
      }

  return(dataset)

#
# Recuperação dos dados
#

dataset = read_ankara(spi=3)
target                          = dataset['targets']
y_train, y_test                 = dataset['y_train'], dataset['y_test']
dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
task                            = dataset['task']
n_samples_test                  = len(y_test)

s=''+'\n'
s+='='*80+'\n'
s+='Dataset                    : '+dataset_name+' -- '+'\n'
s+='Number of training samples : '+str(n_samples_train) +'\n'
s+='Number of testing  samples : '+str(n_samples_test) +'\n'
s+='Number of features         : '+str(n_features)+'\n'
s+='Task                       : '+str(dataset['task'])+'\n'
s+='='*80
s+='\n'     

print(s)

#
# Parametrização do problema de aprendizado de máquina
#
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

#------------------------------------------------------------------
lb  = [0.0]*n_features  + [0.9, 1e-6,  1e-6,    0,  ]
ub  = [1.0]*n_features  + [1.0, 1e+2,  2e+3,    1,  ]
#------------------------------------------------------------------ 
feature_names = dataset['feature_names']


#
# Definição de uma função objetivo
#
def find_kernel(i):
    fk={0:'linear', 1:'rbf', 2:'poly', 3:'sigmoid'}
    return fk[i]

def objective_function(x,*args):
    X,y,random_seed = args
    n_samples, n_features=X.shape
    ft = [ i>0.5 for i in  x[:n_features] ]
    if sum(ft)==0:
        return 1e12                              
    model=SVR(C=x[-2], epsilon=x[-1], gamma=x[-3],
              tol=1e-6, kernel=find_kernel(round(x[-4])),
              max_iter=5000)                    
    cv=TimeSeriesSplit(n_splits=5,)
    r=cross_val_score(model,X[:,ft], y.ravel(), cv=cv, n_jobs=1, scoring='neg_root_mean_squared_error')
    r=np.abs(r).mean()
    return r

def plot_model_results(y_test, y_pred):
    y_pred = np.array(y_pred)
    rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
    #print(rmse, r2,)
    tit=dataset_name+'\nRMSE = '+str(rmse)+'\nR$^2$ = '+str(r2)
                
    fig = pl.figure(figsize=[12,4])
    pl.plot(y_test, 'r-o', y_pred,'b-', ms=3)
    pl.title(tit)
    pl.show()
    #
    #fig = pl.figure(figsize=[5,5])
    #pl.plot(y_test,y_test, 'r-', y_test,y_pred,'bo')
    #pl.title(tit); pl.xlabel('Observed'); pl.ylabel('Predicted')
    #pl.show()
    

#
# Escolha do algoritmo de otimização
#
from scipy.optimize import dual_annealing, differential_evolution as de
from pyswarm import pso


method='DE'
for run in range(10):
    random_seed=run+100
    args=(X_train, y_train, random_seed)
    
    np.random.seed(random_seed)
    init=np.random.uniform(low=lb, high=ub, size=(25,len(lb)))          
    if method=='DE':
        res = de(objective_function, bounds=tuple(zip(lb,ub)), args=args,
                 #strategy=strategy,
                 init=init, maxiter=50, tol=1e-8,  
                 mutation=0.9,  recombination=0.9, 
                 disp=True, 
                 seed=random_seed)
    else:
        res=dual_annealing(objective_function, bounds=list(zip(lb, ub)), args=args, 
                       maxfun=1000, seed=random_seed)
    
    
    
    #
    # Recuperação dos melhores modelos
    #
    z=res['x']
    ft = [ i>0.5 for i in  z[:n_features] ]
    model=SVR(C=z[-2], epsilon=z[-1], gamma=z[-3],
                  tol=1e-6, kernel=find_kernel(round(z[-4])),
                  max_iter=5000,
                  )    
    y_pred=model.fit(X_train[:,ft], y_train.ravel()).predict(X_test[:,ft])
    plot_model_results(y_test, y_pred)
