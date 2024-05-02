#https://sysidentpy.org/examples/multiobjective_parameter_estimation/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.multiobjective_parameter_estimation import AILS
from sysidentpy.basis_function._basis_function import Polynomial, Fourier
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.narmax_tools import set_weights
from sysidentpy.general_estimators import NARX


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression, Ridge, Lasso
from sysidentpy.general_estimators import NARX
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from scipy.optimize import Bounds
from scipy import optimize 
from scipy.stats import qmc

import warnings
warnings.warn("deprecated", DeprecationWarning)
#%%

from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from xgboost import XGBRegressor

from read_data_ankara import *

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
    
    dataset_name = D['name']
    train_data=pd.DataFrame(data=D['X_train'], columns=D['feature_names'])
    train_data[D['target_names'][0]]=D['y_train'].squeeze()
    
    test_data=pd.DataFrame(data=D['X_test'], columns=D['feature_names'])
    test_data[D['target_names'][0]]=D['y_test'].squeeze()
    
    
    target=D['target_names'][0]
    prediction_length=len(test_data)
    
    cols = train_data.columns
    for col in cols:
        train_data[col] = train_data[col].astype(float)
        train_data[col] = train_data[col]/train_data[col].max()
    
    cols = test_data.columns
    for col in cols:
        test_data[col] = test_data[col].astype(float)
        test_data[col] = test_data[col]/test_data[col].max()
    
    #prediction_length=int(365*1)
    #test_data = df[-prediction_length:]
    #train_data=df[:-prediction_length]

    
    
    x_train=train_data.drop([target],axis=1).values
    y_train=train_data[target].values.reshape(-1,1)
    
    x_valid=test_data.drop([target],axis=1).values
    y_valid=test_data[target].values.reshape(-1,1)
    
    n_features=x_train.shape[1]
    
    #%%
    #
    # hybrid DE-Neural-NARX for short-term streamflow prediction
    #
    from sklearn.model_selection import TimeSeriesSplit
    def build_narmax(w,estimator=None, flag='eval'):
        basis_function = Polynomial(degree=round(w[0]))
        n_terms=round(w[1])
        ylag=round(w[2])
        xlag =[[1,round(w[3+i])] for i in range(n_features)]
        model=NARX(
            base_estimator=estimator,
            ylag=ylag,
            xlag=xlag,
            basis_function=basis_function,
            model_type="NARMAX",
        )
        
        if flag!='eval':
            return model
        
        
        tscv = TimeSeriesSplit(n_splits=3)
        rrse = 0
        for train, test in tscv.split(x_train):
            #print("%s %s" % (train, test))
            #plt.figure()
            #plt.plot(y_train,'k-',y_train[train])
            #plt.show()
            model.fit(X=x_train[train], y=y_train[train])
            yhat = model.predict(X=x_train[test], y=y_train[test], steps_ahead=None)
            rrse += root_relative_squared_error(y_train[test], yhat)/len(test)
            #plot_results(y=y_train[test], yhat=yhat, n=1000, )
            #print(rrse)
            
        rrse=1e12 if np.isnan(rrse) else rrse
        #print(rrse,w)
        return rrse
    
    
    act={0:'identity',1:'tanh',2:'relu',3:'logistic'}
    def build_ann(x,flag='eval'):
        n=8
        nc=int(round(x[n+0]))
        activation=act[round(x[n+1])]
        hidden_layer_sizes=[round(x[n+2+i]) for i in range(nc)]
        
        estimator=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation)
        if flag!='eval':
            par={'nc':nc,'ac':activation,'hl':hidden_layer_sizes}
            return par,build_narmax(x[:n],estimator, flag)
        
        rrse=build_narmax(x[:n],estimator, flag)
        #print(rrse,x[n:])
        return rrse
    
    def build_mlp(x,flag='eval'):
        n=8
        nc=int(round(x[n+0]))
        activation=act[round(x[n+1])]
        hidden_layer_sizes=[round(x[n+2]) for i in range(nc)]
        
        estimator=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=2000,)
        if flag!='eval':
            par={'nc':nc,'ac':activation,'hl':hidden_layer_sizes}
            return par,build_narmax(x[:n],estimator, flag)
        
        rrse=build_narmax(x[:n],estimator, flag)
        #print(rrse,x[n:])
        return rrse
    
    
    estimators=[
        #
        ('MLP'  , 
         build_mlp,
         [1.0,1,1,1,1,1,1,1]+[1,0,1,],
         [1.1,50,5,10,10,10,10,10,]+[3,3,30,],
        ),
        
        # ('ANN'  , 
        #   build_ann,
        #   [1,1,1,1,1,1,1,1]+[1,0,1,1,1,],
        #   [3,50,5,10,10,10,10,10,]+[3,3,30,30,30,],
        #  ),        
    ]
    
    for estimator_name, fun, lb, ub in estimators:        
        
        #res = optimize.shgo(fun, bounds=tuple(zip(lb,ub)), args=('eval',), n=10, iters=1, sampling_method='sobol', options={'maxfev':100})
        init=qmc.scale(qmc.LatinHypercube(d=len(lb)).random(n=10), lb, ub)
        res = optimize.differential_evolution(fun, bounds=Bounds(lb,ub), init=init, maxiter=10, workers=1,disp=True, polish=False)
        
        xb=res['x']
        model_param, model=fun(xb,flag='ref')
        print(model_param)
        
        model.fit(X=x_train, y=y_train)
        yhat = model.predict(X=x_valid, y=y_valid, steps_ahead=None)
        rrse = root_relative_squared_error(y_valid, yhat)
        #plot_results(y=y_valid, yhat=yhat, n=1000, title=s+str("\nRRSE = %2.3f"%rrse))
        plt.plot(y_valid,'r-', yhat,'b-',); plt.title(dataset_name+'-'+estimator_name+str("\nRRSE = %2.3f"%rrse)); plt.show()


#%%
