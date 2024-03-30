from sklearn.model_selection import (TimeSeriesSplit)
import pylab as plt
from read_data_ankara import *

dataset=read_data_ankara(variation= 6,station='Ankara', test=0.0, expand_features=False, )
tk=0
y_train, y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']


n_splits=4
fig, ax = plt.subplots(n_splits, figsize=(6, 8))
cmap_cv = plt.cm.coolwarm
cv=TimeSeriesSplit(n_splits=n_splits)
for ii, (tr, tt) in enumerate(cv.split(X=X_train, y=y_train,)):
    # Fill in indices with the training/test groups
    indices = np.array([np.nan] * len(X_train))
    indices[tt] = 1
    indices[tr] = 0

    # Visualize the results
    
    ax[ii].plot(range(len(indices)),y_train,'w', alpha=0.0, )#label='Fold = '+str(ii))
    ax[ii].axvline(tr[-1], ls='--', c='k')
    ax[ii].plot(tr, y_train[tr],'r', label='Train')
    ax[ii].plot(tt, y_train[tt],'b', label='Test' )
    ax[ii].set_ylabel('Fold = '+str(ii+1))
    if ii==0:
        ax[ii].legend(loc=1)
    
plt.savefig('tscv.png',dpi=300,bbox_inches='tight')