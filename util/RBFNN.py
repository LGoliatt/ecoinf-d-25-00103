#!/usr/bin/python
# -*- coding: utf-8 -*

"""rbf - Radial basis functions for interpolation/smoothing scattered Nd data.
Written by John Travers <jtravs@gmail.com>, February 2007
Based closely on Matlab code by Alex Chirokov
Additional, large, improvements by Robert Hetland
Some additional alterations by Travis Oliphant
Interpolation with multi-dimensional target domain by Josua Sassen
Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Copyright (c) 2007, John Travers <jtravs@gmail.com>
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.
    * Neither the name of Robert Hetland nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import division, print_function, absolute_import

#from scipy.interpolate import Rbf

import numpy as np
import sys
from scipy import linalg
#from scipy._lib.six import callable, get_method_function, get_function_code
from six import callable, get_method_function, get_function_code
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ['Rbf']

###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################

class RBFNNRegressor(BaseEstimator, RegressorMixin):

    class Rbf(object):
        """
        Rbf(*args)
        A class for radial basis function interpolation of functions from
        n-dimensional scattered data to an m-dimensional domain.
        Parameters
        ----------
        *args : arrays
            x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
            and d is the array of values at the nodes
        function : str or callable, optional
            The radial basis function, based on the radius, r, given by the norm
            (default is Euclidean distance); the default is 'multiquadric'::
                'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                'gaussian': exp(-(r/self.epsilon)**2)
                'linear': r
                'cubic': r**3
                'quintic': r**5
                'thin_plate': r**2 * log(r)
            If callable, then it must take 2 arguments (self, r).  The epsilon
            parameter will be available as self.epsilon.  Other keyword
            arguments passed in will be available as well.
        epsilon : float, optional
            Adjustable constant for gaussian or multiquadrics functions
            - defaults to approximate average distance between nodes (which is
            a good start).
        smooth : float, optional
            Values greater than zero increase the smoothness of the
            approximation.  0 is for interpolation (default), the function will
            always go through the nodal points in this case.
        norm : str, callable, optional
            A function that returns the 'distance' between two points, with
            inputs as arrays of positions (x, y, z, ...), and an output as an
            array of distance. E.g., the default: 'euclidean', such that the result
            is a matrix of the distances from each point in ``x1`` to each point in
            ``x2``. For more options, see documentation of
            `scipy.spatial.distances.cdist`.
        mode : str, optional
            Mode of the interpolation, can be '1-D' (default) or 'N-D'. When it is
            '1-D' the data `d` will be considered as one-dimensional and flattened
            internally. When it is 'N-D' the data `d` is assumed to be an array of
            shape (n_samples, m), where m is the dimension of the target domain.
        Attributes
        ----------
        N : int
            The number of data points (as determined by the input arrays).
        di : ndarray
            The 1-D array of data values at each of the data coordinates `xi`.
        xi : ndarray
            The 2-D array of data coordinates.
        function : str or callable
            The radial basis function.  See description under Parameters.
        epsilon : float
            Parameter used by gaussian or multiquadrics functions.  See Parameters.
        smooth : float
            Smoothing parameter.  See description under Parameters.
        norm : str or callable
            The distance function.  See description under Parameters.
        mode : str
            Mode of the interpolation.  See description under Parameters.
        nodes : ndarray
            A 1-D array of node values for the interpolation.
        A : internal property, do not use
        Examples
        --------
        >>> from scipy.interpolate import Rbf
        >>> x, y, z, d = np.random.rand(4, 50)
        >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
        >>> xi = yi = zi = np.linspace(0, 1, 20)
        >>> di = rbfi(xi, yi, zi)   # interpolated values
        >>> di.shape
        (20,)
        """
        # Available radial basis functions that can be selected as strings;
        # they all start with _h_ (self._init_function relies on that)
        def _h_multiquadric(self, r):
            return np.sqrt((1.0/self.epsilon*r)**2 + 1)

        def _h_inverse_multiquadric(self, r):
            return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

        def _h_gaussian(self, r):
            return np.exp(-(1.0/self.epsilon*r)**2)

        def _h_linear(self, r):
            return r

        def _h_cubic(self, r):
            return r**3

        def _h_quintic(self, r):
            return r**5

        def _h_thin_plate(self, r):
            return xlogy(r**2, r)

        def _h_relu(self, r):
            #print('>>>>>>>>>',  r)
            return xlogy(r**2, r)

        def _h_sigmoid(self, r):
            return 1.0/(1+self.epsilon*np.exp(-r))

        def _h_swish(self, r):
            return r/(1+self.epsilon*np.exp(-r))

        # Setup self._function and do smoke test on initial r
        def _init_function(self, r):
            if isinstance(self.function, str):
                self.function = self.function.lower()
                _mapped = {'inverse': 'inverse_multiquadric',
                        'inverse multiquadric': 'inverse_multiquadric',
                        'thin-plate': 'thin_plate'}
                if self.function in _mapped:
                    self.function = _mapped[self.function]

                func_name = "_h_" + self.function
                #print(func_name)
                if hasattr(self, func_name):
                    self._function = getattr(self, func_name)
                else:
                    functionlist = [x[3:] for x in dir(self)
                                    if x.startswith('_h_')]
                    raise ValueError("function must be a callable or one of " +
                                    ", ".join(functionlist))
                self._function = getattr(self, "_h_"+self.function)
            elif callable(self.function):
                allow_one = False
                if hasattr(self.function, 'func_code') or \
                hasattr(self.function, '__code__'):
                    val = self.function
                    allow_one = True
                elif hasattr(self.function, "im_func"):
                    val = get_method_function(self.function)
                elif hasattr(self.function, "__call__"):
                    val = get_method_function(self.function.__call__)
                else:
                    raise ValueError("Cannot determine number of arguments to "
                                    "function")

                argcount = get_function_code(val).co_argcount
                if allow_one and argcount == 1:
                    self._function = self.function
                elif argcount == 2:
                    if sys.version_info[0] >= 3:
                        self._function = self.function.__get__(self, Rbf)
                    else:
                        import new
                        self._function = new.instancemethod(self.function, self,
                                                            Rbf)
                else:
                    raise ValueError("Function argument must take 1 or 2 "
                                    "arguments.")

            a0 = self._function(r)
            if a0.shape != r.shape:
                raise ValueError("Callable must take array and return array of "
                                "the same shape")
            return a0

        def __init__(self, *args, **kwargs):
            # `args` can be a variable number of arrays; we flatten them and store
            # them as a single 2-D array `xi` of shape (n_args-1, array_size),
            # plus a 1-D array `di` for the values.
            # All arrays must have the same number of elements
            
            #print('1 >>>>>>>>>>>>>>>',args)#,<<
            #print('2 >>>>>>>>>>>>>>>',kwargs)#,<<

            self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                                for a in args[:-1]])
            self.N = self.xi.shape[-1]

            self.mode = kwargs.pop('mode')#, '1-D')

            if self.mode == '1-D':
                self.di = np.asarray(args[-1]).flatten()
                self._target_dim = 1
            elif self.mode == 'N-D':
                self.di = np.asarray(args[-1])
                self._target_dim = self.di.shape[-1]
            else:
                raise ValueError("Mode has to be 1-D or N-D.")

            if not all([x.size == self.di.shape[0] for x in self.xi]):
                raise ValueError("All arrays must be equal length.")

            self.norm = kwargs.pop('norm')#, 'euclidean')
            self.epsilon = kwargs.pop('epsilon')#, None)
            if self.epsilon is None:
                # default epsilon is the "the average distance between nodes" based
                # on a bounding hypercube
                ximax = np.amax(self.xi, axis=1)
                ximin = np.amin(self.xi, axis=1)
                edges = ximax - ximin
                edges = edges[np.nonzero(edges)]
                self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)

            self.smooth = kwargs.pop('smooth')#, 0.0)
            self.function = kwargs.pop('func')#, 'multiquadric')

            
            # attach anything left in kwargs to self for use by any user-callable
            # function or to save on the object returned.
            for item, value in kwargs.items():
                setattr(self, item, value)

            # Compute weights
            if self._target_dim > 1:  # If we have more than one target dimension,
                # we first factorize the matrix
                self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
                lu, piv = linalg.lu_factor(self.A)
                for i in range(self._target_dim):
                    self.nodes[:, i] = linalg.lu_solve((lu, piv), self.di[:, i])
            else:
                self.nodes = linalg.solve(self.A, self.di)

        @property
        def A(self):
            # this only exists for backwards compatibility: self.A was available
            # and, at least technically, public.
            r = squareform(pdist(self.xi.T, self.norm))  # Pairwise norm
            return self._init_function(r) - np.eye(self.N)*self.smooth

        def _call_norm(self, x1, x2):
            return cdist(x1.T, x2.T, self.norm)

        def __call__(self, *args):
            args = [np.asarray(x) for x in args]
            if not all([x.shape == y.shape for x in args for y in args]):
                raise ValueError("Array lengths must be equal")
            if self._target_dim > 1:
                shp = args[0].shape + (self._target_dim,)
            else:
                shp = args[0].shape
            xa = np.asarray([a.flatten() for a in args], dtype=np.float_)
            r = self._call_norm(xa, self.xi)
            return np.dot(self._function(r), self.nodes).reshape(shp)

    
    def __init__(self, 
                 func               = 'linear',
                 epsilon            = None,
                 smooth             = 0,
                 norm               = 'euclidean',
                 mode               = '1-D',
                 ):
         
                 self.func          = func    
                 self.epsilon       = epsilon            
                 self.smooth        = smooth             
                 self.norm          = norm               
                 self.mode          = mode               
    
    def predict(self, X_test):
        return self.model(*X_test.T)
        
    def fit(self, X_train, y_train):
       model = self.Rbf(*X_train.T, y_train.ravel(),
                       func    = self.func    ,
                       epsilon = self.epsilon , 
                       smooth  = self.smooth  , 
                       norm    = self.norm    , 
                       mode    = self.mode    ,  
                      )
       self.model= model
       return self 
        
    def get_params(self, deep = False):
        p={
                'func'    : self.func    ,
                'epsilon' : self.epsilon , 
                'smooth'  : self.smooth  , 
                'norm'    : self.norm    , 
                'mode'    : self.mode    , 
          }
        return p
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)            
        return self













 
class RBFNN(BaseEstimator, RegressorMixin):
    
    def lhsu(self, xmin,xmax,nsample):
        nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
        for j in range(nvar):
            idx=np.random.permutation(nsample)
            P =(idx.T-ran[:,j])/nsample
            s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
            
        return s

    def __init__(self, n_hidden=20, epsilon=1e-4, random_state=None):
        self.random_state   = random_state
        self.n_hidden       = n_hidden
        self.epsilon        = epsilon
        np.random.seed(self.random_state)
        
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        r = np.linalg.norm(c-d)
        return np.exp(-self.epsilon * r**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.n_hidden), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def fit(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
        Y=Y.reshape(-1,1)  
        self.indim          = X.shape[1]
        self.outdim         = Y.shape[1]
        self.centers        = self.lhsu( [-1]*self.indim, [+1]*self.indim, self.n_hidden) 
        self.W              = np.random.random((self.n_hidden, self.outdim))
        # choose random center vectors from training set
        rnd_idx = np.random.permutation(X.shape[0])[:self.n_hidden]
        self.centers = [X[i,:] for i in rnd_idx]         
        #print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print(G)         
        # calculate output weights (pseudoinverse)
        self.W = np.dot(np.linalg.pinv(G), Y)        
         
    def predict(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y
   
    def get_params(self, deep = False):
        p={
            'n_hidden'          : self.n_hidden,
            'epsilon'              : self.epsilon, 
            'random_state'      : self.random_state, 
          }
        return p
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)            
        return self
      









if __name__ == '__main__':
    
    from sklearn.datasets import load_diabetes, make_regression
    from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error  
    from sklearn.model_selection import train_test_split 
    import pandas as pd
    import seaborn as sns
    import pylab as pl
    
    X, y = load_diabetes(return_X_y=True)
    #X, y = make_regression(n_samples=600, n_features=10, noise=0.3)
    
    X_train, X_test, y_train, y_test = train_test_split(
                             X, y, test_size=0.33, random_state=42)

    y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)
  
    np.random.seed(0)

    clf = RBFNNRegressor()
    
    clf.fit(X_train, y_train)
    print(clf.predict(X_test), y_test.ravel())
    print(clf.get_params())
    
    p={
        'func'      : 'linear',
        'epsilon'   : 0.2,
        'smooth'    : 0.1,
        'norm'      : 'euclidean',
        'mode'      : '1-D'
      }
    
    clf.set_params(**p)
    print(clf.get_params())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test).ravel()
    l = {          
            'R2'    : r2_score(y_test, y_pred),
            'MSE'   : mean_squared_error(y_test, y_pred),
            #'MSLE'  : mean_squared_log_error(y_test, y_pred),
            }
    print(l)   
            
    #aux=[]
    #epsilon=1e-3
    #for k in np.linspace(5,20,16):
        #n_hidden=int(k)
        #print(n_hidden)
        #for random_state in range(20):
            #p={
                #'n_hidden'          : n_hidden,
                #'epsilon'              : epsilon, 
                #'random_state'      : random_state, 
              #}
            #clf = RBFNN()
            #clf.set_params(**p)
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test).ravel()
            #l = {
                  #'HN'    : n_hidden,
                  #'R2'    : r2_score(y_test, y_pred),
                  #'MSE'   : mean_squared_error(y_test, y_pred),
                  ##'MSLE'  : mean_squared_log_error(y_test, y_pred),
                 #}
            #aux.append(l)
            
    #A = pd.DataFrame(aux)
    #print(A.groupby('HN').agg(np.mean))
    #sns.factorplot( col='variable', y='value', x='HN', data=A.melt(id_vars=['HN']), kind='bar', sharey=False,); 
    #pl.show()
    
