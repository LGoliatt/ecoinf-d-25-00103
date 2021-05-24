### Sine training example for ffnet ###

import numpy as np
import pylab as pl
from math import pi, sin, cos
from ffnet import ffnet, mlgraph, tmlgraph, imlgraph
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from  multiprocessing import cpu_count

class MLPRegressor(BaseEstimator, RegressorMixin):
  """Multi-layer perceptron (feedforward neural network) classifier.

  Trained with gradient descent under the loss function which is estimated
  for each sample batch at a time and the model is updated along the way
  with a decreasing strength schedule (aka learning rate).

  Please note that this implementation uses one hidden layer only.

  The regularizer is a penalty added to the loss function that shrinks model
  parameters towards the zero vector.

  This implementation works with data represented as dense and sparse numpy
  arrays of floating point values for the features.

  Parameters
  ----------
  n_hidden : int, default 10
      Number of units in the hidden layer.
      Ex.: 
	  [5]    : one hidden layer  with 5 neurons
	  [5,5]  : two hidden layers with 5 neurons each one
	  [3,3,3]: three hidden layers with 5 neurons each one
	  
  activation : default 'logistic'
      Activation function for the hidden layer.

	- 'logistic' for 1 / (1 + exp(x)).

  algorithm : {'tnc', 'l-bfgs', 'sgd', 'rprop'}, default 'tnc'
      The algorithm for weight optimization.  Defaults to 'l-bfgs'

      - 'l-bfgs' is an optimization algorithm in the family of quasi-
	  Newton methods.

      - 'sgd' refers to stochastic gradient descent.
      
      - 'tnc' refers to gradient information in a truncated Newton algorithm

      - 'rprop' Rprop training algorithm.

  bias : bool, optional
    Indicates if bias (node numbered 0) should be added to hidden
    and output neurons. Default is *True*.
    
  renormalize : bool
        If *True* normalization ranges will be recreated from
        training data at next training call.
      
  connectivity : {'mlgraph', 'tmlgraph'}, default 'mlgraph'

      -'mlgraph'  : Creates standard multilayer network architecture.

      -'tmlgraph' : Creates multilayer network full connectivity list.
		    Similar to `mlgraph`, but now layers are fully connected with all
		    preceding layers.

  max_iter : int, optional, default None
      Maximum number of iterations. The algorithm
      iterates until this number of iterations.

  """
  def __init__(
    self, n_hidden=10, bias=True, activation="logistic",
    algorithm='tnc', renormalize=True, connectivity='mlgraph',
    max_iter=None, 
    ):
    
    #super(BaseEstimator, self).__init__()
    
    self.activation = activation
    self.algorithm = algorithm
    self.n_hidden = n_hidden
    self.bias = bias
    self.renormalize=renormalize
    self.connectivity=connectivity
    self.max_iter=max_iter
    self.classes_ = None
    
    


  def fit(self, X, y):
    """Fit the model to the data X and target y.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
	Training data, where n_samples in the number of samples
	and n_features is the number of features.

    y : numpy array of shape (n_samples)
	Subset of the target values.

    Returns
    -------
    self
    """
    n_samples, n_features = X.shape
    try:
      n_outputs=y.shape[1]
    except:
      n_outputs=1

    #conec = mlgraph((n_features,self.n_hidden,1))
    self.n_outputs=n_outputs
    par=np.r_[n_features, self.n_hidden,self.n_outputs]
    if self.connectivity=='mlgraph':
      conec = mlgraph(par, biases=self.bias)
    else:
      if self.connectivity=='tmlgraph':
          conec = tmlgraph(par, biases=self.bias)
      else:
          return 'Connectivity does not defined.'
      
    net = ffnet(conec)
    net.renormalize = self.renormalize
    if self.algorithm == 'tnc':
      net.train_tnc(X, y, maxfun = self.max_iter, messages=0)
    if self.algorithm == 'sgd':
      net.train_cg(X, y, maxiter = self.max_iter, disp=False)
    if self.algorithm == 'l-bfgs':
      net.train_bfgs(X, y, maxfun = self.max_iter, disp=False)
    if self.algorithm == 'rprop':
      net.train_rprop(X, y, disp=False)
    if self.algorithm == 'genetic':
      net.train_genetic(X, y)

    self.model = net
    return self
    
  def predict(self, X):
    """Predict using the multi-layer perceptron model.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    array, shape (n_samples)
	Predicted target values per element in X.
    """
    y_pred = self.model.call(X)
    if self.n_outputs == 1:
      return y_pred.ravel()
    else:
      return y_pred


  def score(self, X, y):
        """Force use of accuracy score since we don't inherit
           from ClassifierMixin"""

        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
##
##
##

class MLPClassifier(MLPRegressor):
  """Multi-layer perceptron (feedforward neural network) classifier.

  Trained with gradient descent under the loss function which is estimated
  for each sample batch at a time and the model is updated along the way
  with a decreasing strength schedule (aka learning rate).

  Please note that this implementation uses one hidden layer only.

  The regularizer is a penalty added to the loss function that shrinks model
  parameters towards the zero vector.

  This implementation works with data represented as dense and sparse numpy
  arrays of floating point values for the features.

  Parameters
  ----------
  n_hidden : int, default 10
      Number of units in the hidden layer.
      Ex.: 
	  [5]    : one hidden layer  with 5 neurons
	  [5,5]  : two hidden layers with 5 neurons each one
	  [3,3,3]: three hidden layers with 5 neurons each one
	  
  activation : default 'logistic'
      Activation function for the hidden layer.

	- 'logistic' for 1 / (1 + exp(x)).

  algorithm : {'tnc', 'l-bfgs', 'sgd', 'rprop'}, default 'tnc'
      The algorithm for weight optimization.  Defaults to 'l-bfgs'

      - 'l-bfgs' is an optimization algorithm in the family of quasi-
	  Newton methods.

      - 'sgd' refers to stochastic gradient descent.
      
      - 'tnc' refers to gradient information in a truncated Newton algorithm

      - 'rprop' Rprop training algorithm.

  bias : bool, optional
    Indicates if bias (node numbered 0) should be added to hidden
    and output neurons. Default is *True*.
    
  renormalize : bool
        If *True* normalization ranges will be recreated from
        training data at next training call.
      
  connectivity : {'mlgraph', 'tmlgraph'}, default 'mlgraph'

      -'mlgraph'  : Creates standard multilayer network architecture.

      -'tmlgraph' : Creates multilayer network full connectivity list.
		    Similar to `mlgraph`, but now layers are fully connected with all
		    preceding layers.

  max_iter : int, optional, default None
      Maximum number of iterations. The algorithm
      iterates until this number of iterations.

  """
  
  def __init__(
	  self, n_hidden=10, bias=True, activation="logistic",
	  algorithm='tnc', renormalize=False, connectivity='mlgraph',
	  max_iter=None,
	  ):

    super(MLPClassifier, self).__init__(n_hidden=n_hidden,
				      bias=bias, activation=activation,
				      algorithm=algorithm,
				      renormalize=renormalize,
				      connectivity=connectivity,
				      max_iter=max_iter,
				      )

    self.classes_ = None
    self.binarizer = LabelBinarizer(0., 1.)


  def decision_function(self, X):
      """
      This function return the decision function values related to each
      class on an array of test vectors X.

      Parameters
      ----------
      X : array-like of shape [n_samples, n_features]

      Returns
      -------
      C : array of shape [n_samples, n_classes] or [n_samples,]
	  Decision function values related to each class, per sample.
	  In the two-class case, the shape is [n_samples,]
      """
      return super(MLPClassifier, self).predict(X)

  def fit(self, X, y):
      """
      Fit the model using X, y as training data.

      Parameters
      ----------
      X : {array-like, sparse matrix} of shape [n_samples, n_features]
	  Training vectors, where n_samples is the number of samples
	  and n_features is the number of features.

      y : array-like of shape [n_samples, n_outputs]
	  Target values (class labels in classification, real numbers in
	  regression)

      Returns
      -------
      self : object

	  Returns an instance of self.
      """
      self.classes_ = np.unique(y)

      y_bin = self.binarizer.fit_transform(y)

      super(MLPClassifier, self).fit(X, y_bin)

      return self

  def predict(self, X):
      """
      Predict values using the model

      Parameters
      ----------
      X : {array-like, sparse matrix} of shape [n_samples, n_features]

      Returns
      -------
      C : numpy array of shape [n_samples, n_outputs]
	  Predicted values.
      """
      dim_=False
      if X.ndim==1:
          dim_=True
          X = np.array([list(X)])
   
      raw_predictions = self.decision_function(X)
      class_predictions = self.binarizer.inverse_transform(raw_predictions)
	
      return class_predictions[0] if dim_ else class_predictions

  def score(self, X, y):
      """Force use of accuracy score since we don't inherit
	  from ClassifierMixin"""

      from sklearn.metrics import accuracy_score
      return accuracy_score(y, self.predict(X))

##
##
##
if __name__ == "__main__":

  from sklearn import cross_validation
  from sklearn import datasets
  from sklearn import grid_search
  from time import time
  from sklearn import preprocessing
  from sklearn import metrics
  from MLP import MLPRegressor, MLPClassifier
  ##
  ##
  ## 
  estimators=[
    ('MLP',
    MLPRegressor(),
    {'n_hidden':[[5],[5,5],[5,5,5],[5,5,5,5],[10],[20],[30],],
      #'algorithm':['tnc', 'l-bfgs', 'sgd', 'rprop', 'genetic'],
      'algorithm':['tnc'],
      'bias':[True, False], 
      'connectivity':['mlgraph','tmlgraph'],
      'renormalize':[True,False],
      }
    ),
    ]
    
  X,y = datasets.make_friedman2()
  n_samples, n_features = X.shape
  for clf_name, clf, param_grid in estimators:
    print (clf_name)
    for k in range(10):
      np.random.seed(k)
      X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4,random_state=k)
      gs = grid_search.GridSearchCV(clf, param_grid=param_grid)  
      gs.fit(X_train, y_train)
      best_clf=gs.best_estimator_
      y_true, y_pred = y_test, best_clf.predict(X_test)
      print (k, '\t',metrics.r2_score(y_true, y_pred), '\t', gs.best_params_)
      
  ##
  ##
  ##
  
  
  classifiers=[
    ('MLP',
    MLPClassifier(),
    {'n_hidden':[[5],[5,5],[5,5,5],[5,5,5,5],[10],[20],[30],[50]],
      #'algorithm':['tnc', 'l-bfgs', 'sgd', 'rprop', 'genetic'],
      'algorithm':['tnc'],
      'bias':[True, False], 
      'connectivity':['mlgraph','tmlgraph'],
      'renormalize':[True,False],
      }
    ),
    ]
    
  
  iris	= datasets.load_iris()
  X, y	= iris.data, iris.target
  n_samples, n_features = X.shape
  for clf_name, clf, param_grid in classifiers:
    #print clf_name
    for k in range(10):
      np.random.seed(k)
      X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4,random_state=k)
      gs = grid_search.GridSearchCV(clf, param_grid=param_grid)  
      gs.fit(X_train, y_train)
      best_clf=gs.best_estimator_
      y_true, y_pred = y_test, best_clf.predict(X_test)
      #print k, '\t',metrics.accuracy_score(y_true, y_pred), '\t', gs.best_params_
   
  #clf=MLPClassifier()
  #clf.fit(X,y)
  #k=np.random.randint(len(X))
  #print clf.predict(X[k]), '\t', y[k]
  #print clf.predict(X)
  
  #classes_ = np.unique(y)
  #binarizer = LabelBinarizer(0, 1)
  #y_bin = binarizer.fit_transform(y)
  
  #reg = MLPRegressor(renormalize=False)
  #reg.fit(X, y_bin)
  
  #k=np.random.randint(len(X))
  #print reg.predict(X[k]), '\n', y_bin[k]
    
  #conec = mlgraph( (2,3,3,3,1) )
  #net = ffnet(conec)
  #input = [ [0.,0.], [0.,1.], [1.,0.], [1.,1.] ]
  #target  = [ [1.], [0.], [0.], [1.] ]
  #net.train_tnc(input, target, maxfun = 1000)
  #net.test(input, target, iprint = 2)
  #savenet(net, "xor.net")
  #exportnet(net, "xor.f")
  #net = loadnet("xor.net")
  #answer = net( [ 0., 0. ] )
  #partial_derivatives = net.derivative( [ 0., 0. ] )



  #from ffnet import imlgraph, ffnet
  #import networkx as NX
  #import pylab

  #conec = imlgraph((3, [(3,), (6, 3), (3,)], 3), biases=False)
  #net = ffnet(conec)
  #NX.draw_graphviz(net.graph, prog='dot')
  #pylab.show()


  #conec = tmlgraph((2,6,1), biases=True); net = ffnet(conec);NX.draw_graphviz(net.graph, prog='dot');pylab.show()
