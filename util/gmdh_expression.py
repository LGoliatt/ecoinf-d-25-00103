
from gmdhpy.gmdh import MultilayerGMDH

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
                
      
class GMDH(BaseEstimator, RegressorMixin):

    def __init__(self,
                ref_functions= 'linear',
                #criterion_type= 'test_bias',
                seq_type= 'random',
                #feature_names= feature_names,
                min_best_neurons_count=1, 
                criterion_minimum_width=1,
                admix_features= False,
                max_layer_count=5,
                stop_train_epsilon_condition= 0.0001,
                layer_err_criterion= 'top',
                #alpha= 0.5,
                normalize=False,
                l2= 0.1,
                n_jobs= 1,
                 ):

                self.ref_functions                   = ref_functions                
                #self.criterion_type                  = criterion_type               
                self.seq_type                        = seq_type                     
                #self.feature_names                   = feature_names               
                self.min_best_neurons_count          = min_best_neurons_count       
                self.criterion_minimum_width         = criterion_minimum_width      
                self.admix_features                  = admix_features               
                self.max_layer_count                 = max_layer_count              
                self.stop_train_epsilon_condition    = stop_train_epsilon_condition 
                self.layer_err_criterion             = layer_err_criterion          
                #self.alpha                           = alpha                        
                self.normalize                       = normalize                    
                self.l2                              = l2                           
                self.n_jobs                          = n_jobs                       
        
                self.params = {
                    'ref_functions'                   : self.ref_functions               ,               
                    #'criterion_type'                  : self.criterion_type              ,               
                    'seq_type'                        : self.seq_type                    ,               
                    #'feature_names'                   : self.feature_names               ,               
                    #'min_best_neurons_count'          : self.min_best_neurons_count      ,               
                    'criterion_minimum_width'         : self.criterion_minimum_width     ,               
                    'admix_features'                  : self.admix_features              ,               
                    'max_layer_count'                 : self.max_layer_count             ,               
                    'stop_train_epsilon_condition'    : self.stop_train_epsilon_condition,               
                    'layer_err_criterion'             : self.layer_err_criterion         ,               
                    #'alpha'                           : self.alpha                       ,               
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


if __name__ == '__main__':
        
    from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(GEP2())
    
    my_estimator = GMDH(ref_functions=('quadratic', 'linear', 'linear_cov', 'cubic'),)
    my_estimator = GMDH(ref_functions=('quadratic', 'linear', 'linear_cov', ),)
    for param, value in my_estimator.get_params(deep=True).items():
        print(f"{param} -> {value}")
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import pylab as pl
    
    X,y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    feature_names=['x'+str(i) for i in range(X_train.shape[1])]
    
    my_estimator.fit(X_train,y_train, verbose=True)
    y_pred = my_estimator.predict(X_test)
    pl.plot(y_test, 'r-', y_pred,  'b-')
    
    print(my_estimator.get_params())
    print(my_estimator.get_params())
    
    #%%
    #from gmdhpy.plot_model import PlotModel
    from gmdhpy.plot_gmdh import PlotGMDH
    model=my_estimator.estimator
    param=my_estimator.get_params()
    param['alpha']=param['l2']
    param.pop('l2')
    param.pop('min_best_neurons_count')
    model=MultilayerGMDH(**param)

    model.fit(X_train,y_train,)
    y_pred = model.predict(X_test)
    pl.plot(y_test, 'r-', y_pred,  'b-')

    #PlotModel(model, filename='model_house_model', plot_neuron_name=True, view=True).plot()
    PlotGMDH(model, filename='model_house_model', plot_model_name=True, view=True)

    #%%
    # def get_features_name(neuron,input_index, feature_names, layers):
    #         if neuron.layer_index == 0:
    #             s = '{0}'.format(input_index)
    #             s = ''
    #             if len(feature_names) > 0:
    #                 s += '{0}'.format(feature_names[input_index])
    #         else:
    #             neurons_num = len(layers[neuron.layer_index-1])
    #             if input_index < neurons_num:
    #                 s = 'prev_layer_neuron_{0}'.format(input_index)
    #             else:
    #                 s = 'inp_{0}'.format(input_index - neurons_num)
    #                 s = ''
    #                 if len(feature_names) > 0:
    #                     s += '{0}'.format(feature_names[input_index - neurons_num])
    #         return s
        
    # expression=[]
    # import sympy as sp
    # sp.init_printing(forecolor='Yellow')
    
    # print('\n'*10+'='*80+'\n GMDH Expression\n'+'='*80+'\n'*2)
    # s=''
    # for layer in model.layers:
    #     p=layer.describe(feature_names, model.layers)
    #     print(p)
    #     print('-'*80,'\n')
    #     s+='# '+str(layer)+'\n'
    #     for neuron in layer: 
    #         print(
    #               neuron,"|",
    #               neuron.neuron_index, "|",
    #               neuron.fw_size,"|",
    #               neuron.ftype, "|",
    #               neuron.u1_index, "|",
    #               neuron.u2_index,"|",
    #               )
    #         inp=[None,None]
    #         inp[0]=neuron.get_features_name(neuron.u1_index, feature_names, model.layers)
    #         inp[1]=neuron.get_features_name(neuron.u1_index, feature_names, model.layers)
    #         x=[None,None]
    #         x[0]=get_features_name(neuron,neuron.u1_index, feature_names, model.layers)
    #         x[1]=get_features_name(neuron,neuron.u2_index, feature_names, model.layers)
    
    #         print(x)
    #         w=neuron.w
    #         sp.var(['w'+str(i) for i in range(len(w))])
    #         coeff=';'.join(['w'+str(i)+'='+str(w[i]) for i in range(len(w))])
    #         coeff_dict={'w'+str(i):w[i] for i in range(len(w))}
    #         print(w)
    #         expr=neuron.get_name()
    #         print(expr)
    #         prev='prev_layer_neuron_'+str(neuron.neuron_index)
    #         print('-'*4,end='\n')
            
    #         sp.var('u1,u2')
            
    #         ftype=neuron.__str__().split(' - ')[1]
    #         s+='u1='+str(x[0])+';'+'u2='+str(x[1])+';\n'
    #         prev=neuron.transfer(u1,u2,w)
    #         s+='prev_layer_neuron_'+str(neuron.neuron_index)+'='+str(prev)+'\n'
           
    # s+='\noutput = prev_layer_neuron_0;\n'
    # s+='#display(output); \n'
    # print(s)
    # with open("output.py", "w") as text_file:
    #     text_file.write(s)
    #%%
    expression=[]
    import sympy as sp
    sp.init_printing(forecolor='Yellow')

    print('\n'*10+'='*80+'\n GMDH Expression\n'+'='*80+'\n'*2)
    s=''
    sp.var(feature_names)
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
           
           
    s+='\noutput = prev_layer_neuron_0;\n'
    s+='#display(output); \n'
    print(s)
    #%%        
    sp.var(feature_names)
    exec(open("output.py").read())
    #sp.parsing.sympy_parser.parse_expr(s)
    
    for i in range(len(X_test)):
        d=dict(zip(feature_names,X_test[i]))
        print(y_test[i],y_pred[i], output.subs(d))


#%%   
