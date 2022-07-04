
from gmdhpy.gmdh import Regressor
from gmdhpy.gmdh import MultilayerGMDH
# Load the diabetes dataset
  
from sklearn.utils.estimator_checks import check_estimator
#check_estimator(GEP2())

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pylab as pl

X,y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


feature_names=['x'+str(i) for i in range(X_train.shape[1])]
params = {
    'ref_functions': ('quadratic', 'linear', 'linear_cov', 'cubic'),
    #'criterion_type': 'test_bias',
    'feature_names': feature_names,
    'criterion_minimum_width': 5,
    'admix_features': True,
    'max_layer_count': 3,
    'normalize': False,
    'stop_train_epsilon_condition': 0.0001,
    'layer_err_criterion': 'top',
     #'alpha': 0.5,
    'n_jobs': 4
}
model = MultilayerGMDH(**params)

#model = MultilayerGMDH(ref_functions=('quadratic', 'linear', 'linear_cov', 'cubic'),)




model.fit(X_train,y_train, verbose=True)
y_pred = model.predict(X_test)
pl.figure()
pl.plot(y_test, 'r-', y_pred,  'b-')
pl.figure()
pl.plot(y_test,y_test, 'r-', y_test,y_pred,  'bo')
pl.show()

#%%
from gmdhpy.plot_model import PlotModel
PlotModel(model, filename='model_house_model', plot_neuron_name=True, view=True).plot()
#%%
def get_features_name(neuron,input_index, feature_names, layers):
        if neuron.layer_index == 0:
            s = '{0}'.format(input_index)
            s = ''
            if len(feature_names) > 0:
                s += '{0}'.format(feature_names[input_index])
        else:
            neurons_num = len(layers[neuron.layer_index-1])
            if input_index < neurons_num:
                s = 'prev_layer_neuron_{0}'.format(input_index)
            else:
                s = 'inp_{0}'.format(input_index - neurons_num)
                s = ''
                if len(feature_names) > 0:
                    s += '{0}'.format(feature_names[input_index - neurons_num])
        return s
    
expression=[]
import sympy as sp
sp.init_printing(forecolor='Yellow')

print('\n'*10+'='*80+'\n GMDH Expression\n'+'='*80+'\n'*2)
s=''
for layer in model.layers:
    p=layer.describe(feature_names, model.layers)
    print(p)
    print('-'*80,'\n')
    s+='# '+str(layer)+'\n'
    for neuron in layer: 
        print(
              neuron,"|",
              neuron.neuron_index, "|",
              neuron.fw_size,"|",
              neuron.ftype, "|",
              neuron.u1_index, "|",
              neuron.u2_index,"|",
              )
        inp=[None,None]
        inp[0]=neuron.get_features_name(neuron.u1_index, feature_names, model.layers)
        inp[1]=neuron.get_features_name(neuron.u1_index, feature_names, model.layers)
        x=[None,None]
        x[0]=get_features_name(neuron,neuron.u1_index, feature_names, model.layers)
        x[1]=get_features_name(neuron,neuron.u2_index, feature_names, model.layers)

        print(x)
        w=neuron.w
        sp.var(['w'+str(i) for i in range(len(w))])
        coeff=';'.join(['w'+str(i)+'='+str(w[i]) for i in range(len(w))])
        coeff_dict={'w'+str(i):w[i] for i in range(len(w))}
        print(w)
        expr=neuron.get_name()
        print(expr)
        prev='prev_layer_neuron_'+str(neuron.neuron_index)
        print('-'*4,end='\n')
        
        sp.var('u1,u2')
        
        ftype=neuron.__str__().split(' - ')[1]
        s+='u1='+str(x[0])+';'+'u2='+str(x[1])+';\n'
        prev=neuron.transfer(u1,u2,w)
        s+='prev_layer_neuron_'+str(neuron.neuron_index)+'='+str(prev)+'\n'
       
s+='\noutput = prev_layer_neuron_0;\n'
s+='#display(output); \n'
print(s)
with open("output.py", "w") as text_file:
    text_file.write(s)
#%%        
sp.var(feature_names)
exec(open("output.py").read())
#sp.parsing.sympy_parser.parse_expr(s)


for i in range(len(X_test)):
    d=dict(zip(feature_names,X_test[i]))
    print(y_test[i],y_pred[i], output.subs(d))
    
display(output.expand())
print(sp.latex(output))