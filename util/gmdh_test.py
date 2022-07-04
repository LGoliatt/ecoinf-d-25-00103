
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#from gmdhpy.gmdh import MultilayerGMDH,modelessor
#gmdh = MultilayerGMDH()

from GMDH import GMDHRegressor
from gmdhpy.gmdh import Regressor
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
#diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
test_size=40
diabetes_X_train = diabetes_X[:-test_size]
diabetes_X_test = diabetes_X[-test_size:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-test_size]
diabetes_y_test = diabetes_y[-test_size:]

# Create linear modelession object
model = linear_model.LinearRegression()

feature_names = ['x'+str(i+1) for i in range(diabetes_X.shape[1])]
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
model = GMDHRegressor(**params)
#model = Regressor(**params)

# Train the model using the training sets
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = model.predict(diabetes_X_test)

# The coefficients
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
tit="Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred)
print(tit)

# Plot outputs
plt.scatter(diabetes_y_test, diabetes_y_pred, color="black")
plt.plot(diabetes_y_test, diabetes_y_test, color="blue", linewidth=3)
plt.title(tit)
plt.xticks(())
plt.yticks(())
plt.show()

print(model.describe())


#%%
layers=[]
for layer in model.layers:
    #print(layer)
    p=layer.describe(feature_names, model.layers)
    
    p=p.split('\nPolynomModel ')
    for i in range(1,len(p)):
        #print(p[i])
        x=p[i]
        poly_model = x.split('\n')[0].split(' - ')[1]
        u1=x.split('\n')[1].split('index=')[1].split(',')[-1]
        u2=x.split('\n')[2].split('index=')[1].split(',')[-1]
        print(i, poly_model, u1, u2)
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
s+='display(output); \n'
print(s)
with open("output.py", "w") as text_file:
    text_file.write(s)
#%%        
sp.var(feature_names)
exec(open("output.py").read())
#sp.parsing.sympy_parser.parse_expr(s)

for i in range(len(diabetes_y_test)):
    d=dict(zip(feature_names,diabetes_X_test[i]))
    print(diabetes_y_test[i],diabetes_y_pred[i], output.subs(d))


#%%