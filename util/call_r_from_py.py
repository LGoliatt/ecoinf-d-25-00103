#%%
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

frbs = importr('frbs')

#%%
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
ts=robjects.r('ts')
forecast=importr('forecast')


import pandas as pd

from rpy2.robjects import pandas2ri
pandas2ri.activate()

traindf=pd.read_csv('UKgas.csv',index_col=0)
#traindf.index=traindf.index.to_datetime()
traindf.index=pd.to_datetime(traindf.index)

rdata=ts(traindf.value.values,frequency=4)
fit=forecast.auto_arima(rdata)
forecast_output=forecast.forecast(fit,h=16,level=(95.0))

index=pd.date_range(start=traindf.index.max(),periods=len(forecast_output[3])+1,freq='QS')[1:]
forecast=pd.Series(forecast_output[3],index=index)
lowerpi=pd.Series(forecast_output[4],index=index)
upperpi=pd.Series(forecast_output[5],index=index)

#%%