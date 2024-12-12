import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, LinearRegression as LR
from sklearn.metrics import r2_score, explained_variance_score as EVS, mean_squared_error as MSE
from sklearn.model_selection import train_test_split, cross_val_score
from pandas.core.accessor import register_dataframe_accessor

from statsmodels.formula.api import ols

data=pd.read_csv('Data_final.csv')
x = data[['MAINTENANCE','AGE','TOOL']]
y= data['DEFECTS']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

reg = LR().fit(x_train, y_train)
yhat1 = reg.predict(x_test)
yhat2 = reg.predict(x_train)

print('When test size parameter is 0.2')
print("For model training: r2 = ",r2_score(y_train,yhat2))
print("For model testing: r2 = ",r2_score(y_test,yhat1))



#
data=pd.read_csv('Data_final.csv')
x = data[['MAINTENANCE','AGE','TOOL']]
y= data['DEFECTS']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=1)

reg = LR().fit(x_train, y_train)
yhat1 = reg.predict(x_test)
yhat2 = reg.predict(x_train)
print('\n')
print('When test size parameter is 0.25')
print("For model training: r2 = ",r2_score(y_train,yhat2))
print("For model testing: r2 = ",r2_score(y_test,yhat1))