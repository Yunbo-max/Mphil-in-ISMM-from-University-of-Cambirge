



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import tkinter as tk

df_dm = pd.read_csv(r"Data.csv")
train_data_dm, test_data_dm = train_test_split(df_dm, train_size=0.8, random_state=3)


features = ['TOOL', 'MAINTENANCE', 'AGE']
complex_model_R = linear_model.Ridge(alpha=100)
complex_model_R.fit(train_data_dm[features], train_data_dm['DEFECTS'])

pred1 = complex_model_R.predict(test_data_dm[features])
intercept = float(complex_model_R.intercept_)
coef = list(complex_model_R.coef_)

print('Ridge Regression with three Features: TOOL, MAINTENANCE, AGE')
print('Coefficients: {}'.format(coef))
print('Intercept: {}'.format(intercept))
print('y = '+str(coef)+'x + '+str(intercept))
# 计算模型评分
print('R-squared is ')
print(complex_model_R.score(df_dm[features], df_dm['DEFECTS']))


#
#
# #
#
# #
# #
#
df_dm = pd.read_csv(r"./Data.csv")
train_data_dm, test_data_dm = train_test_split(df_dm, train_size=0.8, random_state=3)


features = ['TOOL', 'MAINTENANCE', 'AGE','TECHNICIANS']
complex_model_R = linear_model.Ridge(alpha=100)
complex_model_R.fit(train_data_dm[features], train_data_dm['DEFECTS'])


pred1 = complex_model_R.predict(test_data_dm[features])
intercept = float(complex_model_R.intercept_)
coef = list(complex_model_R.coef_)
print('\n')
print('Ridge Regression with Four Features: TOOL, MAINTENANCE, AGE,TECHNICIANS')
print('Coefficients: {}'.format(coef))
print('Intercept: {}'.format(intercept))
print('y = '+str(coef)+'x + '+str(intercept))
# 计算模型评分
print('R-squared is ')
print(complex_model_R.score(df_dm[features], df_dm['DEFECTS']))


