
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import tkinter as tk


df = pd.read_csv(r"./Data.csv")

train_data,test_data = train_test_split(df,train_size = 0.7, random_state=3)

X_train = np.array(train_data['MAINTENANCE'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['DEFECTS'], dtype=pd.Series)


X_test = np.array(test_data['MAINTENANCE'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['DEFECTS'], dtype=pd.Series)


lr = linear_model.LinearRegression()


lr.fit(X_train,y_train)


pred = lr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test,pred,color='r')
# plt.show()

X = np.array(df['MAINTENANCE']).reshape(-1,1)
# print(lr.score(X,df['DEFECTS']))


coef = float(lr.coef_)
intercept = float(lr.intercept_)
print('Feature 1: MAINTENANCE')
print('y = '+str(coef)+'x + '+str(intercept))

X = np.array(df['MAINTENANCE']).reshape(-1,1)
print('R-squared is ')
print(lr.score(X,df['DEFECTS']))
#

print('\n')


df = pd.read_csv(r"./Data.csv")

train_data,test_data = train_test_split(df,train_size = 0.7, random_state=3)

X_train = np.array(train_data['TECHNICIANS'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['DEFECTS'], dtype=pd.Series)

X_test = np.array(test_data['TECHNICIANS'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['DEFECTS'], dtype=pd.Series)


lr = linear_model.LinearRegression()


lr.fit(X_train,y_train)


pred = lr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test,pred,color='r')
# plt.show()

X = np.array(df['TECHNICIANS']).reshape(-1,1)
# print(lr.score(X,df['DEFECTS']))


coef = float(lr.coef_)
intercept = float(lr.intercept_)
print('Feature 2: TECHNICIANS')
print('y = '+str(coef)+'x + '+str(intercept))

X = np.array(df['TECHNICIANS']).reshape(-1,1)
print('R-squared is ')
print(lr.score(X,df['DEFECTS']))

print('\n')


df = pd.read_csv(r"./Data.csv")

train_data,test_data = train_test_split(df,train_size = 0.7, random_state=3)

X_train = np.array(train_data['AGE'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['DEFECTS'], dtype=pd.Series)


X_test = np.array(test_data['AGE'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['DEFECTS'], dtype=pd.Series)


lr = linear_model.LinearRegression()


lr.fit(X_train,y_train)


pred = lr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test,pred,color='r')
# plt.show()

X = np.array(df['AGE']).reshape(-1,1)
# print(lr.score(X,df['DEFECTS']))


coef = float(lr.coef_)
intercept = float(lr.intercept_)
print('Feature 3: AGE')
print('y = '+str(coef)+'x + '+str(intercept))

X = np.array(df['AGE']).reshape(-1,1)
print('R-squared is ')
print(lr.score(X,df['DEFECTS']))

print('\n')


df = pd.read_csv(r"./Data.csv")

train_data,test_data = train_test_split(df,train_size = 0.7, random_state=3)

X_train = np.array(train_data['TOOL'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['DEFECTS'], dtype=pd.Series)


X_test = np.array(test_data['TOOL'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['DEFECTS'], dtype=pd.Series)


lr = linear_model.LinearRegression()


lr.fit(X_train,y_train)


pred = lr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test,pred,color='r')
# plt.show()

X = np.array(df['TOOL']).reshape(-1,1)
# print(lr.score(X,df['DEFECTS']))


coef = float(lr.coef_)
intercept = float(lr.intercept_)
print('Feature 4: TOOL')
print('y = '+str(coef)+'x + '+str(intercept))

X = np.array(df['TOOL']).reshape(-1,1)
print('R-squared is ')
print(lr.score(X,df['DEFECTS']))