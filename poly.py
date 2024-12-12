import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, LinearRegression as LR
from sklearn.metrics import r2_score, explained_variance_score as EVS, mean_squared_error as MSE
from sklearn.model_selection import train_test_split, cross_val_score
from pandas.core.accessor import register_dataframe_accessor

from statsmodels.formula.api import ols
df = pd.read_csv("Data")
# df = pd.read_csv("Data_final.csv")

# df.info()

from statsmodels.formula.api import ols
import statsmodels.api as sm



X = df[['MAINTENANCE','TECHNICIANS','AGE','TOOL']].values
# X = df[['MAINTENANCE','AGE','TOOL']].values
y = df[['DEFECTS']].values

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

X1 = np.array(x_train)
y1 = np.array(y_train)
X2 = np.array(x_test)
y2 = np.array(y_test)


poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(X1)
poly_features2 = poly.fit_transform(X2)

poly_regression = LinearRegression()
poly_regression.fit(poly_features,y1)

yhat = poly_regression.predict(poly_features2)


X = X[:,:-1]
# plt.scatter(X,y, color='red')
# plt.plot(X,poly_regression.predict(poly_features))
# plt.show()
print("For model training: r2 = ",poly_regression.score(poly_features,y1))
print("For model testing: r2 = ",poly_regression.score(poly_features2,yhat))

# plt.scatter(X,y, color='red')
plt.plot(X1,poly_regression.predict(poly_features))
plt.show()