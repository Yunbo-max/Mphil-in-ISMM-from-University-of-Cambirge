import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Data_final.csv")
# df = pd.read_csv("Data")

# df.info()

from statsmodels.formula.api import ols
import statsmodels.api as sm


lm = ols('DEFECTS ~ MAINTENANCE + AGE + TOOL', data=df).fit()
# lm = ols('DEFECTS ~ MAINTENANCE + TECHNICIANS + AGE + TOOL', data=df).fit()
print(lm.summary())

fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid( lm, fig=fig)
plt.show()