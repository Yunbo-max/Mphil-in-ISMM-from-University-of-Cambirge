# question
# (i) Discuss which methods are appropriate for developing a model to predict number of defects in any given machine?
# (ii) Develop a model to predict number of defects in any given machine. Describe and justify each step that led you to the construction of your final model. List your assumptions.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D

df1 = pd.read_csv(r"./Data.csv")
df1.hist(bins=20,figsize=(10,10))
plt.show()

#
df1 = df1.astype(float)
df1.info()
data_df1 = pd.DataFrame(df1)
print(data_df1.head(30))
col1 = df1.columns

for col in col1:
    f, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    sns.regplot(x=col, y='DEFECTS', data=df1, ax=ax,ci=99)
    x = ax.get_xlabel()
    y = ax.get_ylabel()
    ax.set_xlabel(x, fontsize=18)
    ax.set_ylabel(y, fontsize=18)
    plt.show()


for col in col1:
    f, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    sns.boxplot(x=df1[col],y='DEFECTS', data=df1, ax=ax)
    x = ax.get_xlabel()
    y = ax.get_ylabel()
    ax.set_xlabel(x, fontsize=24)
    ax.set_ylabel(y, fontsize=24)

    plt.show()


corr = df1.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(200, 10, center='light',as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,vmin=-1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .4}, annot=True)

plt.show()





plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

hs2 = df1.drop(['DEFECTS'],axis=1)
col1 = df1.columns
col2 = hs2.columns
combine = pd.DataFrame(itertools.combinations(col2, 2))
for i in range(len(combine)):
    fig = plt.figure(figsize=(10,6))
    ax = Axes3D(fig)
    x=df1[combine[0][i]]
    y=df1[combine[1][i]]
    z=df1['DEFECTS']
    ax.scatter(x,y,z)
    plt.title("三维分析"+combine[0][i]+"-"+combine[1][i]+"-"+"DEFECTS",fontsize=18)
    ax.set_xlabel(combine[0][i],fontsize=14)
    ax.set_ylabel(combine[1][i],fontsize=14)
    ax.set_zlabel('DEFECTS',fontsize=14)
    plt.tick_params(labelsize=10)
    plt.show()
#




