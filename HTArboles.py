import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import sklearn.mixture as mixture
import random
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import sys
import warnings
import scipy.stats as stats
import pylab
from scipy.stats import shapiro
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

Data = pd.read_csv('train.csv')

Test = pd.read_csv('test.csv')

#print(Data)

normal = Data.select_dtypes(include = np.number)

#print(normal.describe())

normal = normal.drop(['Id', 'LowQualFinSF', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'MoSold', 'YrSold', 'MSSubClass', 'OverallCond', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'WoodDeckSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'LotFrontage', 'LotArea', 'MasVnrArea', '1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'OpenPorchSF' ], axis = 1)

correlation_mat = normal.corr()

NC = normal.columns.values

SP = correlation_mat.iloc[-1]

SaleP = normal[['SalePrice']]

#print(SP)

#variables = correlation_mat.query("sector == 'SalePrice & ")
#print(correlation_mat)


#sns.heatmap(correlation_mat, annot = True)
normal = normal.dropna()
CN = normal.columns.values

#print(CN)





# In[268]:

"""
n = 0
while n < 20:
    for i in CN:
        normal = normal[(normal[i] < normal[i].mean()+2*(normal[i].std())) & (normal[i] > normal[i].mean()-2*(normal[i].std()))] 
        n += 1


fig = plt.figure()
g = 0
for i in CN:
	plt.subplot(5,3,g+1)
	sns.distplot(normal[i])
	plt.xlabel(i)
	g += 1

"""
#normal = normal[['OverallQual', 'TotalBsmtSF', 'GrLivArea' ,'FullBath', 'GarageCars', 'SalePrice']]
H = normal
X = np.array(normal)
X.shape
print('Hopkins', pyclustertend.hopkins(X,len(X)))
