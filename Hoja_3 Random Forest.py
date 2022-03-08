import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import sklearn.mixture as mixture
import pyclustertend 
import random
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from kneed import KneeLocator
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
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


normal = normal.dropna()
CN = normal.columns.values


H = normal
X = np.array(normal)
X.shape
print('Hopkins', pyclustertend.hopkins(X,len(X)))

km = cluster.KMeans(n_clusters=3).fit(X)
centroides = km.cluster_centers_
#print(centroides)


normal = km.predict(X)
plt.scatter(X[normal == 0, -1], X[normal == 0, -1],s=100,c='red', label = "Cluster 1")
Cluster_bajo = X[normal == 0, -1]
Cluster_bajo = Cluster_bajo.tolist()
print('máximo primer cluster (rojo)', max(Cluster_bajo))
print('mínimo primer cluster (rojo)',min(Cluster_bajo))
plt.scatter(X[normal == 1, -1], X[normal == 1, -1],s=100,c='blue', label = "Cluster 2")
Cluster_alto = X[normal == 1, -1]
Cluster_alto = Cluster_alto.tolist()
print('máximo segundo cluster (azul)', max(Cluster_alto))
print('mínimo segundo cluster (azul)',min(Cluster_alto))
plt.scatter(X[normal == 2, -1], X[normal == 2, -1],s=100,c='green', label = "Cluster 3")
Cluster_medio = X[normal == 2, -1]
Cluster_medio = Cluster_medio.tolist()
print('máximo tercer cluster (verde)', max(Cluster_medio))
print('mínimo tercer cluster (verde)',min(Cluster_medio))
plt.scatter(km.cluster_centers_[:,-1],km.cluster_centers_[:,-1], s=300, c="yellow",marker="*", label="Centroides")
plt.title("Grupo casa")
plt.xlabel("Precio de venta")
plt.ylabel("Calidad de la casa")
plt.legend()



"""
clf = RandomForestClassifier(n_estimators=100, max_depth=4)
clf.fit(X_train, y_train)
estimator = clf.estimators_[5]

plt.figure()
_ = tree.plot_tree(clf.estimators_[0], feature_names=X.columns, filled=True)
plt.show()

y_pred = clf.predict(X_test)
#print(y_pred)
print ("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall: ", metrics.recall_score(y_test,y_pred,average='weighted'))"""

clf = RandomForestRegressor(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)
estimator = clf.estimators_[5]

plt.figure()
_ = tree.plot_tree(clf.estimators_[0], feature_names=X.columns, filled=True)
plt.show()

y_predf = clf.predict(X_train)

print ("MSE:",metrics.mean_squared_error(y_train, y_predf))

y_pred = clf.predict(X_test)

print ("MSE:",metrics.mean_squared_error(y_test, y_pred))