#ARBOL DE CLASIFICACION----------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train2.csv') #ya procesados los datos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.predict(sc.transform([[30,87000]])))

y_pred = classifier.predict(X_test)


#Accuracy y matriz de confusion
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)



#ARBOL DE REGRESION----------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train2.csv') #ya procesados los datos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')

plt.show()



#ARBOL DE RANDOM FOREST CLASIFICACION----------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train2.csv') #ya procesados los datos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


print(classifier.predict(sc.transform([[30,87000]])))

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)



#ARBOL DE REGRESION RANDOM FOREST----------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train2.csv') #ya procesados los datos
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')

plt.show()



X_test = sc.transform(X_test)