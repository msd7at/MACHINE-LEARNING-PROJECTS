import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

sns.countplot(df_cancer['target'], label = "Count") 

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 

#MODEL TRAININGNM,./
X= df_cancer.drop(["target"],axis=1)

Y=df_cancer["target"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=5)

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, Y_train)

Y_predict = svc_model.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)

sns.heatmap(cm, annot=True)

min_train = X_train.min()
range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train)/range_train

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = Y_train)
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = Y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, Y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(Y_test, Y_predict)

sns.heatmap(cm,annot=True,fmt="d")
print(classification_report(Y_test,Y_predict))

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,Y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(Y_test, grid_predictions)
sns.heatmap(cm, annot=True)

print(classification_report(Y_test,grid_predictions))












