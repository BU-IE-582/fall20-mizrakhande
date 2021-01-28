# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:46:32 2021

@author: hande
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn import preprocessing

data=pd.read_csv("energydata_complete.csv")

names=data.columns[1:28]
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(data.iloc[:,1:28])
data = pd.DataFrame(d, columns=names)

features=data.columns[1:28]
X = data.loc[:,features]
y = data.loc[:,'Appliances'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Model Performance')
    print('Improved RMSE Error:',sqrt(mean_squared_error(test_labels,predictions)))


## Decision Tree
model1 = DecisionTreeRegressor(random_state=(0))
model1.fit(X_train, y_train)
y_test_pred=model1.predict(X_test)
print('RMSE for base model DT:',sqrt(mean_squared_error(y_test,y_test_pred)))

##Parameter Tuning
parameters = {'max_depth':[2,3,8,10], 'min_samples_leaf':[2,3,4,5]}

tuning1 = GridSearchCV(estimator = DecisionTreeRegressor(random_state = 10), 
            param_grid = parameters,n_jobs=4, cv=5)
tuning1.fit(X_train,y_train)

print(tuning1.best_params_)
best_model1 = tuning1.best_estimator_
best_model_accuracy1 = evaluate(best_model1, X_test,y_test)



##Random Forest
model2 = RandomForestRegressor(random_state=(0))
model2.fit(X_train, y_train)
y_test_pred2=model2.predict(X_test)
print('RMSE for base model RF:',sqrt(mean_squared_error(y_test,y_test_pred2))) 

import math
a=len(X.columns)
b=round(math.sqrt(len(X.columns)))
c=round((len(X.columns))/3)
parameters = {'max_features': [a,b,c]}

tuning2 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=500,min_samples_leaf=5, random_state=0), 
            param_grid = parameters,n_jobs=4, cv=5)
tuning2.fit(X_train,y_train)

print(tuning2.best_params_)

best_model2 = tuning2.best_estimator_
best_model_accuracy2 = evaluate(best_model2, X_test,y_test)


##PRA
alphas = np.logspace(-4, 1, 10)
lassocv = linear_model.LassoCV(alphas=alphas,cv=5, random_state=0, max_iter = 2000)
lassocv.fit(X_train, y_train)
lassocv_score_on_train = lassocv.score(X_train, y_train)
lassocv_score_on_test = lassocv.score(X_test, y_test)
lassocv_alphas = lassocv.alphas_
lassocv_alpha = lassocv.alpha_
best_lasso = linear_model.Lasso(alpha=lassocv_alpha)
best_lasso.fit(X_train, y_train)
y_test_pred3=best_lasso.predict(X_test)
print('RMSE for PRA:',sqrt(mean_squared_error(y_test,y_test_pred3))) 


##SGB

model4 = GradientBoostingRegressor(random_state=(0))
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)
print("RMSE for base model SGB:",sqrt(mean_squared_error(y_test,y_test_pred3)))


##Parameter Tuning
parameters = {'learning_rate':[0.2,0.15,0.1,0.05,0.01], 'n_estimators':[100,250,500,750]}

tuning3= GridSearchCV(estimator = model4, 
            param_grid = parameters,n_jobs=4, cv=5)
tuning3.fit(X_train,y_train)

print(tuning3.best_params_)

max_depth = {'max_depth':[2,3,4,5] }
tuning4 = GridSearchCV(estimator =GradientBoostingRegressor(learning_rate = tuning3.best_params_['learning_rate'], n_estimators = tuning3.best_params_['n_estimators'], random_state=0), 
            param_grid = max_depth,n_jobs=4, cv=5)
tuning4.fit(X_train,y_train)
print(tuning4.best_params_)

best_model4 = GradientBoostingRegressor(learning_rate = tuning3.best_params_['learning_rate'], n_estimators = tuning3.best_params_['n_estimators'], max_depth = tuning4.best_params_['max_depth'],random_state=0)
best_model4.fit(X_train,y_train)
best_accuracy = evaluate(best_model4, X_test,y_test)