# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:49:39 2021

@author: hande
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import math

filename = 'dermatology_data.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=str)
data= pd.DataFrame(data)
missing_values=np.where(data.iloc[:,:]=='?')
data=data.drop(missing_values[0])
data.astype('int')


features=data.columns[0:34]
X = data.loc[:,features]
y = data.loc[:,34].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Model Performance')
    print('Improved Accuracy = {:0.2f}%.',metrics.accuracy_score(test_labels, predictions))

#Decision Tree
model1 = DecisionTreeClassifier(random_state=0)
model1.fit(X_train, y_train)
y_test_pred=model1.predict(X_test)
#Score the model
print("Accuracy for base model DT:",metrics.accuracy_score(y_test, y_test_pred))


parameters = {'max_depth':[2,3,8,10], 'min_samples_leaf':[2,3,4,5]}

tuning1 = GridSearchCV(estimator =DecisionTreeClassifier(random_state=10), 
            param_grid = parameters, scoring='accuracy',n_jobs=4, cv=5)
tuning1.fit(X_train,y_train)

print(tuning1.best_params_)
best_model = tuning1.best_estimator_
random_accuracy1 = evaluate(best_model, X_test,y_test)

##Random Forest
model2=RandomForestClassifier(random_state=0)
model2.fit(X_train, y_train)
y_train_pred2=model2.predict(X_train)
y_test_pred2=model2.predict(X_test)

# Score the model
print("Accuracy for RF:",metrics.accuracy_score(y_test, y_test_pred2))

a=len(X.columns)
b=round(math.sqrt(len(X.columns)))
c=round((len(X.columns))/3)
parameters = {'max_features': [a,b,c]}

tuning2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=500,min_samples_leaf=5, random_state=0), 
            param_grid = parameters, scoring='accuracy',n_jobs=4, cv=5)
tuning2.fit(X_train,y_train)

print(tuning2.best_params_)

best_random = tuning2.best_estimator_
random_accuracy2 = evaluate(best_random, X_test,y_test)


##PRA
C = np.logspace(-4, 1, 10) 
for c in C:
    model3 = LogisticRegression(penalty='l1', C=c, solver='liblinear')
    model3.fit(X_train, y_train)
    #y_pred=model3.predict(X_test)
    print('C:', c)
    #print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy for RPA:', model3.score(X_train, y_train))
    print('Test accuracy for RPA:', model3.score(X_test, y_test))
    print('')


##SGB
model4 = GradientBoostingClassifier(random_state=(0))
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)
print("Accuracy for base model SGB:",metrics.accuracy_score(y_test, y_pred))

parameters = {'learning_rate':[0.2,0.15,0.1,0.05,0.01], 'n_estimators':[100,250,500,750]}

tuning = GridSearchCV(estimator =GradientBoostingClassifier(max_features='sqrt', random_state=10), 
            param_grid = parameters, scoring='accuracy',n_jobs=4, cv=5)
tuning.fit(X_train,y_train)

print(tuning.best_params_)

max_depth = {'max_depth':[2,3,4,5] }
tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.01,n_estimators=250,max_features='sqrt', random_state=10), 
            param_grid = max_depth, scoring='accuracy',n_jobs=4, cv=5)
tuning.fit(X_train,y_train)
print(tuning.best_params_)

best_model = GradientBoostingClassifier(learning_rate=0.01,n_estimators=250,max_features='sqrt',max_depth=3,random_state=10)
best_model.fit(X_train,y_train)
best_accuracy = evaluate(best_model, X_test,y_test)
