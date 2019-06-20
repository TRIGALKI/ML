import pandas as pd
from sklearn.model_selection import*
from sklearn.neighbors import*
from sklearn.preprocessing import*
import numpy as np

df=pd.read_csv('Data\wine.data', names=["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash"
, "Magnesium", "Total phenols", "Flavanoids"
, "Nonflavanoid phenols", "Proanthocyanins"
, "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline "])
y = df.loc[:, ["Class"]].values
X = df.drop(["Class"], axis=1).values
a = KFold(n_splits=5, shuffle=True, random_state=42)
acc = []
for i in range(1, 51):
    KNN=KNeighborsClassifier(n_neighbors=i)
    rs=[]
    for tr_ind, te_ind in a.split(X, y):
        X_train = X[tr_ind]
        y_train = y[tr_ind]
        X_test = X[te_ind]
        y_test = y[te_ind]
        KNN.fit(X_train, y_train)
        #res = KNN.score(X_test,y_test)
        #rs.append(res)
    res=cross_val_score(estimator=KNN, X=X, y=y, cv=a, scoring='accuracy')
    acc.append((i, np.mean(res)))
print(max(acc,key=lambda x:x[1]))
X_sc=scale(X)
acc=[]
for i in range(1, 51):
    KNN=KNeighborsClassifier(n_neighbors=i)
    rs=[]
    for tr_ind, te_ind in a.split(X_sc, y):
        X_train = X[tr_ind]
        y_train = y[tr_ind]
        X_test = X[te_ind]
        y_test = y[te_ind]
        KNN.fit(X_train, y_train)
        #res = KNN.score(X_test,y_test)
        #rs.append(res)
    res=cross_val_score(estimator=KNN, X=X_sc, y=y, cv=a, scoring='accuracy')
    acc.append((i , np.mean(res)))
print(max(acc,key=lambda x:x[1]))
