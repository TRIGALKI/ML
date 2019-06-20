import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import sklearn.preprocessing
import sklearn.datasets
from sklearn.neighbors import*
a = sklearn.datasets.load_boston()
X = sklearn.preprocessing.scale(a.data)
y = a.target
acc = []
KV = KFold(n_splits=5, shuffle=True, random_state=42)
p_range = np.linspace(1, 10, 100)
for p in p_range:
    reg = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance', metric="minkowski")
    res = cross_val_score(estimator=reg, X=X, y=y, cv=KV, scoring="neg_mean_squared_error")
    acc.append((p, np.mean(res)))
print(max(acc, key=lambda x: x[1]))


