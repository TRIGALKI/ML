import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_test = pd.read_csv('Data\perceptron-test.csv',header=None)
df_train = pd.read_csv('Data\perceptron-train.csv',header=None)
y_tr = df_train[0]
X_tr = df_train.loc[:, 1:]
y_ts = df_test[0]
X_ts = df_test.loc[:, 1:]
clf = Perceptron(random_state=241)
clf.fit(X_tr, y_tr)
pred_no_scale=clf.predict(X_ts)
sc = accuracy_score(y_ts,pred_no_scale )
X_tr_scaled = scaler.fit_transform(X_tr)
X_ts_scaled = scaler.transform(X_ts)
clf1 = Perceptron(random_state=241)
clf1.fit(X_tr_scaled, y_tr)
pred_with_scale=clf1.predict(X_ts_scaled)
sc1 = accuracy_score(y_ts, pred_with_scale)
ot = sc1-sc
print(ot)
