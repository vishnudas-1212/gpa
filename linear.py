import numpy as np
import pandas as pd
import pickle

df=pd.read_csv('sat.csv')
X = np.array(df[['SAT']])
y = np.array(df[['GPA']])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVR
gp = SVR(kernel='linear').fit(X_train,y_train)


pickle.dump(gp, open('score.pkl', 'wb'))
