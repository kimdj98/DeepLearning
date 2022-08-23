import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

df = pd.read_excel('mlr02.xls', engine='xlrd')
X = df.values
print(X)
X = np.array(X)

# look for relations between X2, X1
plt.scatter(X[:,1], X[:,0])
plt.show()

# look for relations between X3, X1
plt.scatter(X[:,2], X[:,0])
plt.show()

# look for relations between X3, X2
plt.scatter(X[:,2], X[:,1])
plt.show()

Y = df['X1']
df['ones'] = 1
df['random'] = np.random.randn(11)
df
X = df[['X2', 'X3','ones']]
Y = np.array(Y)
X = np.array(X)

# plt.scatter(X[:,1],Y)
# plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = X.dot(w)

d1 = Y - Yhat #SSR
d2 = Y - np.mean(Y) #SST

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(f'r_square: {r2}')