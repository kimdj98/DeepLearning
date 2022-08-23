import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x, y = float(x), float(y)
    X.append([1, x, x**2])
    Y.append(y)


X = np.array(X)
Y = np.array(Y)
# print(X)
# print(Y)
# print(X.shape)
# print(Y.shape)
# plt.xscale('linear')
# plt.yscale('linear')
# plt.scatter(X, Y)
# plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w) # X * w

# plot it all together
plt.scatter(X[:,1], Yhat)
plt.scatter(X[:,1], Y)
plt.show()