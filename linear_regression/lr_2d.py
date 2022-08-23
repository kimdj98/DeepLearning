import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the data
X = []
Y = []
for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

# turn X and Y into numpy arrays
X=np.array(X)
Y=np.array(Y)

# let's plot the data to see what the data looks like

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(X[:,2], X[:,1], Y)
# plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
Yhat = np.dot(X,w)

# compute r-square
d1 = Y - Yhat
d2 = Y - np.mean(Y)

r2 = 1 - d1.dot(d1) / d2.dot(d2)
r2