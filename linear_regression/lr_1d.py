import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('data_1d.csv', header=None)
X = np.array(df[0])
Y = np.array(df[1])

# apply the equations we learned to calculate a and b
x_bar = np.mean(X)
y_bar = np.mean(Y)
xy_bar = np.mean(X*Y)
x_sqaure_bar = np.mean(X*X)

# calculate a
num_a = xy_bar - x_bar * y_bar
denom_a = x_sqaure_bar - x_bar ** 2
a = num_a / denom_a

# calculate b
num_b = xy_bar * x_bar - y_bar * x_sqaure_bar
denom_b = x_bar ** 2 - x_sqaure_bar
b = num_b / denom_b

Yhat = a*X + b

# plot the data
plt.scatter(X,Y)
plt.plot(a*np.linspace(0,100,100)+b)
plt.show()

# calculate r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r_square = 1 - np.sum(d1 ** 2) / np.sum(d2 ** 2)
print(r_square)