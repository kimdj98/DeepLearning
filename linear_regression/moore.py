import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')

    x = int(re.sub(non_decimal,'',r[2].split('[')[0])) # substitute pattern -> '' in r[2].split sth...
    y = int(re.sub(non_decimal,'',r[1].split('[')[0]))
    
    X.append(x)
    Y.append(y)

X = np.array(X) - 1970
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

Y = np.log(Y)
plt.scatter(X,Y)
plt.show()

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
print(np.linspace(0,50,100))
# plot the data
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()

# calculate r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r_square = 1 - np.sum(d1 ** 2) / np.sum(d2 ** 2)
print(r_square)