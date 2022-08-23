import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from preprocess import get_binary_data

X,Y = get_binary_data()
X,Y = shuffle(X,Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return - np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate*(pYtrain-Ytrain).dot(Xtrain)
    b -= learning_rate*(pYtrain-Ytrain).sum()
    if i % 1000 ==0:
        print(i, ctrain, ctest)

print(f'Final train classification_rate {classification_rate(Ytrain, np.round(pYtrain))}')
print(f'Final test classification_rate {classification_rate(Ytest, np.round(pYtest))}')

plt.plot(train_costs, label = 'train cost')
plt.plot(test_costs, label = 'test cost')
plt.legend()
plt.show()
