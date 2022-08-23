import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

ones = np.ones(N).reshape((-1,1))
Xb = np.concatenate((ones, X), axis=1)

W = np.random.randn(D+1)
z = Xb.dot(W)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learning_rate = 0.1
l2 = 0.1
for i in range(100):
    if i%10==0:
        print(cross_entropy(T,Y))
    
    W -= learning_rate * np.dot((Y-T), Xb) - l2*W
    Y = sigmoid(Xb.dot(W))


print("final weight:", W)

# plot the data and seperating line
plt.scatter(X[:,0],X[:,1],c=T,s=100,alpha=0.5)

x_axis = np.linspace(-6,6,100)
y_axis = -(W[1]*x_axis + W[0])/W[2]
plt.plot(x_axis,y_axis)
plt.show()