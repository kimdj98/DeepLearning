import numpy as np
import matplotlib.pyplot as plt

# make up some data and plot it
N = 100
X = np.linspace(0,6*np.pi, N)
Y = np.sin(X)

# plt.plot(X,Y)
# plt.show()

def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for i in range(deg):
        data.append(X**(i+1))
    return np.vstack(data).T

# train the weight
def fit(X,Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def fit_and_display(X, Y, sample, deg):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    Xtrain_poly = make_poly(Xtrain, deg)
    W = fit(Xtrain_poly,Ytrain)
    Yhat = np.dot(Xtrain_poly, W)
    print(W.shape)
    # plt.plot(X,Y)
    print(Xtrain)
    print(Yhat)
    plt.plot(X,Y)
    X_poly = make_poly(X, deg)
    Y2 = np.dot(X_poly, W)
    plt.scatter(Xtrain,Ytrain)
    plt.plot(X,Y2)
    plt.show()

fit_and_display(X,Y,10,8)

def get_mse(Y, Yhat):
    d = Y - Yhat
    return np.mean(d**2)

def plot_train_vs_test_curves(X, Y, samples = 20, max_deg=20): # notice that max_deg < samples
    N = len(X)
    train_idx = np.random.choice(N, samples)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    mse_trains = []
    mse_tests = []

    for deg in range(max_deg):
        test_idx = [idx for idx in range(N) if idx not in train_idx]
        Xtest = X[test_idx]
        Ytest = Y[test_idx]

        Xtrain_poly = make_poly(Xtrain, deg)
        Xtest_poly = make_poly(Xtest, deg)

        w = fit(Xtrain_poly, Ytrain)
        Ytrain_hat = np.dot(Xtrain_poly,w)
        Ytest_hat = np.dot(Xtest_poly,w)

        mse_trains.append(get_mse(Ytrain, Ytrain_hat))
        mse_tests.append(get_mse(Ytest, Ytest_hat))

    plt.plot(mse_trains, label = 'train mse')
    plt.plot(mse_tests, label = 'test mse')
    plt.legend()
    plt.show()

plot_train_vs_test_curves(X,Y, samples = 10, max_deg = 60)