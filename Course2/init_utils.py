import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets

# ---------------- Activation Functions ---------------- #
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def initialize(layer_dims, seed=3):
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # He initialization for weights (good for ReLU)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        # Ensure shapes are correct
        assert parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1])
        assert parameters['b' + str(l)].shape == (layer_dims[l], 1)

    return parameters


# ---------------- Forward Propagation ---------------- #
def forward_propagation(X, parameters, activation='relu'):
    caches = {}
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        w = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        z = np.dot(w, A) + b
        A = relu(z) if activation == 'relu' else sigmoid(z)
        caches['A' + str(l)] = A
        caches['Z' + str(l)] = z

    # last layer: sigmoid 
    wl = parameters['W'+ str(L)]
    bl = parameters['b' + str(L)]
    zl = np.dot(wl, A) + bl
    Al = sigmoid(zl)
    caches['A' + str(L)] = Al
    caches['Z' + str(L)] = zl
    return Al, caches

# ---------------- Backward Propagation ---------------- #
def backward_propagation(x, y, caches, parameters, activation='relu'):
    grads = {}
    m = x.shape[1]
    L = len(parameters) // 2
    Al = caches['A' + str(L)]

    eps = 1e-15
    dAl = - (np.divide(y, Al + eps) - np.divide(1 - y, 1 - (Al + eps)))

    # last layer
    zl = caches['Z' + str(L)]
    wl = parameters['W' + str(L)]
    dZl = dAl * Al * (1 - Al) # sigmoid backward it is.
    A_prev = caches['A' + str(L - 1)]
    dWl = (1 / m) * np.dot(dZl, A_prev.T)
    dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
    grads['dW' + str(L)] = dWl
    grads['db' + str(L)] = dbl
    dA_prev = np.dot(wl.T, dZl)

    # hidden layers
    for l in reversed(range(1, L)):
        z = caches['Z' + str(l)]
        w = parameters['W' + str(l)]
        A_prev = caches['A' + str(l - 1)] if l > 1 else x
        if activation == 'relu':
            dZ = np.array(dA_prev, copy=True)
            dZ[z <= 0] = 0
        else:
            A = caches['A' + str(l)]
            dZ = dA_prev * A * (1 - A)
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
        if l > 1:
            dA_prev = np.dot(w.T, dZ)

    return grads


# ---------------- Update Parameters ---------------- #
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters


# ---------------- Predict ---------------- #
def predict(x, parameters):
    Al, _ = forward_propagation(x, parameters)
    return (Al > 0.5).astype(int)

# ---------------- Compute Loss ---------------- #
def compute_loss(Al, y):
    m = y.shape[1]
    loss = - (1/m) * np.sum(y*np.log(Al+1e-8) + (1-y)*np.log(1-Al+1e-8))
    return np.squeeze(loss)

# ---------------- Visualization ---------------- #
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()

def predict_dec(parameters, X):
    preds = predict(X, parameters)
    return preds.ravel() # flatten to 1D for reshaping


# ------------- for Lab 1 -------------- #
def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=0.2)
    test_X, test_Y = sklearn.datasets.make_moons(n_samples=100, noise=0.2)

    # reshape to match (features, examples)
    train_X = train_X.T
    train_Y = train_Y.reshape(1, train_Y.shape[0])
    test_X = test_X.T
    test_Y = test_Y.reshape(1, test_Y.shape[0])

    return train_X, train_Y, test_X, test_Y
