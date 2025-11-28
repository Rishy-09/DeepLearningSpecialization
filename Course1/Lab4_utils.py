import numpy as np
import h5py
import matplotlib.pyplot as plt


np.random.seed(1)

def initialize(n_x, n_h, n_y):

    np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # Correct assertions
    assert(w1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(w2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    params = {'w1': w1,
              'b1': b1, 
              'w2': w2,
              'b2': b2}

    return params


def initialize_deep(layer_dims):
    # layer_dims: it is a list containing dimensions of each layer in our network
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for i in range(1, L):
        # -->>> This is the corrected He initialization <<<--
        params['w' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2 / layer_dims[i-1])
        params['b' + str(i)] = np.zeros((layer_dims[i], 1))

        assert(params['w' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert(params['b' + str(i)].shape == (layer_dims[i], 1))

    return params

# building the linear part
def linear_forward(A, W, b):

    z = np.dot(W, A) + b

    assert(z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return z, cache

def sigmoid(z):
    A = 1 / (1 + np.exp(-z))

    cache = z
    return A, cache

def relu(z):
    A = np.maximum(0, z)
    cache = z
    return A, cache

def sigmoid_backward(da, cache):
    """
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' stored from forward propagation 
    """
    z = cache
    s = 1 / (1+np.exp(-z))
    dz = da * s * ( 1- s)
    return dz

def relu_backward(da, cache):
    z = cache
    dz = np.array(da, copy=True) # converting to correct obj.
    dz[z <= 0] = 0 # gradient be 0 if z <= 0.
    return dz

def linear_activation_forward(A_prev, W, b, activation):

    z, l_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(z)

    elif activation == 'relu':
        A, activation_cache = relu(z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (l_cache, activation_cache)
    
    return A, cache

# implementing the L-Layer Model
def model_propagate(x, params):
    caches = []
    A = x
    L = len(params) // 2
    for i in range(1, L):
        A_prev = A
        w = params['w' + str(i)]
        b = params['b' + str(i)]
        A, cache = linear_activation_forward(A_prev, w, b, "relu")
        caches.append(cache)

    w = params['w' + str(L)]
    b = params['b' + str(L)]
    Al, cache = linear_activation_forward(A, w, b, "sigmoid")
    caches.append(cache)

    assert(Al.shape == (1, x.shape[1]))
    return Al, caches

def compute_cost(Al, y):
    m = y.shape[1]

    cost =  -(1/m) *  np.sum(np.multiply(y, np.log(Al)) + np.multiply((1-y), np.log(1- Al)))
    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost


def linear_backward(dz, cache):
    A_prev, w, b, = cache
    m = A_prev.shape[1]

    dw = 1/m * np.dot(dz, A_prev.T)
    db = 1/m * np.sum(dz, axis=1, keepdims=True)
    dA_prev = np.dot(w.T, dz)

    assert (dA_prev.shape == A_prev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)
    return dA_prev, dw, db

def linear_activation_backward(da, cache, activation):
    l_cache, a_cache = cache
    if(activation == "relu"):
        dz = relu_backward(da, a_cache)
        dA_prev, dw, db = linear_backward(dz, l_cache)

    elif activation == "sigmoid":
        dz = sigmoid_backward(da, a_cache)
        dA_prev, dw, db = linear_backward(dz, l_cache)

    return dA_prev, dw, db

def L_model_backward(al, y, cache):
    grads = {}
    l = len(cache) # the no. of layers
    m = al.shape[1]
    y = y.reshape(al.shape)

    dal = - (np.divide(y, al) - np.divide(1-y, 1-al)) # derivative of cost with respect to al

    curr_cache = cache[l-1]
    grads['da'+str(l-1)], grads['dw'+str(l)], grads['db'+str(l)] = linear_activation_backward(dal, curr_cache, "sigmoid")

    for i in reversed(range(l-1)):
        curr_cache = cache[i]
        da_prev_temp, dw_temp, db_temp = linear_activation_backward(grads['da'+str(i+1)], curr_cache, "relu")
        
        grads['da'+str(i)] = da_prev_temp
        grads['dw'+str(i+1)] = dw_temp
        grads['db'+str(i+1)]=db_temp

    return grads

def update_params(params, grads, lr):
    for key in params.keys():  # 'w1', 'b1', 'w2', 'b2'
        dw_key = 'd' + key  # 'dw1', 'db1', etc.
        # print("The keys of grads are: ", grads.keys())
        if dw_key in grads:
            params[key] -= lr * grads[dw_key]
    return params

# quick unit test
np.random.seed(1)
layer_dims = [5,4,3,1]
params = initialize_deep(layer_dims)   # returns w1..w3, b1..b3
X = np.random.randn(5, 7)
Y = (np.random.rand(1,7) > 0.5).astype(int)

# forward
AL, caches = model_propagate(X, params)
print("AL.shape:", AL.shape, "expected (1,7)")
print("len(caches):", len(caches), "expected", len(params)//2)

# cost
c = compute_cost(AL, Y)
print("cost scalar ok:", np.isscalar(c))

# backward (sanity)
grads = L_model_backward(AL, Y, caches)
print("grads keys sample:", list(grads.keys())[:6])

# param update small step
new_params = update_params(params, grads, lr=0.01)
print("update done. W1 shape:", new_params['w1'].shape)

 