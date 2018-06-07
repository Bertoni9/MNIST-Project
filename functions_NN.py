"""Created on Tue Apr 24 17:11:32 2018 """
""" Be mindful, be a poet"""
from IPython import get_ipython; get_ipython().magic('clear')
import numpy as np; import tensorflow as tf; import matplotlib as plt
from utili_np import *
"""----------------------------------------------------------------------"""

#%% Activation Functions
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def softmax(Z):
    """
    Input: Z = (nx, m)  Output: A = (nx, m)
    """
    T = np.exp(Z - np.max(Z)) #np max is just a  trick for numerical stability, it cancels out
    sumt = np.sum(T, axis = 0, keepdims = True)
    A = T/sumt
    return A

def sigmoid_derivative(dA, Z):
    a = 1/(1+np.exp(-Z))
    dZ = dA * a * (1-a)
    assert (dZ.shape == Z.shape)
    return dZ

def relu_derivative(dA, Z):
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
                                #copy = True è default
    dZ[Z <= 0] = 0 # When z <= 0, you should set dz to 0. 
    assert (dZ.shape == Z.shape)
    return dZ

#def softmax_derivative(dA, Z):
#    return

#%% Parameters Initialization
def initialize_params(nx, ny, hidden_layers, existing = "No"):
    "Initialization of parameters W and b"
    
    #Loading existing parameters for cats dataset. 
    #Hyperparams: Alpha = 0.0075; N epochs = 2500; hidden layers =[20,7,5] .
    #In this case, expected Cost after iteration 0: 0.771749
    # Cost after iteration 2400: 0.092878. Training Accuracy: 0.985645933014 Test Accuracy: 0.8
    if existing == "Yes":
        file = np.load("init_params.npz")
        values = file['arr_0']
        params = values[()]
        #To save them:np.savez("init_params.npz", params) 

    else:
        #Creation of list with layer dimensions
        layer_dims = []
        layer_dims.append(nx)
        for i in range(len(hidden_layers)):
            layer_dims.append(hidden_layers[i])
        layer_dims.append(ny)
        assert(len(layer_dims) == len(hidden_layers)+2)
        
        params = {}
        L = len(layer_dims)
        for l in range(1,L):
            
            #Random Initialization
            params["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])#* 0.01
            params["b"+str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(params['W'+str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(params['b'+str(l)].shape == (layer_dims[l], 1))

    return params
    
#%% Forward Prop
def layer_FP(A_prev, W, b, activation):
    """
    >>> X = np.array([[-1.02387576, 1.12397796], [-1.62328545, 0.64667545],[-1.74314104, -0.59664964]])
    >>> W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    >>> b = 5
    >>> A1, cache = layer_FP(X, W, b, "relu")
    >>> A1
    array([[ 3.1980455 ,  7.85763489]])
    >>> A2, cache = layer_FP(X, W, b, "sigmoid")
    >>> A2
    array([[ 0.96076066,  0.99961336]])
    """
    Z = np.dot(W, A_prev) + b
    
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "softmax":
        A = softmax(Z)
    elif activation == "relu":
        A = relu(Z)
    
    cache = (A_prev, W, b, Z)
    return A, cache
        
def forward_prop(X, params):
    """
    >>> X = np.array([[-1.02387576, 1.12397796],[-1.62328545, 0.64667545],[-1.74314104, -0.59664964]])
    >>> params = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],[-1.07296862,  0.86540763, -2.3015387 ]]), \
                  'W2': np.array([[ 1.74481176, -0.7612069 ]]), 'b1': np.array([[ 0.], [ 0.]]),'b2': np.array([[ 0.]])}
    >>> A, caches = forward_prop(X, params)
    >>> A
    array([[ 0.0844367 ,  0.92356858]])
    """
    A = X
    L = len(params) // 2 # Number of parameters lead to the number of layers
    caches = []
    for l in range(1,L+1):
        A_prev = A
        if l != L:
            A, cache = layer_FP(A_prev, params["W"+str(l)], params["b"+str(l)],
                                           activation = "relu")
            caches.append(cache)
        else:
            A, cache = layer_FP(A_prev, params["W"+str(l)], params["b"+str(l)],
                                           activation = "sigmoid")
            caches.append(cache)
    
    
    return A, caches

#%% Cost function
def compute_cost(A, Y, params = [], lambd = 0):
    """
#    >>> Y = np.asarray([[1, 1, 1]])
#    >>> A = np.array([[.8,.9,0.4]])
#    >>> compute_cost(A, Y)
#    0.41493159961539694
    """
    m = Y.shape[1]
    layers = len(params)//2
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
#    if lambd != 0:
#        reg_cost = 0
#        for l in range(layers):
#            reg_cost += np.sum((params["W"+str(l+1)])**2)
#        reg_cost = reg_cost*lambd/(2*m)
#        cost = cost + reg_cost
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

#%% Backward Prop
def layer_BP(dA, cache, activation):
    """To compute a single layer BackProp"""
    
    A_prev, W, b, Z = cache
    
    if activation == "relu":
        dZ = relu_derivative(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_derivative(dA, Z)
        
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    return dA_prev, dW, db
    
def backward_prop(A, Y, caches):
    """ 
    Multi-layer Backprop. Tested with Gradient Checking
    >>> new_case = np.load("NN_example.npz")
    >>> NN_ex =new_case['arr_0']
    >>> params = NN_ex[()]["params"]
    >>> grads_ex = NN_ex[()]["grads"]
    >>> X = NN_ex[()]["X"]
    >>> Y = NN_ex[()]["Y"].reshape(1,3)
    >>> A, caches = forward_prop(X, params)
    >>> grads = backward_prop(A, Y, caches)
    >>> diff, _ = gradient_check(X, Y, params, grads, eps = 1e-7)
    >>> diff < 2e-7
    True
    
    """
    grads = {}
    L = len(caches)
    m = A.shape[1]
    Y = Y.reshape(A.shape)
    grads["dA"+str(L)] = - Y/A + (1-Y)/(1-A) #The first derivative: dJ/dA. I don't need J for that

    for l in reversed(range(L)):
        cache = caches[l]
        if l == L-1:
            d_A_prev, dW, db = layer_BP(grads["dA"+str(l+1)], cache, "sigmoid")
        else:
            d_A_prev, dW, db = layer_BP(grads["dA"+str(l+1)], cache, "relu")
        
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = d_A_prev, dW, db
        #In case of 1 layer I compute cost as f(A1, Y) then I compute dA1 before
        # and then I use bp to compute dZ1 and dW1, db1 and dA0(not useful in this case)
    return grads


def gradient_check(X, Y, params, grads, eps = 1e-7):
    
    theta = dict_to_vector(params) #to unroll the matrices of parameters in a vector
    d_theta = dict_to_vector(grads)#to unroll the matrices of gradients in a vector
    num_params = theta.shape[0] #length of the vector, same for dtheta
    J_plus = np.zeros((num_params,1))
    J_minus = np.zeros((num_params,1))
    d_theta_approx = np.zeros((num_params,1))
    
    for i in range(num_params): #at every loop I modify just one element
        theta_plus = np.copy(theta)  #Otherwise theta_plus will always remain equal to theta
        theta_minus = np.copy(theta) # Arrays are like lists: mutable
        theta_plus[i][0] += eps
        theta_minus[i][0] -= eps
        params_plus = vector_to_dict(theta_plus, params)
        params_minus = vector_to_dict(theta_minus, params)
        A_plus, _ = forward_prop(X, params_plus)
        A_minus, _ = forward_prop(X, params_minus)
        J_plus[i] = compute_cost(A_plus, Y)
        J_minus[i] = compute_cost(A_minus, Y)
        
        #Every element of d_theta_approx is the result of slight increase on a single value
        d_theta_approx[i] = (J_plus[i] - J_minus[i])/(2*eps)
        
    difference = np.linalg.norm(d_theta - d_theta_approx)/(
            np.linalg.norm(d_theta) + np.linalg.norm(d_theta_approx))
    
    return difference, d_theta_approx
    

#%% Gradient Descent
def update_params(params, grads, alpha):
    
    L = len(params)//2
    for l in range(1, L+1):
        params['W'+str(l)] = params['W'+str(l)] - alpha*grads['dW'+str(l)]
        params['b'+str(l)] = params['b'+str(l)] - alpha*grads['db'+str(l)]
    
    return params

   
#%%  Accuracy estimation
def predict(X, Y, params):
    "The function predicts the accuracy of a neural network with trained params"
    m = X.shape[1]
    P = np.zeros((1,m))
    Y_new= np.zeros((1,m)) #For multiclass prediction
    A, caches = forward_prop(X, params)
    assert(A.shape[1] == m)
    
    #Binary Case
    if A.shape[0] == 1:
        for i in range(m):
            if A[0,i] > 0.5:
                P[0,i] = 1
        acc = np.sum(P == Y) / m
    #Multi class case  
    else:
        for i in range(m):
            P[0,i] = np.argmax(A[:,i]) #I take the most confident of my units. The value is given by its position in the vector.
            Y_new[0,i] = np.argmax(Y[:,i])
        acc = np.sum(P == Y_new) / m
    
    return P, acc
    
#%% Doctest
if __name__ == "__main__":
    import doctest
    doctest.testmod() 
    

    
        
        
        
    
    
    
