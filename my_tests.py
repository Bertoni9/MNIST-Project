"""Created on Wed May  9 15:39:08 2018 """
from IPython import get_ipython; get_ipython().magic('clear')
import numpy as np; import tensorflow as tf; import matplotlib.pyplot as plt
import pandas as pd; from utili_np import * ; from functions_NN import *
"""----------------------------------------------------------------------"""


#X_train, Y_train, X_test, Y_test, classes = load_cats_dataset()
#X_train, Y_train, X_test, Y_test, classes = load_MNIST_dataset()
    
#m = X_train.shape[1]
#nx = X_train.shape[0]
#ny = Y_train.shape[0]



#Softmax
#Z = np.array([[5, 2, -1, 3],[2, 1, 4, 5]]).T
#
#T = np.exp(Z - np.max(Z)) #np max hust for numerical stability, it cancels out
#sumt = np.sum(T, axis = 0, keepdims = True)
#A = T/sumt

#Test case loading an example with 3 layers and corrects grads
#X = (4,3), Y = (1,3), 3 Layers
#
#new_case = np.load("NN_example.npz")
#NN_ex =new_case['arr_0']
#params = NN_ex[()]["params"]
#grads_ex = NN_ex[()]["grads"]
#X = NN_ex[()]["X"]
#Y = NN_ex[()]["Y"].reshape(1,3)
#
#A, caches = forward_prop(X, params)
#grads = backward_prop(A, Y, caches)
#dW1 = grads["dW1"]
#dW1_ex = grads_ex["dW1"]
#
#diff, d_theta = gradient_check(X, Y, params, grads, eps = 1e-7)
#grads_apx = vector_to_dict(d_theta, grads_ex)
#dW1_apx = grads_apx["dW1"]
#
#
#file = np.load("init_params2.npz")
#values = file['arr_0']
#params2 = values[()]


#print(X_train[12,12])
##print(Y_train[0,12])
#A, caches = forward_prop(X_train, params)
#
#print(np.sum(Y_train))
#
#new_case = np.load("init_params.npz")
#NN_ex =new_case['arr_0']
#params_ex = NN_ex[()]
#
#A_ex, caches = forward_prop(X_train, params_ex)
#print(np.sum(A_ex))
#cost_ex = compute_cost(A_ex, Y_train.T)
#print(cost_ex)



