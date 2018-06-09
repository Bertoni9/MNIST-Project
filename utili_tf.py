"""Created on Tue Mar 13 14:26:37 2018 """
import numpy as np
import h5py
 
def convert_to_one_hot(Y, C):
   Y = np.eye(C)[Y.reshape(-1)].T #To be cleared this command
   return Y


def random_mini_batches(X, Y, mb_size):
    m = X.shape[1]#Attention if I am using CNN, X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    n_complete_mb = m//mb_size #the last one will be shorter
    
    for i in range(n_complete_mb):
        mb_X = shuffled_X[:, i*mb_size : (i+1)*mb_size]
        mb_Y = shuffled_Y[:, i*mb_size : (i+1)*mb_size]
        mb_tuple = (mb_X, mb_Y)
        mini_batches.append(mb_tuple) # a list of tuples
        
    if m % mb_size != 0:
        mb_X = shuffled_X[:, m - mb_size*n_complete_mb : m]
        mb_Y = shuffled_Y[:, m - mb_size*n_complete_mb : m]
        mb_tuple = (mb_X, mb_Y)
        mini_batches.append(mb_tuple)
    
    return mini_batches

    