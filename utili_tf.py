"""Created on Tue Mar 13 14:26:37 2018 """
import numpy as np
 
def convert_to_one_hot(Y, C): #needed for binary classification
    if C == 1:
        C = 2
        Y = np.eye(C)[Y.reshape(-1)].T
    assert(Y.shape[0] == C)
    return Y

def random_mini_batches(X, Y, mb_size):
    m = X.shape[1]
    assert(m > mb_size)
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