"""Created on Tue Apr 24 16:42:36 2018 """
import numpy as np
import h5py
import matplotlib.pyplot as plt
"""----------------------------------------------------------------------"""


def load_MNIST_dataset(n_train, n_val, n_test):
    """"Dataset of grey images divided in 2 datasets with 14000 training images
    and 1400 test images
    
    """
    assert(n_train + n_val <= 14000 and n_test <= 1400)
    # 1) Extracting the images from the dataset-------------------------------
    with h5py.File('datasets/MNIST_dataset_small.hdf5', 'r') as f_r:
        X_train_orig = np.array(f_r['training_set/X_train'][:])
        Y_train_orig = np.array(f_r['training_set/Y_train'][:])
        X_test_orig = np.array(f_r['training_set/X_val'][:])
        Y_test_orig = np.array(f_r['training_set/Y_val'][:])
        classes = [str(i) for i in range(Y_train_orig.shape[1])] #List of strings
    
    # 2) Printing an image
#    idx = 212
#    plt.imshow(X_train_orig[idx].reshape(28,28),cmap='binary')#to specify that I am working with black/white
#    #imshow accept 2D vectors or 3D vectors if the 3rd component is 3 or 4
#    print('y = ' + str(np.argmax(Y_train_orig[idx])))
    
   #3) Standardize the format-------------------------------------------------
    X_train = X_train_orig[:n_train,:,:,:] 
    Y_train = Y_train_orig[:n_train,:]
    X_val = X_train_orig[n_train : n_train + n_val, :]
    Y_val = Y_train_orig[n_train : n_train + n_val, :]
    X_test = X_test_orig[:n_test, :, :, :]
    Y_test = Y_test_orig[:n_test, :]
    
    #To explicitly compute FP I need X = (nx,m), since W = (nh1,nx) and Y = (ny,m)
    X_train = X_train.reshape(X_train.shape[0],-1).T
    X_val = X_val.reshape(X_val.shape[0],-1).T
    X_test = X_test.reshape(X_test.shape[0],-1).T
    Y_train = Y_train.T
    Y_val = Y_val.T
    Y_test = Y_test.T
    assert(Y_train.shape[0] == len(classes) and Y_train.shape[1] == n_train)
    assert(Y_val.shape[0] == len(classes) and Y_val.shape[1] == n_val)
    assert(Y_test.shape[0] == len(classes) and Y_test.shape[1] == n_test)
     
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, classes

def load_cats_dataset():
    
    """ Dataset of coloured images, divided in 2 datasets: training and test
    with 209 and 50 images respectively" Images are in utf8 format, for pixels
    """
    n_train = 209
    n_test = 50 #ALl images available in the dataset
    
    # 1) Extracting the images from the dataset-------------------------------
    
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    X_train_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    Y_train_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    X_test_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    Y_test_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    Y_train_orig = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
    Y_test_orig = Y_test_orig.reshape((1, Y_test_orig.shape[0]))
    
    #2) Decoding the names of the classes into strings and printing------------
    
    new_classes = [classes[i].decode("utf-8") for i in range(len(classes))]
    
    #3) printing an example, i.e. an image and its class (cat or non cat)
#    index = 10
#    plt.imshow(X_train_orig[index])
#    print ("y = " + str(Y_train_orig[0,index]) + ". It's a " +   \
#           new_classes[Y_train_orig[0,index]] +  " picture.") #classes it's just cat or non-cat
#    
    #4) Standardize the format------------------------------------------------
    
    X_train = X_train_orig[:n_train,:,:,:] 
    Y_train = Y_train_orig[:n_train,:]
    X_test = X_test_orig[:n_test, :, :, :]
    Y_test = Y_test_orig[:n_test, :]
    
    #To explicitly compute FP I need X = (nx,m), since W = (nh1,nx)
    X_train = X_train.reshape(X_train.shape[0],-1).T 
    X_test = X_test.reshape(X_test.shape[0],-1).T
    assert(Y_train.shape[0] == 1 and Y_train.shape[1] == n_train)
    assert(Y_test.shape[0] == 1 and Y_test.shape[1] == n_test)
    
    #Standardize data (Just the X!) to have feature values between 0 and 1
    assert(np.max(X_train) == 255)
    X_train = X_train/255.
    X_test = X_test/255.
    
    return X_train, Y_train, X_test, Y_test, new_classes


def dict_to_vector(params):
    """
    (dict of parameters)--> 1D vector
    >>> W1 = np.ones((3,3))*1
    >>> b1 = np.ones((3,1))*2
    >>> W2 = np.ones((2,2))*3
    >>> b2 = np.ones((2,1))*4
    >>> params = {'W2':W2,'b1':b1, 'W1':W1, 'b2':b2}
    >>> theta = dict_to_vector(params)
    >>> np.sum(theta)
    35.0
    """
    #First I need to make sure to access the dictionary in the same order
    #Different names in case of grads or params. In case of grads I consider only dW and db
    assert ("W1" in params or "dW1" in params)
    ordered_keys = []
    
    if "W1" in params:
        n_layers = len(params)//2
        for l in range(1, n_layers + 1):
            ordered_keys.append('W'+str(l))
            ordered_keys.append('b'+str(l))
    elif "dW1" in params:
        n_layers = (len(params)-1)//3 #Grads include also dA0 and dA1 ecc
        for l in range(1, n_layers + 1): 
            ordered_keys.append('dW'+str(l))
            ordered_keys.append('db'+str(l))
        
    count = 0
    for key in ordered_keys:
        new_vector = params[key].reshape(-1,1)#So that is always (x,1)
        if count==0:
            theta = new_vector
        else:
            theta = np.concatenate((theta,new_vector),axis=0) 
        count += 1
    
    return theta

def vector_to_dict(theta, params):
    """
    (vector, dict)---> dict
    To transform back the updated vector of parameters into matrix form
    >>> W1 = np.random.rand(3,3)
    >>> b1 = np.random.rand(3,1)
    >>> W2 = np.random.rand(2,2)
    >>> b2 = np.random.rand(2,1)
    >>> params = {'W1':W1,'b2':b2, 'W2':W2, 'b1':b1}
    >>> theta = dict_to_vector(params)
    >>> new_params = vector_to_dict(theta,params)
    >>> Aa = np.sum(np.sum(new_params['W1'] - params['W1']))
    >>> Bb = np.sum(np.sum(new_params['b1'] - params['b1']))
    >>> Aa
    0.0
    >>> Bb
    0.0
    """
    #First I need to make sure to access the dictionary in the same order.
    assert ("W1" in params or "dW1" in params)
    ordered_keys = []
    
    if "W1" in params:
        n_layers = len(params)//2
        for l in range(1, n_layers + 1):
            ordered_keys.append('W'+str(l))
            ordered_keys.append('b'+str(l))
    elif "dW1" in params:
        n_layers = (len(params)-1)//3 #Grads include also dA0 and dA1 ecc
        for l in range(1, n_layers + 1): 
            ordered_keys.append('dW'+str(l))
            ordered_keys.append('db'+str(l))
    
    new_params = {}
    idx = 0
    for key in ordered_keys:
        row = params[key].shape[0] #Extract dimension of original matrix
        col = params[key].shape[1]
        vector = theta[idx : row*col + idx]#Extraxt values of a specific matrix
        new_params[key] = vector.reshape(row,col) #Reshape in the original form
        
        idx += row*col #Update the index
        
    return new_params

def printing_mislabeled_images(X, Y, P, classes, num_images = 5):
    """
    The example indicates how np.where works
    >>> P = np.array([1,1,0,0,0,1]).reshape(1,-1)
    >>> Y = np.array([0,1,0,0,1,0]).reshape(1,-1)
    >>> temp = np.asarray(np.where(P + Y == 1))
    >>> indici = temp[1]
    >>> indici
    array([0, 4, 5], dtype=int64)
    
    np.where return indeces where the condition is verified, np.asarray transform tuples of array into an array.
    first row of temp is all zeros because it refers positions in the first
    dimension of A and A is (1,m) so first row of temp refers always to the 1
    """
    #Binary Case
    if len(classes) == 2:
        temp = np.asarray(np.where(P + Y == 1))
        
        indici = temp[1] #Which examples among the m examples has been mislabeled
        n_mislabeled = len(indici)
        assert(n_mislabeled >= num_images)  #In fact I need x errors to print x images
        
        plt.figure(3, figsize = (60.0/num_images, 60.0/num_images)) # Fig size in inches
    
        for i in range(num_images):
            idx = indici[i]
            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:,idx].reshape(64,64,3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(P[0,idx])] + " \n Class: " + classes[Y[0,idx]])
    
    #Multiclass case
    else:
        m = X.shape[1]
        Y_new = np.zeros((1,m))
        for i in range(m):
            Y_new[0,i] = np.argmax(Y[:,i])

        temp = np.asarray(np.where(P != Y_new))
        indici = temp[1]
        n_mislabeled = len(indici)
        assert(n_mislabeled >= num_images)

        plt.figure(3, figsize = (60.0/num_images, 60.0/num_images)) # Fig size in inches

        for i in range(num_images):
            idx = indici[i]
            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:,idx].reshape(28,28),cmap='binary')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(P[0,idx])] + " \n Class: " \
                                                   + classes[int(Y_new[0,idx])])
            


def dataset_show(num_images = 8):
    
    X_train, Y_train, X_test, Y_test, classes = load_cats_dataset()
    print("Binary Dataset: Cat vs Non Cat Classification")
    plt.figure(1, figsize = (150.0/num_images, 150.0/num_images))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(X_train[:,i].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title (classes[Y_train[0, i]] + "   (y = " + str(Y_train[0,i]) + ")")
           #classes it's just cat or non-cat
    plt.show()     
    X_train, Y_train, X_val, Y_val, X_test, Y_test, classes =  load_MNIST_dataset(1000, 100, 100)
    print("Multi-Class Dataset: MNIST")
    plt.figure(2, figsize = (150.0/num_images, 150.0/num_images))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_train[:,10+i].reshape(28,28),cmap='binary')
        plt.axis('off')
        mx = np.argmax(Y_train[:,10+i])
        plt.title("y = " + str(mx))
    plt.show()
    
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
    
if __name__ == "__main__":
    import doctest
    doctest.testmod() 

        
        
        
        
        
    