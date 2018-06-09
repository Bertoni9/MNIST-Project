# Neural-Networks-from-scratch

This project aims to help machine learning practitioners to design and code a neural network from scratch in Python (using only Numpy).


In the second part of the notebook the implementation is compared with the TensorFlow one. Apparently, without GPU support the Numpy version is way faster.


The model presented is simple but it is quite generic and easy to understand and to customize. Any feedforward architecture can be chosen, both for binary and multi-class classification.


Two datasets are included: cat vs non-cat pictures for binary classification and the MNIST Dataset for multi-class classification.

The model easily reaches 100% in the training set but it overfits the test set due to a lack of regularization and other features. A good accuracy is out of the scope in this simple version, features can be easily added in TensorFlow.

All the characteristics of the neural network (cost functions, activation functions, optimizer and datasets characteristics) are illustrated at the beginning of the notebook.

Enjoy :)
