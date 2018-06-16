# Neural-Networks-from-scratch

This project aims to help machine learning practitioners to design and code a neural network from scratch in Python (using only Numpy).


In the second part of the notebook the implementation is compared with the TensorFlow one. Without GPU support the Numpy version is faster (about a factor of 2) than the TensorFlow one. The accuracy is comparable: 95% on test set in about one minute of training (without any hyperparameters tuning)


The model presented is simple but it is quite generic and easy to understand and to customize. Any feedforward architecture can be chosen, both for binary and multi-class classification.


Two datasets are included: cat vs non-cat pictures for binary classification and the MNIST Dataset for multi-class classification.

All the characteristics of the neural network (cost functions, activation functions, regularization, optimizer and datasets characteristics) are illustrated at the beginning of the notebook.

Enjoy :)
