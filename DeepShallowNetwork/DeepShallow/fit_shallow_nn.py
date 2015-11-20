'''
This fits the output values to the input values with shallow neural network
'''

import numpy as np

# below is the modified version of supervised pretraining
# main difference is that neurons are not generated randomly
# to see why below code works, try visualizing what it does
def fit_shallow_nn(X, Y):
    
    means = X[:-1,0];
    W = np.ones((3, X.shape[0]));
    W[1,1:] = -means;
    W[1,0] = -(min(X) - 1);
    
    Inputs = np.column_stack((X, X[:,[0]]*0+1));
    H = np.maximum(np.dot(Inputs, W[:-1,]), 0)
    W[-1,] = np.linalg.lstsq(H, Y)[0][:,0]
    obj = 0.5 * (np.linalg.norm(np.dot(H,W[-1,:]) - np.transpose(Y), 2) ** 2) / Y.shape[0];
    
    return W, obj