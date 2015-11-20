'''
Created on Oct 4, 2015

@author: Iaroslav
'''

import numpy as np
from fnn_py import FNN
from mnist_py import MNIST
from compress_points import compress
from fit_shallow_nn import fit_shallow_nn
from shallow_to_deep import shallow_to_deep

N = 10000;

# ============ loading MNIST  ============ 

data = MNIST(verbose=True);

# this are the contents of the loaded data:
X = compress(data.train_images[:N,]) # here images are compressed to single number
Y = data.train_labels[:N,]
    
xsz = 1;
ysz = 1;

# network objective
def training_loss(ffnnd):
    Yp = X[:,0];
    for i in range(len(X)):
        Yp[i] = ffnnd.forward(X[i,])
    return 0.5 * (np.linalg.norm(Yp - np.transpose(Y), 2) ** 2) / Yp.shape[0];

# ============ fitting shallow net ============ 

W, obj = fit_shallow_nn(X, Y)

print "shallow init MSE: ", obj # non - zero loss value do to num. errors

# ============ converting shallow net to deep one ============ 

ffnnd = shallow_to_deep(W)

print "deep network MSE:", training_loss(ffnnd)