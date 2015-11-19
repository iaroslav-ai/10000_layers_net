'''
Created on Oct 4, 2015

@author: Iaroslav
'''

import numpy as np
from fnn_py import FNN
from sgd_py import TrainNetwork
from example_data import GenerateData
from numgrad import NumericalGradient
from mnist_py import MNIST
import matplotlib.pyplot as plt

N = 1000;

# MNIST loader:
data = MNIST(verbose=True);

# this are the contents of the loaded data:
X = data.train_images[:N,]
Y = data.train_labels[:N,]

xsz = X.shape[1]

# do compression to single number encoding 
compressed = False;
while not compressed:
    # do compression of training points
    w = np.random.randn(xsz);
    H = np.dot(X,w);
    # check if all projections are unique
    dct = {};
    compressed = True;
    for h in H:
        if h in dct:
            compressed = False;
            break;
        dct[h] = True;
    # there are colliding projections
    if not compressed:
        continue
    # there are no colliding projections
    # make proj. positive and normalized
    H = H - np.min(H);
    H = H / np.max(H);
    X = np.random.rand(H.shape[0],1);
    X[:,0] = H*100
    
xsz = 1;
ysz = 1;

# network objective
def trainobj(ffnnd):
    obj = 0
    Yp = X[:,0];
    for i in range(len(X)):
        Yp[i] = ffnnd.forward(X[i,])
    
    obj = 0.5 * (np.linalg.norm(Yp - np.transpose(Y), 2) ** 2);
    return obj

means = X[:-1,0];
W = np.ones((3, X.shape[0]));
W[1,1:] = -means;
W[1,0] = -(min(X) - 1);

Inputs = np.column_stack((X, X[:,[0]]*0+1));
H = np.maximum(np.dot(Inputs, W[:-1,]), 0)
W[-1,] = np.linalg.lstsq(H, Y)[0][:,0]
val =(np.dot(H,W[-1,:]))
obj = 0.5 * (np.linalg.norm(np.dot(H,W[-1,:]) - np.transpose(Y), 2) ** 2);

plt.plot(X, Y,'*')
plt.plot(X, val,'o')

plt.xlim([-0.0,1.1])
plt.ylim([-1,10])

plt.xlabel('projection')
plt.ylabel('label')
plt.grid(True)
plt.show()

neurons = W.shape[1]
layers = 1

fnn = FNN(xsz, ysz, neurons,layers)

fnn.W = W;

print obj

lsz = xsz + 1 + ysz*2

ffnnd = FNN(xsz, ysz, lsz ,neurons)

ffnnd.training = True;
ffnnd.forward(X[0,]) # this records layer boundaries

template = ffnnd.W[:lsz, ];

ffnnd.W = ffnnd.W * 0

for i in range(neurons):
    bounds = ffnnd.bounds[i];
    l = bounds[0]
    r = bounds[1]
    if i == 0:
        ffnnd.W[l:r-1,:xsz] = np.identity(xsz)
        ffnnd.W[l:r,xsz] = fnn.W[:-1, i]
    else:
        ffnnd.W[l:(l+xsz),:xsz] = np.identity(xsz) # input transport
        
        ffnnd.W[l:(l+xsz),xsz] = fnn.W[:-2, i] # processing
        ffnnd.W[r-1,xsz] = fnn.W[-2, i]
        
        ffnnd.W[(l+xsz), -2] = fnn.W[-1, i-1]
        ffnnd.W[(l+xsz+1), -2] = 1 # positive part of output
        ffnnd.W[(l+xsz+2), -2] = -1 # positive part of output
        
        ffnnd.W[(l+xsz), -1] = -fnn.W[-1, i-1]
        ffnnd.W[(l+xsz+1), -1] = -1 # negative part of output
        ffnnd.W[(l+xsz+2), -1] = 1 # negative part of output

# outputs of the final linear layer
ffnnd.W[-1, -1] = -1
ffnnd.W[-1, -2] = 1
ffnnd.W[-1, -3] = fnn.W[-1,-1]


print "network objective:", trainobj(ffnnd)