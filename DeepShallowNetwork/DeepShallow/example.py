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

N = 100;

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
    X[:,0] = H
    
xsz = 1;
ysz = 1;

means = X[1:,0] + X[:-1,0];
W = np.ones(3, X.shape[0]);
W[0,1:] = -means;
W[0,0] = min(X) - 1;

Inputs = np.column_stack((X, X[:,[0]]*0+1));
H = np.maximum(np.dot(Inputs, W[:-1,]), 0)
W[-1,] = np.linalg.lstsq(H, Y)[0][:,0]
obj = 0.5 * (np.linalg.norm(np.dot(H,W[-1,:]) - np.transpose(Y), 2) ** 2);


def supPretr(X, Y, fnn, iters):
    W = np.copy(fnn.W)
    Inputs = np.column_stack((X, X[:,[0]]*0+1));
    def Objective(W):
        W = W
        H = np.maximum(np.dot(Inputs, W[:-1,]), 0)
        W[-1,] = np.linalg.lstsq(H, Y)[0][:,0]
        obj = 0.5 * (np.linalg.norm(np.dot(H,W[-1,:]) - np.transpose(Y), 2) ** 2);
        return obj
    objv = Objective(W)
    print -1, objv
    for i in range(iters):
        print i
        Wt = np.copy(W)
        for k in range( int(np.sqrt(neurons)) ):
            Wt[ np.random.randint(Wt.shape[0]-2), ] += np.random.randn(Wt.shape[1])*0.0001;
        objl = Objective(Wt)
        if objl < objv:
            objv = objl
            W = np.copy(Wt)
            print objv
    fnn.W = W
    return objv

neurons = N*2
layers = 1

fnn = FNN(xsz, ysz, neurons,layers)

objv =  supPretr(X, Y, fnn, 1000)

print "final:",objv

"""
Inputs = np.column_stack((X, X[:,[1]]*0+1))
H = np.maximum(np.dot(Inputs, fnn.W[:-1,]), 0)
obt = 0.5 * (np.linalg.norm(np.dot(H,fnn.W[-1,:]) - np.transpose(Y), 2) ** 2);

print obt"""

#TrainNetwork(X,Y, fnn, alpha=0.000000001);

print "shallow init finished"

"""
construct super deep nn
neurons on layer: 
    input transport:     xsz (assumption: x values are positive)
    processing:          1
    output transport:    ysz * 2

"""

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

TrainNetwork(X,Y, ffnnd, alpha=0.00000000001);

