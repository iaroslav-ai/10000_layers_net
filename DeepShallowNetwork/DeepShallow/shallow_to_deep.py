'''
This initializes deep network with shallow network
Function is somewhat complicated due to the format used to store parameters
of the network (single 2d matrix).
This way it is easy to experiment with different descent techniques (from 
where the code actually originates).
'''

import numpy as np
from fnn_py import FNN

# this converts shallow network to equivalent deep one
# this function assumes that input and output are scalars for the network
def shallow_to_deep(W):
    
    xsz = 1
    ysz = 1
    neurons = W.shape[1]
    
    layer_size = xsz + 1 + ysz*2

    ffnnd = FNN(xsz, ysz, layer_size ,neurons)
    
    ffnnd.training = True;
    ffnnd.forward([1]) # this records layer boundaries
    
    
    ffnnd.W = ffnnd.W * 0
    
    for i in range(neurons):
        bounds = ffnnd.bounds[i];
        l = bounds[0]
        r = bounds[1]
        if i == 0:
            ffnnd.W[l:r-1,:xsz] = np.identity(xsz)
            ffnnd.W[l:r,xsz] = W[:-1, i]
        else:
            ffnnd.W[l:(l+xsz),:xsz] = np.identity(xsz) # input transport
            
            ffnnd.W[l:(l+xsz),xsz] = W[:-2, i] # processing
            ffnnd.W[r-1,xsz] = W[-2, i]
            
            ffnnd.W[(l+xsz), -2] = W[-1, i-1]
            ffnnd.W[(l+xsz+1), -2] = 1 # positive part of output
            ffnnd.W[(l+xsz+2), -2] = -1 # positive part of output
            
            ffnnd.W[(l+xsz), -1] = -W[-1, i-1]
            ffnnd.W[(l+xsz+1), -1] = -1 # negative part of output
            ffnnd.W[(l+xsz+2), -1] = 1 # negative part of output
    
    # outputs of the final linear layer
    ffnnd.W[-1, -1] = -1
    ffnnd.W[-1, -2] = 1
    ffnnd.W[-1, -3] = W[-1,-1]
    
    return ffnnd