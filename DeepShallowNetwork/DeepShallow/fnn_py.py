'''
Created on Oct 16, 2015

@author: Iaroslav
'''

import numpy as np
# Simple RNN
class FNN:
    training = False; # collect data for gradient computation when true
    def __init__(self, xsz, ysz, neurons, layers):
        self.W = np.random.randn(xsz+1+(neurons+1)*(layers-1)+ysz, neurons)*0.001
        self.xsz = xsz
        self.ysz = ysz
        self.layers = layers
    # compute outputs for given sequence x
    def forward(self,x):
        netinput = np.concatenate((x, [1]));
        self.activations, self.inputs, self.bounds = [], [], []
        layer_start = 0;
        for i in range(self.layers):
            H = np.maximum( np.dot(netinput,self.W[layer_start:(layer_start+netinput.shape[0]),]) , 0)
            if self.training:
                self.activations.append(np.copy(H));
                self.inputs.append(np.copy(netinput));
                self.bounds.append([layer_start, layer_start+netinput.shape[0]])
            layer_start += netinput.shape[0]
            netinput = np.concatenate((H, [1]));
        y = np.dot(H, np.transpose( self.W[-self.ysz:,] ))
        return y
    # compute gradient of the network; assumes that "forward" was called with
    # training flag set to true
    def backward(self,backprop):
        grad = np.copy( self.W )*0;
        H = self.activations[-1]
        grad[-self.ysz:,] += np.outer( backprop, H )
        bck = np.dot( backprop, self.W[-self.ysz:,] )
        for i in range(len(self.activations)-1,-1,-1):
            H = self.activations[i]
            bck = (bck) * (H > 0);
            grad[self.bounds[i][0]:self.bounds[i][1],] += np.outer(self.inputs[i], bck)
            bck = np.dot( self.W[self.bounds[i][0]:self.bounds[i][1],], bck )
            bck = bck[:-1]
        return grad