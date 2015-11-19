'''
Created on Oct 16, 2015

@author: Iaroslav
'''

import numpy as np

# train network with sgd
def TrainNetwork(X,Y, network, alpha = 0.001, valchecks = 10, training_fraction = 1):
    Wbest = np.copy( network.W )
    ValBest, checks = np.Inf, valchecks
    while checks > 0:
        # validation of the network
        obj = 0
        
        #for i in range(int(training_fraction * len(X)), len(X)):
        for i in range(int(training_fraction * len(X))):
            obj += 0.5*(np.linalg.norm( network.forward(X[i,]) - Y[i,], 2) ** 2); # L2 objective
            
        # update stopping criterion
        if obj < ValBest:
            ValBest , checks, alpha, Wbest = obj, valchecks, alpha*1.1, np.copy(network.W)
        else:
            checks, alpha = checks-1, alpha*0.7
        
        # output the training progress
        print "Objective:", obj, "best:", ValBest
        
        # Training of the network for one epoch
        network.training = True;
        for i in range(int(training_fraction * len(X))):
            backprop = network.forward(X[i,]) - Y[i,]; 
            grad = network.backward(backprop);
            network.W -= grad*alpha ;
        network.training = False;
        
    network.W = Wbest