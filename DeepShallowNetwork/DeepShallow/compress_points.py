'''
Function below encodes every feature vector in matrix X with unique single number
'''

import numpy as np

def compress(X, Xt):
    # do compression to single number encoding 
    compressed = False;
    while not compressed:
        # do compression of training points
        w = np.random.randn(X.shape[1]);
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
        
        Ht = np.dot(Xt,w)
        
        m = np.min(np.concatenate((H,Ht)))
        M = np.max(np.concatenate((H,Ht))-m)
        
        # there are no colliding projections
        # make proj. positive and normalized
        H = H - m;
        H = H / M;
        result = np.random.rand(H.shape[0],1);
        result[:,0] = H
        
        Ht = Ht - m;
        Ht = Ht / M;
        result_t = np.random.rand(Ht.shape[0],1);
        result_t[:,0] = Ht
    return result, result_t