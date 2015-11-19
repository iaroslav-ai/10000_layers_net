'''
Created on Oct 20, 2015

@author: Iaroslav
'''

import numpy as np

def GenerateData(xsz, N, w):
    x = np.random.rand(N, xsz)*3.1415*2;
    y = np.cos( np.dot( np.sin(x),w )*10 );
    return x,y