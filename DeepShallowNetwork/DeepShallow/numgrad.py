'''
Created on Oct 27, 2015

@author: Iaroslav
'''

def NumericalGradient(fnc, x):
    grad = x * 0
    f0 = fnc(x)
    step = 1e-6
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += step
            grad[i,j] = (fnc(x) - f0) / step
            x[i,j] -= step
    return grad