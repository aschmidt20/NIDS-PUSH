from numpy import *
from math import sqrt
import numpy as np

np.random.seed(9001)
# -*- coding: utf-8 -*-

"""
Python translation from Matlab for extrapush versus Subgradient push for least squares regression
- minimize F(x) = f1(x)+f2(x)+f3(x)+f4(x)+f5(x)
- f_i=0.5*\|B_ix-b_i\|^2
- Network: 5 nodes, strongly connected
"""

"""
Authors: Tyler Will, Andrew Schmidt
"""


def LS_grad(x,A,b):
    """
    Gradient for least squares
    Calculates f_i(x) = (1/2) * norm(Ax-b)^2
    Nasty syntax, but pure * notation was not working
    """
    
    return matmul(A.T, matmul(A, x)-b)


"""
Defining objective functions
"""

m = 100
p = 256
k = 40

# Matlab uses randperm inclusive; range is half open
d = random.permutation(range(1, p + 1))

# True signal
xs = np.zeros((p, 1))

# We have to subtract one from the left hand side because indexing starts at 0
# Otherwise we get an error because 256 is selected (at times)
xs[d[0:k] - 1] = 2 * random.randn(k, 1)

sigma = 0.01

B1 = matrix(0.1 * random.randn(m, p))
B2 = matrix(0.1 * random.randn(m, p))
B3 = matrix(0.1 * random.randn(m, p))
B4 = matrix(0.1 * random.randn(m, p))
B5 = matrix(0.1 * random.randn(m, p))

b1 = B1 * xs + ((sigma/sqrt(m)) * random.randn(m,1))
b2 = B2 * xs + ((sigma/sqrt(m)) * random.randn(m,1))
b3 = B3 * xs + ((sigma/sqrt(m)) * random.randn(m,1))
b4 = B4 * xs + ((sigma/sqrt(m)) * random.randn(m,1))
b5 = B5 * xs + ((sigma/sqrt(m)) * random.randn(m,1))

# At this point in the script, b1-b5 look similar to how they do in the matlab 
# script 




"""
Mixing matrices of the connected network 
"""

A = matrix([[1/4, 1/4, 0, 1/2, 0], [1/4, 1/4, 0, 0, 1/3], [1/4, 0, 1/2, 0, 1/3], 
            [0, 1/4, 0, 1/2, 0], [1/4, 1/4, 1/2, 0, 1/3]])
n = shape(A)[0]
A1 = (A + identity(n))/2

##############################################################################
B = (transpose(B1) * B1 + transpose(B2) * B2 + transpose(B3) * B3 + 
     transpose(B4) * B4 + transpose(B5) * B5)
b = (transpose(B1) * b1 + transpose(B2) * b2 + transpose(B3) * b3 + 
     transpose(B4) * b4 + transpose(B5) * b5)

# The B\b syntax used in matlab does not exist  
Opt_x = linalg.solve(B, b)

MaxIter = 10000





## %%%%%%%% Initialization of EXTRA-Push %%%%%%%%%

# Step size parameter for EXTRA-Push
alpha0 = 0.1
alpha1 = 0.02

# Initialization of w-sequence 
w0 = ones((n,p))

# Random initialization of sequence z 
z0 = random.randn(n, p)
x0 = z0
w1 = matmul(A, w0)

# Gradient function fixed to return a 256 x 5 matrix
grad01 = LS_grad(transpose(x0[0, newaxis]), B1, b1)
grad02 = LS_grad(transpose(x0[1, newaxis]), B2, b2)
grad03 = LS_grad(transpose(x0[2, newaxis]), B3, b3)
grad04 = LS_grad(transpose(x0[3, newaxis]), B4, b4)
grad05 = LS_grad(transpose(x0[4, newaxis]), B5, b5)


myfunMD_grad0 = np.vstack([grad01.T,grad02.T,grad03.T,grad04.T,grad05.T])

z00 = z0
z01 = z0

z10 = (A*z0) - (alpha0*myfunMD_grad0)

# Divide each element of z10 by w1 (both are 5x256 matrices)
x10 = np.divide(z10,w1)
z11 = A*z0 - alpha1*myfunMD_grad0
x11 = np.divide(z11,w1)
myfunMD_grad00 = myfunMD_grad0
myfunMD_grad01 = myfunMD_grad0

MSE_Sum0 = np.zeros((MaxIter,1))
Dist_Grad0 = np.zeros((MaxIter,1))

MSE_Sum1 = np.zeros((MaxIter,1))
Dist_Grad1 = np.zeros((MaxIter,1))

## %%%%%%%%%%% Initialization of Normalized ExtraPush %%%%%%%%%%%%%%%%%%%%%
AA = np.power(A,100)
phi = AA[:,0]
nalpha0 = 0.1
nalpha1 = 0.02

nz0 = random.randn(n,p)
nx0 = np.diag(n*phi)
nx0 = np.power(nx0,(-1)*nz0)

# Gradient function fixed to return a 256 x 5 matrix
ngrad01 = LS_grad(transpose(nx0[0, newaxis]), B1, b1)
ngrad02 = LS_grad(transpose(nx0[1, newaxis]), B2, b2)
ngrad03 = LS_grad(transpose(nx0[2, newaxis]), B3, b3)
ngrad04 = LS_grad(transpose(nx0[3, newaxis]), B4, b4)
ngrad05 = LS_grad(transpose(nx0[4, newaxis]), B5, b5)


nmyfunMD_grad0 = np.vstack([grad01.T,grad02.T,grad03.T,grad04.T,grad05.T])

nz00 = nz0;
nz01 = nz0;

nz10 = A*nz00 - nalpha0*nmyfunMD_grad0
nx10 = np.power(diag(n*phi),(-1)*nz10)

nz11 = A*nz01 - nalpha1*nmyfunMD_grad0
nx11 = np.power(diag(n*phi),(-1)*nz11)

nmyfunMD_grad00 = nmyfunMD_grad0
nmyfunMD_grad01 = nmyfunMD_grad0

nzk0 = np.zeros((n,p))
nzk1 = np.zeros((n,p))

MSE_nSum0 = np.zeros((MaxIter,1))
Dist_nGrad0 = np.zeros((MaxIter,1))

MSE_nSum1 = np.zeros((MaxIter,1))
Dist_nGrad1 = np.zeros((MaxIter,1))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
Runs ExtraPush algorithm for MaxIter iterations
"""
def Run_ExtraPush():
    k = 0
    while k < MaxIter:
        ## Step size = 1, alpha = 0.001
        grad110 = LS_grad(transpose(x0[0, newaxis]), B1, b1)
        grad120 = LS_grad(transpose(x0[1, newaxis]), B2, b2)
        grad130 = LS_grad(transpose(x0[2, newaxis]), B3, b3)
        grad140 = LS_grad(transpose(x0[3, newaxis]), B4, b4)
        grad150 = LS_grad(transpose(x0[4, newaxis]), B5, b5)

        myfunMD_grad10 = np.vstack([grad110.T, grad120.T, grad130.T, grad140.T, grad150.T])

        Dist_Grad0[k] = np.linalg.norm(np.ones((n,1)).T*myfunMD_grad10)


        zk0 = 2*A1*z10 - A1*z00 -alpha0*(myfunMD_grad10-myfunMD_grad00)
        wk = A*w1
        xk0 = np.divide(zk0,wk)
        MSE_Sum0[k] = np.linalg.norm(xk0-np.tile(Opt_x.T,(n,1)))    ## tile is numpy equivalent of repmat
        UpdateExtraPushVariables(myfunMD_grad10, z10, zk0, x10, xk0)
        k += 1
    return MSE_Sum0

"""
Runs ExtraPush algorithm for MaxIter iterations
"""
def Run_NormalizedExtraPush():
    k = 0
    while k < MaxIter:
        ## Step size = 1, alpha = 0.001
        ngrad111 = LS_grad(transpose(nx11[0, newaxis]), B1, b1)
        ngrad121 = LS_grad(transpose(nx11[1, newaxis]), B2, b2)
        ngrad131 = LS_grad(transpose(nx11[2, newaxis]), B3, b3)
        ngrad141 = LS_grad(transpose(nx11[3, newaxis]), B4, b4)
        ngrad151 = LS_grad(transpose(nx11[4, newaxis]), B5, b5)

        nmyfunMD_grad11 = np.vstack([ngrad111.T, ngrad121.T, ngrad131.T, ngrad141.T, ngrad151.T])

        Dist_nGrad1[k] = np.linalg.norm(np.ones((n,1)).T*nmyfunMD_grad11)


        nzk1 = 2*A1*nz11 - A1*nz01 -nalpha1*(nmyfunMD_grad11-nmyfunMD_grad01)
        nxk1 = (np.power(np.diag(n*phi),(-1)*nzk1))
        MSE_nSum1[k] = np.linalg.norm(nxk1-np.tile(Opt_x.T,(n,1)))    ## tile is numpy equivalent of repmat
        UpdateNormalizedExtraPushVariables(nmyfunMD_grad11, nz11, nzk1, nx11, nxk1)
        k += 1
    return MSE_nSum1

"""
Runs NIDS algorithm for MaxIter iterations
"""
def Run_NIDS():
    k = 0
    while k < MaxIter:
        ## Step size = 1, alpha = 0.001
        grad110 = LS_grad(transpose(x0[0, newaxis]), B1, b1)
        grad120 = LS_grad(transpose(x0[1, newaxis]), B2, b2)
        grad130 = LS_grad(transpose(x0[2, newaxis]), B3, b3)
        grad140 = LS_grad(transpose(x0[3, newaxis]), B4, b4)
        grad150 = LS_grad(transpose(x0[4, newaxis]), B5, b5)

        myfunMD_grad10 = np.vstack([grad110.T, grad120.T, grad130.T, grad140.T, grad150.T])

        Dist_Grad0[k] = np.linalg.norm(np.ones((n,1)).T*myfunMD_grad10)


        zk0 = 2*A1*z10 - A1*(z00 +alpha0*(myfunMD_grad10-myfunMD_grad00))
        wk = A*w1
        xk0 = np.divide(zk0,wk)
        MSE_Sum0[k] = np.linalg.norm(xk0-np.tile(Opt_x.T,(n,1)))    ## tile is numpy equivalent of repmat
        UpdateExtraPushVariables(myfunMD_grad10, z10, zk0, x10, xk0)
        k += 1
    return MSE_Sum0

""" Updates variables on each iteration of ExtraPush """
def UpdateExtraPushVariables(myfunMD_grad10, z10, zk0, x10, xk0):

    z00 = z10
    z10 = zk0
    x00 = x10
    x10 = xk0
    myfunMD_grad00 = myfunMD_grad10

""" Updates variables on each iteration of ExtraPush """

def UpdateNormalizedExtraPushVariables(nmyfunMD_grad11, nz11, nzk1, nx11, nxk1):
    nz01 = nz11
    nz11 = nzk1
    nx01 = nx11
    nx11 = nxk1
    nmyfunMD_grad01 = nmyfunMD_grad11


def main():
    result = Run_ExtraPush()
    print(result)
    result3 = Run_NIDS()
    print(result3)
    return 0
if __name__ == "__main__":
    main()





