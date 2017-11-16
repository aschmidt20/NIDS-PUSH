from numpy import *
from math import sqrt
import numpy as np
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




"""
Initialization of EXTRA-Push
"""
# Step size parameter for EXTRA-Push
alpha0 = 0.1
alpha1 = 0.02

# Initialization of w-sequence 
w0 = ones((n,p))

# Random initialization of sequence z 
z0 = random.randn(n, p);
x0 = z0
w1 = matmul(A, w0)

# Gradient function fixed to return a 256 x 1 matrix 
# The problem is that x0[0] has dimension (256, ), which causes problems
# We use newaxis command to make dimension (256, 1) which fixes dim problem
grad01 = LS_grad(transpose(x0[0, newaxis]), B1, b1)
grad02 = LS_grad(transpose(x0[1, newaxis]), B2, b2)
grad03 = LS_grad(transpose(x0[2, newaxis]), B3, b3)
grad04 = LS_grad(transpose(x0[3, newaxis]), B4, b4)
grad05 = LS_grad(transpose(x0[4, newaxis]), B5, b5)

## STILL PROBLEMATIC- the vstack function returns a (5, 256) matrix- are we 
# sure this is the right command?
myfunMD_grad0 = np.vstack([grad01.T,grad02.T,grad03.T,grad04.T,grad05.T])

z00 = z0
z01 = z0

# Comment back when dimensions are right
"""
z10 = (A*z0) - (alpha0*myfunMD_grad0)
# Divide each element of z10 by w1 (both are 5x256 matrices)
x10 = np.divide(z10,w1)
z11 = A*z0 - alpha1*myfunMD_grad0
x11 = mp.divide(z11,w1)
myfunMD_grad00 = myfunMD_grad0
myfunMD_grad01 = myfunMD_grad0
"""




