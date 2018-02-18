from numpy import *
from math import sqrt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

A = norm.ppf(np.random.rand(3, 5))
A = A.T  # IMPORTANT


def LS_grad(x, A, b):
    """
    Gradient for least squares
    Calculates f_i(x) = (1/2) * norm(Ax-b)^2
    Nasty syntax, but pure * notation was not working
    """

    return matmul(A.T, matmul(A, x) - b)


"""
Defining objective functions 
"""


m = 100
p = 256
k = 40

# Matlab uses randperm inclusive; range is half open
d = [97, 172, 136, 165, 147, 231, 34, 117, 157, 85, 234, 220, 186, 132, 215, 212, 183, 120, 251, 225, 179, 75, 103, 210,
     94, 134, 109, 70, 96, 223, 190, 114, 48, 252, 46, 235, 240, 166, 256, 55, 76, 86, 13, 133, 43, 52, 226, 44, 79,
     254, 82, 137, 33, 112, 176, 32, 126, 54, 4, 7, 248, 187, 174, 222, 113, 38, 104, 162, 180, 95, 152, 185, 72, 49,
     155, 228, 199, 71, 202, 47, 98, 209, 219, 92, 58, 189, 200, 236, 41, 167, 193, 110, 203, 119, 78, 2, 145, 19, 106,
     242, 26, 173, 169, 68, 1, 198, 30, 25, 175, 21, 237, 156, 255, 218, 123, 151, 149, 14, 87, 80, 246, 135, 250, 99,
     115, 158, 28, 243, 39, 233, 171, 131, 62, 18, 208, 239, 63, 139, 128, 11, 84, 232, 138, 31, 153, 182, 111, 142, 53,
     143, 74, 238, 35, 140, 178, 93, 249, 207, 27, 148, 66, 81, 227, 50, 37, 204, 116, 83, 45, 168, 69, 177, 201, 164,
     184, 9, 67, 214, 10, 59, 42, 181, 89, 221, 244, 20, 195, 101, 188, 65, 64, 108, 56, 91, 88, 12, 194, 216, 247, 224,
     15, 159, 100, 127, 150, 129, 105, 144, 102, 36, 241, 197, 161, 196, 125, 191, 205, 160, 40, 230, 130, 8, 118, 229,
     206, 24, 163, 170, 61, 107, 245, 3, 60, 217, 121, 90, 16, 122, 5, 29, 124, 17, 57, 77, 73, 141, 211, 213, 22, 154,
     23, 146, 192, 6, 253, 51];

# True signal
xs = np.zeros((p, 1))

# We have to subtract one from the left hand side because indexing starts at 0
# Otherwise we get an error because 256 is selected (at times)

xs = matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             -1.56435446200353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.621812134139797, 0, -1.77141172280630, 0, 0, 0, 0, 0,
             0, 2.16387269695249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0312787742423007, 0, 0, 0, 0,
             3.68803592393863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.17722509125471, 0, 0, 0, 0, 0, 0, 0, 0, -0.259140992764408,
             0, 3.09461254316339, -0.302980902884577, 0, 0, 0, 0, 0, 3.79759196653980, 0, 0, 0, 0, 0, 0.693279820582987,
             0, 0, 0, 0, -1.69546374074908, 0, 0, 2.27614013845244, 0, 0, 0.187659069947048, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, -0.124617636073475, 0, -0.345351430303277, 0, 2.82587047965940, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             3.01997021955705, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.10021271853978, 0, 0, 0, 0, 0, 0, 0, -1.57080124685514,
             -1.43235840563211, 0, 0, 0, 0, 0, -0.513678760535945, 0, 0, 0, 0, 0, 0, -0.241532575167062, 0, 0, 0,
             3.15849841582437, 0, 0, -2.03216039641158, 0, 0, 0, 0.413183428289348, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 2.68710592108248, 0, 2.94407482617290, 0, 0, 1.70068851894709, 0, 0, 0, 0,
             1.56247756939122, 0, 0, -0.285691238808731, 0, 1.34397871261334, 0, 0, 0, 0, 0, 4.07393697389629, 0, 0,
             0.297159281083094, 1.96034460561865, 0, 0, 0, 0, 0.786731822701869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             -0.481159163122467, -4.24692125018727, 0, 0, 0, 0.0725135925455697]);
xs = xs.T

sigma = 0.01

B1 = matrix(0.1 * norm.ppf(np.random.rand(p, m).T))
b1 = B1 * xs + ((sigma / sqrt(m)) * norm.ppf(np.random.rand(1, m).T))
B2 = matrix(0.1 * norm.ppf(np.random.rand(p, m).T))
b2 = B2 * xs + ((sigma / sqrt(m)) * norm.ppf(np.random.rand(1, m).T))
B3 = matrix(0.1 * norm.ppf(np.random.rand(p, m).T))
b3 = B3 * xs + ((sigma / sqrt(m)) * norm.ppf(np.random.rand(1, m).T))
B4 = matrix(0.1 * norm.ppf(np.random.rand(p, m).T))
b4 = B4 * xs + ((sigma / sqrt(m)) * norm.ppf(np.random.rand(1, m).T))
B5 = matrix(0.1 * norm.ppf(np.random.rand(p, m).T))
b5 = B5 * xs + ((sigma / sqrt(m)) * norm.ppf(np.random.rand(1, m).T))

# At this point in the script, b1-b5 look similar to how they do in the matlab
# script




"""
Mixing matrices of the connected network 
"""

A = matrix([[1 / 4, 1 / 4, 0, 1 / 2, 0], [1 / 4, 1 / 4, 0, 0, 1 / 3], [1 / 4, 0, 1 / 2, 0, 1 / 3],
            [0, 1 / 4, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 2, 0, 1 / 3]])
n = shape(A)[0]
A1 = (A + identity(n)) / 2

##############################################################################
B = (transpose(B1) * B1 + transpose(B2) * B2 + transpose(B3) * B3 +
     transpose(B4) * B4 + transpose(B5) * B5)
b = (transpose(B1) * b1 + transpose(B2) * b2 + transpose(B3) * b3 +
     transpose(B4) * b4 + transpose(B5) * b5)

# The B\b syntax used in matlab does not exist
Opt_x = linalg.solve(B, b)

MaxIter = 10000

## %%%%%%%% Initialization of EXTRA-Push %%%%%%%%%

alpha0 = 0.1
alpha1 = 0.02

# Initialization of w-sequence
w0 = ones((n, p))

# Random initialization of sequence z
z0 = norm.ppf(np.random.rand(p, n))
z0 = z0.T
x0 = z0
w1 = matmul(A, w0)

# Gradient function fixed to return a 256 x 5 matrix
grad01 = LS_grad(transpose(x0[0, newaxis]), B1, b1)
grad02 = LS_grad(transpose(x0[1, newaxis]), B2, b2)
grad03 = LS_grad(transpose(x0[2, newaxis]), B3, b3)
grad04 = LS_grad(transpose(x0[3, newaxis]), B4, b4)
grad05 = LS_grad(transpose(x0[4, newaxis]), B5, b5)

myfunMD_grad0 = np.vstack([grad01.T, grad02.T, grad03.T, grad04.T, grad05.T])

z00 = z0
z01 = z0

z10 = (A * z0) - (alpha0 * myfunMD_grad0)

# Divide each element of z10 by w1 (both are 5x256 matrices)
x10 = np.divide(z10, w1)
z11 = A * z0 - alpha1 * myfunMD_grad0
x11 = np.divide(z11, w1)
myfunMD_grad00 = myfunMD_grad0
myfunMD_grad01 = myfunMD_grad0



MSE_Sum0 = np.zeros((MaxIter, 1))
Dist_Grad0 = np.zeros((MaxIter, 1))

MSE_Sum1 = np.zeros((MaxIter, 1))
Dist_Grad1 = np.zeros((MaxIter, 1))



##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



## %%%%%%%% Initialization of NIDS%%%
alpha0 = 0.1
alpha1 = 0.02


# Initialization of w-sequence
nsw0 = ones((n, p))

# Random initialization of sequence z
nsz0 = norm.ppf(np.random.rand(p, n))
nsz0 = nsz0.T
nsx0 = nsz0
nsw1 = matmul(A, nsw0)

# Gradient function fixed to return a 256 x 5 matrix
nsgrad01 = LS_grad(transpose(nsx0[0, newaxis]), B1, b1)
nsgrad02 = LS_grad(transpose(nsx0[1, newaxis]), B2, b2)
nsgrad03 = LS_grad(transpose(nsx0[2, newaxis]), B3, b3)
nsgrad04 = LS_grad(transpose(nsx0[3, newaxis]), B4, b4)
nsgrad05 = LS_grad(transpose(nsx0[4, newaxis]), B5, b5)

nsmyfunMD_grad0 = np.vstack([nsgrad01.T, nsgrad02.T, nsgrad03.T, nsgrad04.T, nsgrad05.T])

nsz00 = nsz0
nsz01 = nsz0

nsz10 = (A * nsz0) - (alpha0 * nsmyfunMD_grad0)

# Divide each element of z10 by w1 (both are 5x256 matrices)
nsx10 = np.divide(nsz10, nsw1)
nsz11 = A * nsz0 - alpha1 * nsmyfunMD_grad0
nsx11 = np.divide(nsz11, nsw1)
nsmyfunMD_grad00 = nsmyfunMD_grad0
nsmyfunMD_grad01 = nsmyfunMD_grad0


nsMSE_Sum0 = np.zeros((MaxIter, 1))
nsDist_Grad0 = np.zeros((MaxIter, 1))

nsMSE_Sum1 = np.zeros((MaxIter, 1))
nsDist_Grad1 = np.zeros((MaxIter, 1))


"""
Runs ExtraPush algorithm for MaxIter iterations
"""


def Run_ExtraPush(alpha, k, z00, z10, x00, x10, myfunMD_grad00):
    #  k = 0
    #  while k < MaxIter:
    ## Step size = 1, alpha = 0.001

    if k == 0:
        z10 = (A * z0) - (alpha * myfunMD_grad0)

        # Divide each element of z10 by w1 (both are 5x256 matrices)
        x10 = np.divide(z10, w1)
        z11 = A * z0 - alpha * myfunMD_grad0
        x11 = np.divide(z11, w1)
        myfunMD_grad00 = myfunMD_grad0
        myfunMD_grad01 = myfunMD_grad0




    #handler gui for inputing user function in place of gradient
    grad110 = LS_grad(transpose(x10[0, newaxis]), B1, b1)
    grad120 = LS_grad(transpose(x10[1, newaxis]), B2, b2)
    grad130 = LS_grad(transpose(x10[2, newaxis]), B3, b3)
    grad140 = LS_grad(transpose(x10[3, newaxis]), B4, b4)
    grad150 = LS_grad(transpose(x10[4, newaxis]), B5, b5)

    myfunMD_grad10 = np.vstack([grad110.T, grad120.T, grad130.T, grad140.T, grad150.T])

    Dist_Grad0[k] = np.linalg.norm(np.ones((n, 1)).T * myfunMD_grad10)

    zk0 = 2 * A1 * z10 - A1 * z00 - alpha * (myfunMD_grad10 - myfunMD_grad00)
    wk = A * w1
    xk0 = np.divide(zk0, wk)
    MSE_Sum0[k] = np.linalg.norm(xk0 - np.tile(Opt_x.T, (n, 1)))  ## tile is numpy equivalent of repmat

    z00 = z10
    z10 = zk0
    x00 = x10
    x10 = xk0
    myfunMD_grad00 = myfunMD_grad10
    # k += 1
    return [MSE_Sum0[k], k, z00, z10, x00, x10, myfunMD_grad00]


"""
Runs NIDS algorithm for MaxIter iterations+
"""
def Run_NIDS(alpha, k, nsz00, nsz10, nsx00, nsx10, nsmyfunMD_grad00):
    #  k = 0
    #  while k < MaxIter:
    ## Step size = 1, alpha = 0.001

    if k == 0:
        nsz10 = (A * nsz0) - (alpha * nsmyfunMD_grad0)

        # Divide each element of z10 by w1 (both are 5x256 matrices)
        nsx10 = np.divide(nsz10, nsw1)
        nsz11 = A * nsz0 - alpha * nsmyfunMD_grad0
        nsx11 = np.divide(nsz11, nsw1)
        nsmyfunMD_grad00 = nsmyfunMD_grad0
        nsmyfunMD_grad01 = nsmyfunMD_grad0

    #handler gui for inputing user function in place of gradient
    nsgrad110 = LS_grad(transpose(nsx10[0, newaxis]), B1, b1)
    nsgrad120 = LS_grad(transpose(nsx10[1, newaxis]), B2, b2)
    nsgrad130 = LS_grad(transpose(nsx10[2, newaxis]), B3, b3)
    nsgrad140 = LS_grad(transpose(nsx10[3, newaxis]), B4, b4)
    nsgrad150 = LS_grad(transpose(nsx10[4, newaxis]), B5, b5)

    nsmyfunMD_grad10 = np.vstack([nsgrad110.T, nsgrad120.T, nsgrad130.T, nsgrad140.T, nsgrad150.T])

    nsDist_Grad0[k] = np.linalg.norm(np.ones((n, 1)).T * nsmyfunMD_grad10)
    nszk0 = 2 * A1 * nsz10 - A1 * (nsz00 + (alpha * (nsmyfunMD_grad10)) - (alpha * (nsmyfunMD_grad00)))
    nswk = A * nsw1
    nsxk0 = np.divide(nszk0, nswk)
    nsMSE_Sum0[k] = np.linalg.norm(nsxk0 - np.tile(Opt_x.T, (n, 1)))  ## tile is numpy equivalent of repmat

    nsz00 = nsz10
    nsz10 = nszk0
    nsx00 = nsx10
    nsx10 = nsxk0
    nsmyfunMD_grad00 = nsmyfunMD_grad10
    # k += 1
    return [nsMSE_Sum0[k], k, nsz00, nsz10, nsx00, nsx10, nsmyfunMD_grad00]


def main():

    k = 0
    result = []
    x00 = 0
    alpha = 0.1
    update = Run_ExtraPush(alpha, k, z00, z10, x00, x10, myfunMD_grad00)
    r = update[0]
    result.insert(k, r)
    k += 1
    while k < MaxIter:
        update = Run_ExtraPush(alpha, k, update[2], update[3], update[4], update[5], update[6])
        r = update[0]
        result.insert(k, r)
        k = k + 1

    k = 0
    result2 = []
    nsx00 = 0

    update = Run_NIDS(alpha, k, nsz00, nsz10, nsx00, nsx10, nsmyfunMD_grad00)
    r = update[0]
    result2.insert(k, r)
    k += 1
    while k < MaxIter:
        update = Run_NIDS(alpha, k, update[2], update[3], update[4], update[5], update[6])
        r = update[0]
        result2.insert(k, r)
        k = k + 1

    # GUI for step size
    iter = np.arange(MaxIter)
    plt.style.use('dark_background')
    plt.yscale('log')
    plt.plot(iter, Dist_Grad0, label='ExtraPush')
    plt.plot(iter, nsDist_Grad0, label='NIDS')
    plt.title("ExtraPush and NIDSPush Iterative Error versus Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Iterative Error")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()





