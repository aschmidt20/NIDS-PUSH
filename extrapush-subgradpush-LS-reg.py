import numpy as np
# -*- coding: utf-8 -*-
"""
Python translation from Matlab for extrapush versus Subgradient push for least squares regression
- minimize F(x) = f1(x)+f2(x)+f3(x)+f4(x)+f5(x)
- f_i=0.5*\|B_ix-b_i\|^2
- Network: 5 nodes, strongly connected
"""

# Gradient for Least Squares
# f_i(x) = 1/2*norm(Ax-b)^2
def LS_grad(x,A,b):
 return np.transpose(A).matmul(A.matmul(x)-b)




