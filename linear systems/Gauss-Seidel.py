import numpy as np

"""Note: for GS method you must not have zero on diagonal, moreover 
if it does not converge try to rearrange matrix in order to have
 |diag_element| > sum(|(other element of the row)|) that is 'strictly diagonal matrix'"""

def GSeid(A, b, x0, tol):
    # Solving linear systems via Gauss-Seidel method
    # Array size
    N = len(b)
    # Vector containing diagonal elements
    d = np.diag(A)
    # Vector containing inverse of diagonal
    dm1 = (np.diag(A)) ** (-1)

    x = x0

    acc = tol + 1.
    while (acc > tol):

        # \sum Aij xj - Aii xi
        y = np.matmul(A, x) - np.multiply(x, d)   #matmul fa moltiplicazione fra matrici, multiply moltiplicazione fra array (elemento per elemento)
        # Gauss-Seidel formula
        x1 = np.multiply(dm1, b - y)

        # Calculating accuracy and updating x
        acc = np.max(abs(x1 - x))
        x = x1

    return x

"""Example: write a script to implement the Gauss-Seidel method and use it to solve "
 the following equations. 
 [4 -1 -1   x1      12
  -1 4 -2   x2  =   -1
  1 -2 4 ]  x3      5
  """
matrix = np.array([[4, -1, 1], [-1, 4, -2], [1, -2, 4]], dtype=float)
b = np.array([12., -1., 5.])
guess = [1, 1, 1]
tol = 1e-10
solution = GSeid(matrix, b, guess, tol)
print(solution)

