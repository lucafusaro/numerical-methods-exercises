"""Use the Gauss-Seidel method and the Gauss elimination method
wth backsubstitution ( or LU with backsubs) to solve circuit.
Check with numpy.linalg.solve"""

import numpy as np
from numpy.linalg import solve

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
        y = np.matmul(A, x) - np.multiply(x, d)
        # matmul fa moltiplicazione fra matrici, multiply moltiplicazione fra array (elemento per elemento)
        # Gauss-Seidel formula
        x1 = np.multiply(dm1, b - y)

        # Calculating accuracy and updating x
        acc = np.max(abs(x1 - x))
        x = x1

    return x


def linear_solve_gauss(A, b):
    N = len(b)

    for i in range(N):
        # Normalize diagonal to 1.
        diag_element = A[i, i]
        A[i, :] /= diag_element
        b[i] /= diag_element

        for j in range(i + 1, N):
            # Subtract upper rows, multiplied by suitable
            # factor, to generate an upper diagonal matrix
            # At the end of each loop, column i (set by loop above)
            # has become (1, 0, .... , 0)
            under_diag = A[j, i]
            A[j, :] -= under_diag * A[i, :]    #nota che l'elemento sopra l'elemento under diag è diventato 1 per i passaggi di prima, stiamo quindi portando a zero l'elemento under diag (ma dobbiamo modificare tutta la riga)
            b[j] -= under_diag * b[i]
            #print(A)
        #arrivati a quato punto abbiamo tutti 0 sotto la diagonale
    #backsubstitution
    # Initialize solution vector
    x = np.zeros(len(b))
    # Gets last solution, x_(N-1)
    x[N - 1] = b[N - 1]

    # Proceed backward, line by line
    for i in range(N - 2, -1, -1):
        v = np.dot(A[i, i + 1:N], x[i + 1:N])  #nota che per i = -1 prende l'ultima riga
        # if (i==1): print(i, i+1, N-1, A[i,i+1:N])
        x[i] = (b[i] - v)

    return x



matrix = np.array([[4, -1, -1, -1], [-1, 3, 0, -1], [-1, 0, 3, -1], [-1, -1, -1, 4]], dtype=float)
b = np.array([5., 0., 5., 0.])
guess = [1, 1, 1, 1]
tol = 1e-5
sol_GS = GSeid(matrix, b, guess, tol)
sol_back = linear_solve_gauss(matrix, b)
sol_linalg = solve(matrix, b)
print("Solution vector [V1 V3 V2 V4]\nWith Gauss-Seidel x =", sol_GS)
print("With Gauss with backsubstitution x =", sol_back)
print("Check with numpy.linalg.solve x =", sol_linalg)
