"""Produce a python script to implement the Gauss elimination method
with backsubstitution and solve for system 14 (see slide);
(When the matrix has zero elements on the diagonal backsubstitution doesn't work --> divide by zero
We can swip two rows in order to have on diagonal elements a non-zero value)"""

import numpy as np
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
        #arrivati a questo punto abbiamo tutti 0 sotto la diagonale
    #backsubstitution
    # Initialize solution vector
    x = np.zeros(len(b))
    # Gets last solution, x_(N-1)
    # x[N - 1] = b[N - 1]/A[N-1, N-1]   ma A[N-1, N-1] = 1
    x[N - 1] = b[N - 1]
    # Proceed backward, line by line
    for i in range(N - 2, -1, -1):
        v = np.dot(A[i, i + 1:N], x[i + 1:N])  #nota che per i = -1 prende l'ultima riga
        # if (i==1): print(i, i+1, N-1, A[i,i+1:N])
        # x[i] = (b[i] - v)/A[i, i] ma A[i, i] = 1
        x[i] = (b[i] - v)

    return x

# Create example for testing

A = [ [2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)
#print(b)
x = linear_solve_gauss(A,b)

#print(A)

print('The solution vector x is')
print(x)