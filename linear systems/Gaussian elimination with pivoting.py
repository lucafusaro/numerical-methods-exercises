"""Adding pivoting (useful when we have zero element on diagonal) to the code of gaussian elimination"""

import numpy as np
from numpy.linalg import solve


def linsolve_gauss_pivot(A, b):
    # Solve linear system Ax = b via
    # Gauss elimination with pivoting

    N = len(b)

    # Vector of diagonal elements of A
    # To be updated during pivoting
    diag = np.empty(N)

    for i in range(N):

        # Partial pivoting
        #######################
        # Collecting elements of column i in vector 'col'
        coloumn = abs(A[:, i])
        # Find max element in col (I can of course also perform this and
        # the previous step in a single line)
        max_index = np.argmax(coloumn)   #argmax ritorna l'indice del max value
        # If index of max is not the current i in the loop, swap rows
        if (max_index > i):
            # Swap
            A[[i, max_index]] = A[[max_index, i]]   #nota che con A[[i, max_index]] selezioni due righe: la i-sima e  quella di max_index
            b[[i, max_index]] = b[[max_index, i]]

        # Normalize diagonal to 1.
        fact = A[i, i]
        A[i, :] /= fact
        b[i] /= fact

        for j in range(i + 1, N):
            # Subract upper rows, multiplied by suitable
            # factor, to generate an upper diagonal matrix
            # At the end of each loop, column i (set by loop above)
            # has become (1, 0, .... , 0)
            fact = A[j, i]
            A[j, :] -= fact * A[i, :]
            b[j] -= fact * b[i]

    # Final step: backsubstitution

    # Initialize solution vector
    x = np.zeros(len(b))
    # Gets last solution, x_(N-1)
    x[N - 1] = b[N - 1] / A[N - 1, N - 1]

    # Proceed backward, line by line
    for i in range(N - 2, -1, -1):
        v = np.dot(A[i, i + 1:N], x[i + 1:N])
        # if (i==1): print(i, i+1, N-1, A[i,i+1:N])
        x[i] = (b[i] - v) / A[i, i]

    return x

# Testing example for pivoting

A = [ [0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)
#print(b)

x1 = linsolve_gauss_pivot(A,b)
print("Result obtained with our script: \n",x1)

print("Ax -b =:", (np.dot(A,x1)-b) )

# Cross checking against python solve
x2 = solve(A,b)
print("\nResult obtained with numpy.linalg.solve: \n", x2)

# Another test
print("\n ####Altro test: matrice con due zeri sulla diagonale####")
A = [ [0, 1, 4, 1], [3, 0, -1, -1], [1, -4, 1, 5], [2, -2, 1, 0]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)
#print(b)
x1 = linsolve_gauss_pivot(A,b)
print("\nResult obtained with our script: \n",x1)
x2 = solve(A,b)
print("\nResult obtained with numpy.linalg.solve: \n", x2)

#Check with 5x5 array
print("\n ####Test con array 5 x 5####")
A = [ [0, 1, 4, 1,  7], [3, 0, -1, -1, 3], [1, -4, 1, 5, 0], [2, -2, 1, 0, -5], [-6, 3, 12, -10, 4]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7, 8]
b = np.array(b,dtype=float)
#print(b)
x1 = solve(A,b)
print("\nResult obtained with our script: \n",x1)
x2 = linsolve_gauss_pivot(A,b)
print("\nResult obtained with numpy.linalg.solve: \n", x2)
