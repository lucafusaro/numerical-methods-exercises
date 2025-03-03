import numpy as np
from numpy.linalg import solve
from numpy.linalg import inv


class solveinput:

    # Defining an object that contains all the input ingredients to solve Ax = b.
    # That is: L, U and the list which records the swaps when doing the LU decomp

    def __init__(self, L, U, rec):
        self.lower = L
        self.upper = U
        self.swaps = rec


def LUdec(A):
    # Given square array A, finds its LU decomposition
    # Applies partial pivoting

    N = len(A)

    # Vector of diagonal elements of A
    # To be updated during pivoting
    diag = np.empty(N)

    L = np.identity(N)

    swap = []

    for i in range(N):

        # Partial pivoting
        #######################
        # Collecting diagonal elements of A in vector 'diag'
        # Find max element in diag
        ind = np.argmax(abs(A[:, i]))
        # If index of max is not the current i in the loop, swap rows
        if (ind > i):
            # Swap
            A[[i, ind]] = A[[ind, i]]
            # Recording all permutations in a list
            # Needed for later
            swap.append(ind)
        else:
            swap.append(i)
        #########

        norm = A[i, i]
        L[i, i] = 1.0

        for j in range(i + 1, N):
            # Updating L
            fact = A[j, i]
            L[j, i] = fact / norm
            # Gauss elimination to get U
            A[j, :] -= fact * (A[i, :] / norm)

    # Applying pivoting permutations to L
    # For each column j in L, I have to interchange rows
    # for all swaps that were performed at steps j+1...N
    # For example. If am considering column 1, I swap rows
    # loking at all the permutations from step 2 onwards, but I DON'T
    # apply swaps made at steps 0 and 1.
    for j in range(N - 1):
        for i in range(j + 1, N):
            p = swap[i]
            L[i, j], L[p, j] = L[p, j], L[i, j]

    LU = solveinput(L, A, swap)

    return LU


def linsolve_LU(LU, b):
    L = LU.lower
    U = LU.upper
    swaps = LU.swaps

    N = len(b)

    # Applying all swaps to b (can be done at once)
    for i in range(len(swaps)):
        b[i], b[swaps[i]] = b[swaps[i]], b[i]

    ######################################################
    # Now solving LUx = b via double backsubstitution
    # First solve Ly = b, to get y. Then solve Ux = y,
    # to get the desired solution

    # Solving Ly = b.
    # Initialize solution vector
    y = np.zeros(N)
    # Gets first solution
    y[0] = b[0] / L[0, 0]
    # Proceed forward, line by line
    for i in range(1, N):
        v = np.dot(L[i, 0:i], y[0:i])
        y[i] = (b[i] - v) / L[i, i]

    # Solving Ux = y
    # Initialize solution vector
    x = np.zeros(N)
    # Gets last solution, x_(N-1)
    x[N - 1] = y[N - 1] / U[N - 1, N - 1]
    # Proceed backward, line by line
    for i in range(N - 2, -1, -1):
        v = np.dot(U[i, i + 1:N], x[i + 1:N])
        # if (i==1): print(i, i+1, N-1, A[i,i+1:N])
        x[i] = (y[i] - v) / U[i, i]

    return x

# Testing example
A = [ [2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)
#print(b)

LU = LUdec(A)

#disp = np.round(U,2)
#print('U =', '\n', disp)
#disp = np.round(L,2)
#print('L=', '\n', disp, '\n')

x = linsolve_LU(LU,b)

print('My solution =', x)

# Cross checking against numpy solver

A = [ [2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)
#print(b)

x = solve(A,b)
print('Numpy solution = ', x)

# Testing example for pivoting

A = [ [0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)
#print(b)

LU = LUdec(A)

x = linsolve_LU(LU,b)

print('My solution =', x)

A = [ [0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]]
A = np.array(A,dtype=float)
#print(A)

b = [-4, 3, 9, 7]
b = np.array(b,dtype=float)

y = solve(A,b)
print('Numpy solution =', y)

# Exploiting LU to solve systems with same A and different b

A_in = np.array([ [0, 1, 4, 1, 0], [3, 4, -1, -1, 7], [1, -4, 1, 5, -14], [2, -2, 1, 3, -1], [21, -6, -9, 8, 1] ], dtype=float)
b_in = np.array([0, 0, 3, -6, 21], float)

A = np.copy(A_in)

# Do this only once for any b
LU = LUdec(A)

###########################
b = np.copy(b_in)
x = linsolve_LU(LU,b)
print('My solution =', x)

A = np.copy(A_in)
b = np.copy(b_in)
y = solve(A,b)
print('Numpy solution =', y, '\n')

b_in = np.array([1, 7, 102, -6, -49.5],float)
b = np.copy(b_in)
x = linsolve_LU(LU,b)
print('My solution =', x)

A = np.copy(A_in)
b = np.copy(b_in)
y = solve(A,b)
print('Numpy solution =', y, '\n')