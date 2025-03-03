"""Modification of gaussian elimination + pivoting in order to remember the transformation on the
matrix A, usefulfor solve linear system in which A does not change but change only known
values vector b. A = LU with L = lower triangular, U = upper triangular."""

import numpy as np
from numpy.linalg import solve, inv


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
        # Find max element in column i
        ind = np.argmax(abs(A[:, i]))
        # If index of max is not the current i in the loop, swap rows
        if (ind > i):
            # Swap
            A[[i, ind]] = A[[ind, i]]
            # Recording all permutations in a list.
            # Needed to: permute rows of A, when comparing it with LU,
            # permute rows of L, permute entries of b when solving Ax = b
            swap.append(ind)
        else:
            swap.append(i)
        #########

        norm = A[i, i]
        #L[i, i] = 1.0

        # Normalizing diagonal of A to 1 for gauss elimination
        # A[i,:] /= norm

        for j in range(i + 1, N):
            # Updating L
            fact = A[j, i]
            L[j, i] = fact / norm
            # Gauss elimination to get U
            A[j, :] -= fact * (A[i, :] / norm)

    # Applying pivoting permutations to L
    # For each column j in L, I have to interchange rows
    # for all swaps that were performed at steps j+1...N
    # For example. If I am considering column 1, I swap rows
    # looking at all the permutations from step 2 onwards, but I DON'T
    # apply swaps made at steps 0 and 1.
    for j in range(N - 1):
        for i in range(j + 1, N):
            p = swap[i]
            L[i, j], L[p, j] = L[p, j], L[i, j]

    return L, A, swap

# Example for testing. No pivoting needed here
A = [ [1, 1, 4, 1], [1, 5, -1, -1], [1, -4, 2, 5], [2, -2, 1, 0]]
A = np.array(A,dtype=float)

#print(A,'\n')

Ain = np.copy(A)

#swaps = [ ]

L, U, swaps = LUdec(A)

disp = np.round(L,2)
print('L=', '\n', disp)
disp = np.round(U,2)
print('U=', '\n', disp)

print('Checking that LU = A (modulo swaps)', '\n')
LU = np.dot(L,U)
# Redoing the swaps in inverse order
# to "recompose" the original A
for i in range(len(swaps)-1,-1,-1):
    p = swaps[i]
    LU[[i,p]]=LU[[p,i]]
disp = np.round(LU,2)
print('LU = \n',disp)
print(' ')
print('Input A = \n', Ain)

# Testing example with pivoting

print("\nTesting example with pivoting")
A = [[0, 1, 4, 2], [3, 7, -1, 6], [1, -4, 1, 10], [0, 11, 0, 2]]
A = np.array(A,dtype=float)
#print(A)

Ain = np.copy(A)

L2, U2, swaps = LUdec(A)

disp = np.round(L2,2)
print('L=', '\n', disp)
disp = np.round(U2,2)
print('U=', '\n', disp)

print('Checking that LU = A', '\n')
LU = np.dot(L2,U2)
# Redoing the swaps in inverse order
# to "recompose" the original A
for i in range(len(swaps)-1,-1,-1):
    p = swaps[i]
    LU[[i,p]]=LU[[p,i]]
disp = np.round(LU,2)
print("LU =", disp)
print(' ')
print("A =", Ain)