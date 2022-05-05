import numpy as np

A = np.array([4, 8, 5, 3, 6, 10])
nr = A.shape[0]
B = np.array([7, 4, 8, 1, 9, 0])
n = np.flip(A.argsort())
C = B[n]
jj  = 0
auxs2 = B[jj]
B[jj] = B[n[jj]]
auxs = auxs2
for ii in range(nr-1):
    for kk in range(nr):
        if jj == n[kk]:
            jj = kk
            break
    auxs2 = B[jj]
    B[jj] = auxs
    auxs  = auxs2

print(C)
print(B)
