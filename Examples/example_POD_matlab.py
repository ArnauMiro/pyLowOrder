#!/usr/bin/env python
#
# Example of POD following the MATLAB script.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pyLOM


## Data loading
UALL = np.load('DATA/UALL.npy')
X    = UALL 


## Compute POD after subtracting mean (i.e., do PCA)
Uavg = pyLOM.POD.temporal_mean(X)
X_m  = pyLOM.POD.subtract_mean(X,Uavg)
Y    = X_m

PSI,S,V = pyLOM.POD.svd(Y)
# PSI are POD modes

# Plot accumulative S
plt.figure(figsize=(8,6),dpi=100)


'''
figure(500)
n_snaps = 1:N;
accumulative_S = zeros(1,N);
diag_S = diag(S);

for i = n_snaps
    accumulative_S(i) = norm(diag_S(i:N),2)/norm((diag_S),2);
end

semilogy(n_snaps,accumulative_S, 'bo')
ylabel('varepsilon1')
xlabel('Truncation size')
title('Tolerance')
ylim([0 1])
hold on
'''

## Show and print timings
pyLOM.cr_info()
plt.show()