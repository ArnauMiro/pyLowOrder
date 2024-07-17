import pyLOM
import numpy as np

t     = np.arange(20, dtype=np.double)
areal = np.random.random(10)
aimag = np.random.random(10)
alpha = areal + 1j*aimag

pyLOM.cr_start('numpy exponentials', 0)
A = np.exp(np.outer(alpha, t))
pyLOM.cr_stop('numpy exponentials', 0)

B = pyLOM.math.exponentials(alpha, t)

print(A-B)

pyLOM.cr_info()
