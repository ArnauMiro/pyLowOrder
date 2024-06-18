#!/usr/bin/env python
#
# POD analysis.
#
# Last revision: 14/06/2024
import numpy as np
import matplotlib.pyplot as plt
import pyLOM


def generate_X():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 200)
    z = np.linspace(0, 1, 300)
    XX, YY, ZZ = np.meshgrid(x, y, z)
    F = np.sin(2*np.pi*(XX + ZZ)) + np.cos(2*np.pi*(2*YY + ZZ))
    return F

# Generate synthetic database
X = generate_X()
print('Shape of synthetic database:',X.shape) # (200, 100, 300)

#########################################################################################################
## However, pyLOM takes a database of the shape (spatial_dims,time_dims)
## thus we need to reshape this matrix accordingly.
#########################################################################################################
X_rsh = X.reshape([-1, X.shape[-1]])
print('Shape of reshaped database:',X_rsh.shape) # (200*100 = 20000, 300)


#########################################################################################################
## POD using the whole database.
#########################################################################################################
PSI, S, V = pyLOM.POD.run(X_rsh, remove_mean=False)
PSI_trunc, S_trunc, V_trunc = pyLOM.POD.truncate(PSI,S,V,r=1e-6)
X_rsh_rec = pyLOM.POD.reconstruct(PSI_trunc, S_trunc, V_trunc)
print('Shape of reconstructed database:',X_rsh_rec.shape) # (200*100 = 20000, 300)
print("RMSE for whole database =", pyLOM.math.RMSE(X_rsh_rec,X_rsh))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].pcolor(X_rsh[:,0].reshape((200,100)))
axs[0].set_title("Original matrix")
axs[1].pcolor(X_rsh_rec[:,0].reshape((200,100)))
axs[1].set_title("Reconstructed matrix")
plt.show()


########################################################################################################
# POD using just the first "snapshot" (as if we are doing data compression).
########################################################################################################
X_rsh_2 = X_rsh[:,:1].copy()
PSI, S, V = pyLOM.POD.run(X_rsh_2, remove_mean=False)
PSI_trunc, S_trunc, V_trunc = pyLOM.POD.truncate(PSI,S,V,r=1e-6)
X_rsh_rec = pyLOM.POD.reconstruct(PSI_trunc, S_trunc, V_trunc)
print('Shape of reconstructed database:',X_rsh_rec.shape) # (200*100 = 20000, 0)
print("RMSE for whole database =", pyLOM.math.RMSE(X_rsh_rec,X_rsh_2))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].pcolor(X_rsh_2[:,0].reshape((200,100)))
axs[0].set_title("Original matrix")
axs[1].pcolor(X_rsh_rec[:,0].reshape((200,100)))
axs[1].set_title("Reconstructed matrix")
plt.show()