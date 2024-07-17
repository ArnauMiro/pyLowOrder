import numpy as np
import matplotlib.pyplot as plt
import pyLOM

def f1(x, t):
    return 1.0/np.cosh(x + 3) * np.cos(2.3 * t)

def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.sin(2.8 * t)

nx = 65  # number of grid points along space dimension
nt = 129  # number of grid points along time dimension

# Define the space and time grid for data collection.
x = np.linspace(-5, 5, nx)
t = np.linspace(0, 4 * np.pi, nt)
xgrid, tgrid = np.meshgrid(x, t)
dt = t[1] - t[0]  # time step between each snapshot

# Data consists of 2 spatiotemporal signals.
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
#X = X1 + X2

X = np.load('Xn_optimized.npz')['arr_0']

titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$"]
data = [X1, X2, X]

fig = plt.figure(figsize=(17, 6), dpi=200)
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.colorbar()

## Numerical parameters
delays = 2

## Create time delay embedding
hankel_X  = pyLOM.math.pseudo_hankel_matrix(X.T, delays)
delay_t   = t[: -delays + 1]
delay_t   = delay_t[:-1]

muReal, muImag, Phi_delay, b = pyLOM.DMD.run_optimized(hankel_X, delay_t, r=4, constraints=None, remove_mean=False)

## Postprocessing
pyLOM.DMD.ritzSpectrum(muReal, muImag)
disc_eigs = np.exp((muReal*muImag*1j) * dt)

Phi = np.average(Phi_delay.reshape((delays, Phi_delay.shape[0] // delays, Phi_delay.shape[1])), axis=0)

plt.figure()
plt.plot(x, Phi[:,0])
plt.xlabel('x')
plt.ylabel('Mode 1')

plt.figure()
plt.plot(x, Phi[:,1])
plt.xlabel('x')
plt.ylabel('Mode 2')

plt.figure()
plt.plot(x, Phi[:,2])
plt.xlabel('x')
plt.ylabel('Mode 3')


pyLOM.cr_info()
plt.show()