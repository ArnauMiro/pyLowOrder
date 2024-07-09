import numpy as np
import matplotlib.pyplot as plt
import pyLOM

from   scipy.linalg            import qr
from   scipy.sparse            import csr_matrix

def f1(x, t):
    return 1.0/np.cosh(x + 3) * np.cos(2.3 * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.sin(2.8 * t)

def compute_error(H, _phi, B):
     """
     Compute the current residual, objective, and relative error.
     """
     residual = H - _phi.dot(B)
     objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
     error = np.linalg.norm(residual, "fro") / np.linalg.norm(H, "fro")
     return residual, objective, error

def compute_B(_phi, H):
    """
    Update B for the current a.
    """
    # Compute B using least squares.
    B = np.linalg.lstsq(_phi, H, rcond=None)[0]
    return B

def step(_lambda, neig, alpha, scales_pvt, rhs, ij_pvt):
    """
    Helper function that, when given a step size _lambda,
    computes and returns the updated step and a vectors.
    """
    # Compute the step delta.
    rjac[neig:] = _lambda * np.diag(scales_pvt)
    delta = np.linalg.lstsq(rjac, rhs, rcond=None)[0]
    delta = delta[ij_pvt]
    # Compute the updated a vector.
    a_updated = alpha.ravel() + delta.ravel()
    #a_updated = self._push_eigenvalues(a_updated)
    return delta, a_updated

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
_lambda   = 1.0
delays    = 2
maxiter   = 30
lambda_m  = 52
lambda_u  = 2
eps_stall = 1e-12
tol       = 1e-6

## Create time delay embedding
hankel_X  = pyLOM.math.pseudo_hankel_matrix(X.T, delays)
delay_t   = t[: -delays + 1]
delay_t   = delay_t[:-1]

#muReal, muImag, Phi, bJov = pyLOM.DMD.run(hankel_X, r=4, remove_mean=False)
iniReal, iniImag, H = pyLOM.DMD.run_optimized(hankel_X, t=delay_t, r=4, constraints=None, remove_mean=False)

rH, cH      = H.shape
alpha       = iniReal +1j*iniImag
neig        = alpha.shape[0]
_phi        = pyLOM.math.exponentials(alpha, delay_t)
B           = compute_B(_phi, H)
Up,sp,VTp   = np.linalg.svd(_phi, full_matrices=False)
Sp          = np.diag(sp)
res,obj,err = compute_error(H, _phi, B)

all_error = np.zeros(maxiter)
djac_matrix = np.zeros((rH*cH, neig), dtype="complex")
rjac = np.zeros((2*neig, neig), dtype="complex")
scales = np.zeros(neig)
for ii in range(maxiter):
    for ieig in range(neig):
        # Build the approximate expression for the Jacobian.
        dphi_temp = pyLOM.math.dExponentials(alpha, delay_t, ieig)
        ut_dphi   = csr_matrix(Up.conj().T @ dphi_temp)
        uut_dphi  = csr_matrix(Up @ ut_dphi)
        djac_a    = (dphi_temp - uut_dphi) @ B
        djac_matrix[:, ieig] = djac_a.ravel(order="F")

        # Compute the full expression for the Jacobian.
        transform = np.linalg.multi_dot([Up, np.linalg.inv(Sp), VTp])
        dphit_res = csr_matrix(dphi_temp.conj().T @ res)
        djac_b    = transform @ dphit_res
        djac_matrix[:, ieig] += djac_b.ravel(order="F")
        scales[ieig] = min(np.linalg.norm(djac_matrix[:, ieig]), 1)
        scales[ieig] = max(scales[ieig], 1e-6)

    rhs_temp = np.copy(res.ravel(order="F"))[:, None]
    q_out, djac_out, j_pvt = qr(djac_matrix, mode="economic", pivoting=True)
    ij_pvt = np.arange(neig)
    ij_pvt = ij_pvt[j_pvt]
    rjac[:neig] = np.triu(djac_out[:neig])
    rhs_top = q_out.conj().T.dot(rhs_temp)
    scales_pvt = scales[j_pvt[:neig]]
    rhs = np.concatenate((rhs_top[:neig], np.zeros(neig, dtype="complex")), axis=None)

    # Take a step using our initial step size init_lambda.
    delta_0, alpha_0 = step(_lambda, neig, alpha, scales_pvt, rhs, ij_pvt)
    _phi = pyLOM.math.exponentials(alpha_0, delay_t)
    B_0  = compute_B(_phi, H)
    res_0, obj_0, err_0 = compute_error(H, _phi, B_0)
    # Check actual improvement vs predicted improvement.
    actual_improvement = obj - obj_0
    pred_improvement   = (0.5*np.linalg.multi_dot([delta_0.conj().T, djac_matrix.conj().T, rhs_temp])[0].real)
    improvement_ratio  = actual_improvement/pred_improvement
    if err_0 < err:
        # Rescale lambda based on the improvement ratio.
        _lambda *= max(1 / 3, 1 - (2 * improvement_ratio - 1) ** 3)
    else:
        # Increase lambda until something works.
        for _ in range(lambda_m):
            _lambda *= lambda_u
            delta_0, alpha_0 = step(_lambda, neig, alpha, scales_pvt, rhs, ij_pvt)
            _phi = pyLOM.math.exponentials(alpha_0, delay_t)
            B_0 = compute_B(_phi, H)
            res_0, obj_0, err_0 = compute_error(H, _phi, B_0)
            if err_0 < err:
                break
        if err_0 > err:
            pyLOM.pprint(0, "Not converged!!", flush=True)
            break

    ## Update information
    alpha, B      = alpha_0, B_0
    res, obj, err = res_0, obj_0, err_0
    _phi          = pyLOM.math.exponentials(alpha, delay_t)
    Up, sp, VTp   = np.linalg.svd(_phi, full_matrices=False)
    Sp            = np.diag(sp)

    # Record the current relative error.
    all_error[ii] = err

    # Update termination status and terminate if converged or stalled.
    if err < tol:
        pyLOM.pprint(0, "Convergence reached!", flush=True)
        break
    if (ii > 0) and ((all_error[ii-1]-all_error[ii]) < eps_stall*all_error[ii - 1]):
        pyLOM.pprint(0, "Stalled!", flush=True)
        break

print(alpha)
STOP


#delta, omega = pyLOM.DMD.frequency_damping(muReal, muImag, dt)
#print(muReal+muImag*1j)
#print(np.round(np.log(muReal+muImag*1j)/dt, decimals=12))
#pyLOM.DMD.ritzSpectrum(muReal, muImag)

plt.show()