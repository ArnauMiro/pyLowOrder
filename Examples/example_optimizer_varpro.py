import numpy as np
import matplotlib.pyplot as plt
import pyLOM

from   scipy.sparse import csr_matrix


def compute_error(B, a, Phi, H):
     """
     Compute the current residual, objective, and relative error.
     """
     K = Phi(a, t)
     print(K.shape, B.shape)
     residual = H - Phi(a, t).dot(B)
     objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
     error = np.linalg.norm(residual, "fro") / np.linalg.norm(H, "fro")
     return residual, objective, error

def compute_B(a, Phi, H):
    """
    Update B for the current a.
    """
    # Compute B using least squares.
    B = np.linalg.lstsq(Phi(a, t), H, rcond=None)[0]
    return B

def step(_lambda, scales_pvt=scales_pvt, rhs=rhs, ij_pvt=ij_pvt):
    """
    Helper function that, when given a step size _lambda,
    computes and returns the updated step and a vectors.
    """
    # Compute the step delta.
    rjac[IA:] = _lambda * np.diag(scales_pvt)
    delta = np.linalg.lstsq(rjac, rhs, rcond=None)[0]
    delta = delta[ij_pvt]
    # Compute the updated a vector.
    a_updated = a.ravel() + delta.ravel()
    #a_updated = self._push_eigenvalues(a_updated)
    return delta, a_updated

def Phi(alpha, t):
    return np.exp(np.outer(t, alpha))



## Data of the system
t = np.array([0, 0.1, 0.22, 0.31, 0.46, 0.50, 0.63, 0.78, 0.85, 0.97])
y = np.array([6.9842, 5.1851, 2.8907, 1.4199, -0.2473, -0.5243, -1.0156, -1.0260, -0.9165, -0.6805])

## Numerical perimeters
maxiter  = 30
tol      = 1e-6
lambda_i = 1.0
lambda_m = 52
lambda_u = 2.0

## Initialize variables and shapes
a_i      = np.array([0.5, 2, 3])
a        = a_i
_lambda  = lambda_i
M, N     = y.shape
IA       = len(a_i)
B        = compute_B(a)
U, S, Vh = pyLOM.POD.run(Phi(a, t))
residual, objective, error = compute_error(B, a)

# Initialize storage.
all_error = np.zeros(maxiter)
djac_matrix = np.zeros((M * N, IA), dtype="complex")
rjac = np.zeros((2 * IA, IA), dtype="complex")
scales = np.zeros(IA)

for itr in range(maxiter):
    # Build Jacobian matrix, looping over a indices.
    for i in range(IA):
        # Build the approximate expression for the Jacobian.
        dphi_temp = dPhi(a, t, i)
        ut_dphi   = csr_matrix(U.conj().T @ dphi_temp)
        uut_dphi  = csr_matrix(U @ ut_dphi)
        djac_a    = (dphi_temp - uut_dphi) @ B
        djac_matrix[:, i] = djac_a.ravel(order="F")
        
        # Compute the full expression for the Jacobian.
        transform = np.linalg.multi_dot([U, np.linalg.inv(S), Vh])
        dphit_res = csr_matrix(dphi_temp.conj().T @ residual)
        djac_b    = transform @ dphit_res
        djac_matrix[:, i] += djac_b.ravel(order="F")
        scales[i] = min(np.linalg.norm(djac_matrix[:, i]), 1)
        scales[i] = max(scales[i], 1e-6)

# Take a step using our initial step size init_lambda.
delta_0, a_0 = step(_lambda)
B_0 = compute_B(a_0)
residual_0, objective_0, error_0 = compute_error(B_0, a_0)

# Check actual improvement vs predicted improvement.
actual_improvement = objective - objective_0
pred_improvement = (0.5*np.linalg.multi_dot([delta_0.conj().T, djac_matrix.conj().T, rhs_temp])[0].real)
improvement_ratio = actual_improvement / pred_improvement
if error_0 < error:
    # Rescale lambda based on the improvement ratio.
    _lambda *= max(1 / 3, 1 - (2 * improvement_ratio - 1) ** 3)
    a, B = a_0, B_0
    residual, objective, error = residual_0, objective_0, error_0
else:
    # Increase lambda until something works.
    for _ in range(maxlam):
        _lambda *= lamup
        delta_0, a_0 = step(_lambda)
        B_0 = compute_B(a_0)
        residual_0, objective_0, error_0 = compute_error(B_0, a_0)
        if error_0 < error:
            break

    # ...otherwise, update and proceed.
    a, B = a_0, B_0
    residual, objective, error = residual_0, objective_0, error_0
    # Update SVD information.
    U, S, Vh = pyLOM.POD.run(Phi(a, t))
    # Record the current relative error.
    all_error[itr] = error
    # Update termination status and terminate if converged or stalled.
    converged = error < tol
    error_reduction = all_error[itr - 1] - all_error[itr]
    if converged:
        pyLOM.pprint(0, "Convergence reached!", flush=True)
    else:
        pyLOM.pprint(0, "Not converged!", flush=True)


plt.figure()
plt.plot(t, y, 'b', linewidth=2)
plt.plot(t, y, 'rx')

plt.show()