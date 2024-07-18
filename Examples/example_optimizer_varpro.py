import numpy as np
import pyLOM
from   scipy.sparse import csr_matrix

def _exponentials(alpha, t):
	'''
	Matrix of exponentials
	'''
	return np.exp(np.outer(t, alpha))

def _dExponentials(alpha, t, i):
	"""
	Derivatives of the matrix of exponentials.
	"""
	m = len(t)
	n = len(alpha)
	if i < 0 or i > n - 1:
		raise ValueError("Invalid index i given to exp_function_deriv.")
	A = np.multiply(t, np.exp(alpha[i] * t))
	return csr_matrix((A, (np.arange(m), np.full(m, fill_value=i))), shape=(m, n))

def _varpro_opt_compute_B(_phi, H):
	"""
	Update B for the current a.
	"""
	# Compute B using least squares.
	B = np.linalg.lstsq(_phi, H, rcond=None)[0]
	return B

def _varpro_opt_compute_error(H, _phi, B):
	"""
	Compute the current residual, objective, and relative error.
	"""
	residual = H - _phi.dot(B)
	objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
	error = np.linalg.norm(residual, "fro") / np.linalg.norm(H, "fro")
	return residual, objective, error

def variable_projection_optimizer(H, iniReal, iniImag, time, maxiter=30, _lambda=1.0, lambda_m=52, lambda_u=2, eps_stall=1e-12, tol=1e-6):

	rH, cH      = H.shape
	alpha       = iniReal +1j*iniImag
	neig        = alpha.shape[0]
	_phi        = _exponentials(alpha, time)
	B           = _varpro_opt_compute_B(_phi, H)
	res,obj,err = _varpro_opt_compute_error(H, _phi, B)
	Up,sp,VTp   = np.linalg.svd(_phi, full_matrices=False)
	Sp          = np.diag(sp)

	all_error = np.zeros(maxiter)
	djac_matrix = np.zeros((rH*cH, neig), dtype="complex")
	rjac = np.zeros((2*neig, neig), dtype="complex")
	scales = np.zeros(neig)
	for ii in range(maxiter):
		for ieig in range(neig):
			# Build the approximate expression for the Jacobian.
			dphi_temp = _dExponentials(alpha, time, ieig)
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

	return scales

data = np.load('test_varpro.npz')

H, iniReal, iniImag, t = data['H'], data['iniReal'], data['iniImag'], data['t']

pyLOM.cr_start('non_compiled', 0)
var1 = variable_projection_optimizer(H, iniReal, iniImag, t)
pyLOM.cr_stop('non_compiled', 0)

var2 = pyLOM.math.variable_projection_optimizer(H, iniReal, iniImag, t)
print(np.max(np.abs(var1-var2)))

pyLOM.cr_info()