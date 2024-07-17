import numpy as np
import pyLOM

def _exponentials(alpha, t):
	'''
	Matrix of exponentials
	'''
	return np.exp(np.outer(t, alpha))

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


	return res

data = np.load('test_varpro.npz')

H, iniReal, iniImag, t = data['H'], data['iniReal'], data['iniImag'], data['t']

pyLOM.cr_start('non_compiled', 0)
var1 = variable_projection_optimizer(H, iniReal, iniImag, t)
pyLOM.cr_stop('non_compiled', 0)

var2 = pyLOM.math.variable_projection_optimizer(H, iniReal, iniImag, t)

print(var1-var2)

pyLOM.cr_info()