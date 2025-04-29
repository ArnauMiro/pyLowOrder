#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# GPOD Module
#
# Last rev: 29/04/2025

from __future__ import print_function

import numpy as np

from ..utils.gpu import cp
from ..utils     import cr_nvtx as cr, raiseError
from ..vmmath    import temporal_mean, subtract_mean, matmul, vector_sum, least_squares, ridge_regresion
from ..POD       import run, truncate


class GappyPOD:
	"""
	This class implements the Gappy POD algorithm. Gappy POD model for reconstructing 
	incomplete data using POD basis modes.

	Attributes:
		centered (bool): If True, subtract the mean from inputs.
		truncate (bool): If True, apply modal truncation based on a threshold.
		truncation_param (float or int): Parameter controlling truncation.
		reconstruction_type (callable): Function to compute POD coefficients.
		ridge_lambda (float): Regularization weight for ridge reconstruction.
		mean (np.ndarray or cp.ndarray): Mean vector computed during fitting.
		U_truncated (np.ndarray or cp.ndarray): Truncated POD modes.
		S_truncated (np.ndarray or cp.ndarray): Corresponding singular values.
		U_truncated_scaled (np.ndarray or cp.ndarray or cp.ndarray): Modes scaled by singular values.
	"""
	def __init__(
		self,
		centered=False,
		apply_truncation=False,
		truncation_param=-0.99,
		reconstruction_method="standard",
		ridge_lambda=0.01,
	):
		"""
		Initialize a Gappy POD instance.

		Args:
			centered (bool): Whether to subtract the mean from data.
			apply_truncation (bool): Whether to truncate POD modes.
			truncation_param (float or int): Parameter controlling truncation (r):
				If r >= 1, it is treated as the number of modes.
				If r < 1 and r > 0 it is treated as the residual target.
				If r < 1 and r < 0 it is treated as the fraction of cumulative energy to retain.
				Note:  must be in (0,-1] and r = -1 is valid
			reconstruction_method (str): 
				"standard" for least squares.
				"ridge" for ridge regression.
			ridge_lambda (float): Regularization parameter for ridge solver.

		Raises:
			ValueError: If `reconstruction_method` is not "standard" or "ridge".
		"""
		# Validate reconstruction method
		if not reconstruction_method.lower() in ["standard", "ridge"]:
			raiseError("Reconstruction method must be either 'standard' or 'ridge'.")

		self.centered            = centered
		self.truncate            = apply_truncation
		self.truncation_param    = truncation_param
		self.reconstruction_type = least_squares if reconstruction_method.lower() == "standard" else lambda a,b : ridge_regresion(a,b,ridge_lambda)

		# Attributes to store fitted data
		self.mean               = None
		self.U_truncated_scaled = None

	@cr('GPOD.fit')
	def fit(self, snapshot_matrix: np.ndarray, **kwargs) -> None:
		"""
		Fit the Gappy POD model using a complete snapshot matrix.

		Computes the POD basis (via SVD) of the input data, optionally centers
		the data and truncates modes.

		Args:
			snapshot_matrix (np.ndarray or cp.ndarray):
				Array of shape (n_features, n_samples) containing training snapshots.
			**kwargs: Additional keyword arguments forwarded to the POD `run` function.

		Returns:
			None
		"""
		cnp = cp if type(snapshot_matrix) is cp.ndarray else np
		# Center data if required
		self.mean = temporal_mean(snapshot_matrix) if self.centered else cnp.zeros(snapshot_matrix.shape[0])
		X = subtract_mean(snapshot_matrix,self.mean)

		# Compute SVD through POD this way we can reuse some nice features
		self.U_truncated, self.S_truncated, VT = run(X, remove_mean=False, **kwargs)

		# Apply truncation if specified
		if self.truncate:
			self.U_truncated, self.S_truncated, _ = truncate(self.U_truncated,self.S_truncated,VT,self.truncation_param)

		# Masked and scaled POD modes
		self.U_truncated_scaled = self.U_truncated * self.S_truncated

	@cr('GPOD.predict')
	def predict(self, gappy_vector: np.ndarray) -> np.ndarray:
		"""
		Reconstruct a single vector with missing entries.

		Uses the fitted POD basis to estimate missing values in the input vector.

		Args:
			gappy_vector (np.ndarray or cp.ndarray):
				Input vector of length n_features where missing entries are zeroed.

		Returns:
			np.ndarray or cp.ndarray:
				Full reconstructed vector of length n_features.

		Raises:
			RuntimeError: If the model has not been fitted prior to prediction.
		"""
		if self.U_truncated_scaled is None:
			raiseError("The model must be fitted before calling predict.")

		# Prepare the masked input
		mask        = (gappy_vector != 0) # Binary mask for observed data
		gappy_input = gappy_vector - self.mean*mask
		PT_U        = mask[:,None]*self.U_truncated_scaled

		# Solve for coefficients
		coef = self.reconstruction_type(PT_U,gappy_input)

		# Reconstruct missing data
		vector_reconstructed = matmul(self.U_truncated_scaled,coef[:,None]).flatten() + self.mean
		return vector_reconstructed

	@cr('GPOD.reconstruct')
	def reconstruct_full_set(
		self, incomplete_snapshot: np.ndarray, iter_num: int
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Iteratively reconstruct an entire snapshot matrix with missing data.

		Applies the Gappy POD algorithm for a specified number of iterations,
		updating missing entries based on previous reconstructions.

		Args:
			incomplete_snapshot (np.ndarray or cp.ndarray):
				Array of shape (n_features, n_samples) with zeros indicating missing values.
			iter_num (int): Number of iterations to perform.

		Returns:
			tuple:
				- reconstructed (np.ndarray or cp.ndarray):
					Reconstructed snapshot matrix of same shape as input.
				- eig_spec_iter (np.ndarray or cp.ndarray):
					Eigenvalue spectrum (normalized squared singular values)
					at each iteration, shape (n_samples, iter_num).
				- cumulative_energy (np.ndarray or cp.ndarray):
					Cumulative energy content of modes per iteration,
					shape (n_modes, iter_num).
		"""
		cnp = cp if type(incomplete_snapshot) is cp.ndarray else np
		# Step 1: Create mask for missing values
		mask_incomplete = (incomplete_snapshot != 0).astype(int)

		# Step 2: Compute row-wise mean for non-missing values
		g_i_mean = cnp.array([vector_sum(incomplete_snapshot[i, :]) / vector_sum(mask_incomplete[i, :].astype(incomplete_snapshot.dtype))
		for i in range(incomplete_snapshot.shape[0])])

		# Initialize the reconstructed snapshot matrix
		h_recons = incomplete_snapshot.copy()
		for i in range(h_recons.shape[0]):
			h_recons[i, mask_incomplete[i] == 0] = g_i_mean[i]

		# Step 3: Initialize arrays to store results
		eig_spec_iter = cnp.empty((h_recons.shape[1], iter_num))
		c_e = cnp.empty((h_recons.shape[1], iter_num))

		# Iterative reconstruction
		for k in range(iter_num):
			# Fit the model with the current reconstructed snapshot matrix
			self.fit(h_recons)

			# Reconstruct all snapshots
			snapshots_recons = cnp.empty(h_recons.shape)
			for i in range(h_recons.shape[1]):
				gappy_vector = cnp.copy(incomplete_snapshot[:, i])
				snapshots_recons[:, i] = self.predict(gappy_vector)

			# Update the reconstruction matrix
			h_recons = incomplete_snapshot.copy()
			for j in range(h_recons.shape[1]):
				mask = mask_incomplete[:, j] == 0
				h_recons[mask, j] = snapshots_recons[mask, j]

			# Compute the eigenvalue spectrum and cumulative energy
			_, S, _ = run(h_recons, remove_mean=True)
			eig_spec_iter[:, k] = S**2/ vector_sum(S**2,0)
			normS = vector_sum(S,0)
			cumulative = 0
			for ii in range(S.shape[0]):
				cumulative += S[ii]
				c_e[ii, k] = cumulative / normS
				
		return h_recons, eig_spec_iter, c_e
