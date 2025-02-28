from __future__ import print_function

import numpy as np
import os
from contextlib import redirect_stdout

from ..utils  import cr, raiseError
from ..vmmath import temporal_mean, subtract_mean
from ..POD    import run, truncate

class GappyPOD:
    def __init__(
        self,
        centered=False,
        apply_truncation=False,
        truncation_param=-0.99,
        reconstruction_method="standard",
        ridge_lambda=0.01,
    ):
        """
        Gappy POD model for reconstructing incomplete data using POD modes.

        Args:
            centered (bool): Whether to center the data by subtracting the mean.
            apply_truncation (bool): Whether to apply truncation.
            truncation_method (float or int): Threshold for truncation.
            reconstruction_method (str): Reconstruction method ('standard' or 'ridge').
            ridge_lambda (float): Regularization parameter for ridge reconstruction.
        """
        # Validate reconstruction method
        if not reconstruction_method.lower() in ["standard", "ridge"]:
            raiseError("Reconstruction method must be either 'standard' or 'ridge'.")

        self.centered            = centered
        self.truncate            = apply_truncation
        self.truncation_param    = truncation_param
        self.reconstruction_type = reconstruction_method
        self.ridge_lambda        = ridge_lambda

        # Attributes to store fitted data
        self.mean               = None
        self.U_truncated_scaled = None

    def _ridge_regresion(self, masked_U, gappy_input):
        I = np.sqrt(self.ridge_lambda) * np.eye(masked_U.shape[1])
        augmented_U = np.vstack([masked_U, I])
        augmented_input = np.vstack([gappy_input[:, None], np.zeros((I.shape[0], 1))])
        coef = np.linalg.lstsq(augmented_U, augmented_input, rcond=None)[0]
        return coef

    @cr('GPOD.fit')
    def fit(self, snapshot_matrix: np.ndarray, **kwargs) -> None:
        """
        Fit the Gappy POD model using the snapshot matrix.

        Args:
            snapshot_matrix  (np.ndarray): Training matrix [n_features, n_samples].
        """
        # Center data if required
        self.mean = temporal_mean(snapshot_matrix) if self.centered else np.zeros(snapshot_matrix.shape[0])
        X = subtract_mean(snapshot_matrix,self.mean)

        # Compute SVD through POD this way we can reuse some nice features
        self.U_truncated, self.S_truncated, VT = run(X, remove_mean=False, **kwargs)

        # Apply truncation if specified
        if self.truncate:
            self.U_truncated, self.S_truncated, _ = truncate(U,S,VT,self.truncation_param)

        # Masked and scaled POD modes
        self.U_truncated_scaled = self.U_truncated * self.S_truncated

    def predict(self, gappy_vector: np.ndarray) -> np.ndarray:
        """
        Reconstruct missing data using the fitted Gappy POD model.

        Args:
            gappy_vector (np.ndarray): Sparse vector with missing values.

        Returns:
            np.ndarray: Reconstructed data vector.
        """
        if self.U_truncated_scaled is None:
            raise ValueError("The model must be fitted before calling predict.")

        # Prepare the masked input
        gappy_input = gappy_vector - (self.mean * (gappy_vector != 0))
        mask = (gappy_vector != 0).astype(int)  # Binary mask for observed data
        PT_U = mask[:, None] * self.U_truncated_scaled
        gappy_input_reshaped = gappy_input[:, None]

        # Solve for coefficients
        if self.reconstruction_type == "standard":
            coef = np.linalg.lstsq(PT_U, gappy_input_reshaped, rcond=None)[0]
        else:  # Ridge Gappy POD
            coef = self._ridge_regresion(PT_U, gappy_input)

        # Reconstruct missing data
        vector_reconstructed = (self.U_truncated_scaled @ coef).flatten() + self.mean
        return vector_reconstructed

    def reconstruct_full_set(
        self, incomplete_snapshot: np.ndarray, iter_num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iteratively reconstruct an incomplete snapshot matrix using Gappy POD.

        Args:
            incomplete_snapshot (np.ndarray): Snapshot matrix with missing values (n_features, n_samples).
            iter_num (int): Number of iterations for the iterative reconstruction process.

        Returns:
            tuple: Reconstructed snapshot matrix, eigenvalue spectrum across iterations,
                   and cumulative energy.
        """
        # Step 1: Create mask for missing values
        mask_incomplete = (incomplete_snapshot != 0).astype(int)

        # Step 2: Compute row-wise mean for non-missing values
        g_i_mean = np.sum(incomplete_snapshot, axis=1) / np.sum(mask_incomplete, axis=1)

        # Initialize the reconstructed snapshot matrix
        h_recons = np.where(
            mask_incomplete == 0, g_i_mean[:, np.newaxis], incomplete_snapshot
        )

        # Step 3: Initialize arrays to store results
        eig_spec_iter = np.empty((h_recons.shape[1], iter_num))
        c_e = np.empty((h_recons.shape[1], iter_num))

        # Iterative reconstruction
        for k in range(iter_num):
            # Fit the model with the current reconstructed snapshot matrix
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull):
                    self.fit(h_recons)

            # Reconstruct all snapshots
            snapshots_recons = np.empty(h_recons.shape)
            for i in range(h_recons.shape[1]):
                gappy_vector = np.copy(incomplete_snapshot[:, i])
                snapshots_recons[:, i] = self.predict(gappy_vector)

            # Update the reconstruction matrix
            h_recons = np.copy(incomplete_snapshot)
            for j in range(h_recons.shape[1]):
                where_zero = np.where(mask_incomplete[:, j] == 0)[0]
                h_recons[:, j][where_zero] = snapshots_recons[:, j][where_zero]

            # Compute the eigenvalue spectrum and cumulative energy
            _, S, _ = run(h_recons, remove_mean=True)
            eig_spec_iter[:, k] = S**2 / np.sum(S**2)
            c_e[:, k] = np.cumsum(S) / np.sum(S)

        return h_recons, eig_spec_iter, c_e
