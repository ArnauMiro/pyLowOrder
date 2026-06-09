#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# RBF-ELM (Radial Basis Function-augmented Extreme Learning Machine) class.
#
# Last rev: 09/06/2026

import math, os, json, pickle, numpy as np, torch, torch.nn as nn

from typing                 import Dict, Optional, Tuple, List
from sklearn.neighbors      import NearestNeighbors
from torch.utils.data       import DataLoader
from sklearn.cluster        import MiniBatchKMeans

from ..                     import DEVICE, PIN_MEMORY, set_seed
from ..                     import Dataset as NNDataset, RobustScaler
from ...utils.errors        import raiseError
from ...                    import pprint, cr

try:
    from optuna.exceptions import TrialPruned
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


class RBFLayer(nn.Module):
    r"""
    Radial Basis Function (RBF) layer with Gaussian kernels. 
    
    Args:
        n_centers (int): Number of RBF neurons (hidden layer size).
        input_size (int): Dimensionality of input features.
    """
    def __init__(self, n_centers: int, input_size: int):
        super().__init__()
        self.n_centers  = n_centers
        self.input_size = input_size

        self.register_buffer("centers", torch.zeros(n_centers, input_size))
        self.register_buffer("gamma", torch.ones(n_centers))

    def set_gamma(self, gamma):
        r"""
        Set the gamma parameter(s) for the RBF layer.

        Args:
            gamma (float or torch.Tensor): If a float or a integer, all centers share the same gamma. If a tensor, must have shape (n_centers,) to specify a separate gamma for each center.
        """
        if isinstance(gamma, float) or isinstance(gamma, int):
            gamma_tensor = torch.full(
                (self.n_centers,),
                float(gamma),
                device=self.gamma.device,
                dtype=self.gamma.dtype
            )
        else:
            gamma_tensor = gamma
            if gamma_tensor.shape != (self.n_centers,):
                raiseError(f"gamma tensor must have shape "f"({self.n_centers},)")

        self.gamma.copy_(gamma_tensor)

    def forward(self, X):
        # X shape: (B, D)
        # C shape: (L, D)
        # Avoid computing ||x - c||^2 with shape (B, L, D) by using the expansion:
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x@c^T

        x_sq = (X ** 2).sum(dim=-1, keepdim=True)                   # (B, 1)
        c_sq = (self.centers ** 2).sum(dim=-1, keepdim=True).T      # (1, L)
        dist_sq = x_sq + c_sq - 2.0 * (X @ self.centers.T)          # (B, L)
        return torch.exp(-dist_sq * self.gamma.unsqueeze(0))        # (B, L)


class RBFELM(nn.Module):
    r"""
    Radial Basis Function Extreme Learning Machine (RBF-ELM) for regression.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        n_centers (int): Number of RBF neurons in the hidden layer.
        gamma (float): Width parameter of the Gaussian RBF kernel. Ignored if ``gamma_mode="local"``.
        reg_lambda (float, optional): Tikhonov regularisation coefficient (default: ``1e-8``).
        center_sampling (str, optional): Method to sample RBF centers from the training data (default: ``"random"``).
            - ``"random"``: Uniform random sampling of rows from the training set.
            - ``"uniform"``: Voxel-grid subsampling that divides the bounding box into a 3-D grid and picks one random point per occupied cell, then adjusts to return exactly ``n_centers`` points.
        gamma_mode (str, optional): Method to determine gamma values (default: ``"fixed"``).
            - ``"fixed"``: All centers share the same gamma value specified by the ``gamma`` parameter.
            - ``"local"``: Each center has its own gamma value estimated from the local spacing of centers.  The ``gamma_k`` and ``gamma_alpha`` parameters control the estimation.
        gamma_k (int, optional): Number of nearest neighbours used to estimate local spacing for the ``"local"`` gamma mode (default: ``10``).
        gamma_alpha (float, optional): Scaling factor for local gamma estimation (default: ``1.0``).  Values < 1 produce narrower kernels; > 1 produce wider kernels.
        device (torch.device, optional): Computation device (default: ``torch.device("cpu")``).
        seed (int, optional): Seed for reproducible center sampling (default: ``None``).
        model_name (str, optional): Base name for checkpoints (default: ``"rbfelm"``).
        verbose (bool, optional): Print hyperparameters on construction (default: ``True``).
        kwargs: Ignored; kept for API compatibility.
    """
    def __init__(
        self,
        input_size:         int,
        output_size:        int,
        n_centers:          int,
        reg_lambda:         float = 1e-8,
        center_sampling:    str = "random",
        gamma_mode:         str = "fixed",
        gamma:              float = 1.0,
        gamma_k:            int = 10,
        gamma_alpha:        float = 1.0,
        device:             torch.device = DEVICE,
        seed:               Optional[int] = None,
        model_name:         str = "rbfelm",
        verbose:            bool = True,
        **kwargs,
    ):
        super().__init__()

        self.input_size         = input_size
        self.output_size        = output_size
        self.n_centers          = n_centers
        self.gamma              = gamma
        self.reg_lambda         = reg_lambda
        self.center_sampling    = center_sampling
        self.gamma_mode         = gamma_mode
        self.gamma_k            = gamma_k
        self.gamma_alpha        = gamma_alpha
        self.device             = device
        self.seed               = seed
        self.model_name         = model_name

        self.hidden = RBFLayer(n_centers, input_size)
        self.register_buffer("beta", None)

        if seed is not None:
            set_seed(seed)

        self.to(self.device)

        if verbose:
            pprint(0, f"Creating model: {self._model_name}")
            keys_print = [
                "input_size",
                "output_size",
                "n_centers",
                "gamma",
                "reg_lambda",
                "center_sampling",
                "gamma_mode",
                "gamma_k",
                "gamma_alpha",
                "device",
                "seed",
                "model_name",
            ]
            for key in keys_print:
                pprint(0, f"\t{key}: {getattr(self, key)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted():
            raise raiseError("Model has not been fitted yet. Call fit() first.")
        
        return self.hidden(x) @ self.beta

    def _is_fitted(self) -> bool:
        return self.beta is not None
    
    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("model_name must be a string")
        value = value.strip()
        if not value:
            raise ValueError("model_name cannot be empty")
        self._model_name = value

    @staticmethod
    def _sample_centers(
        all_x: torch.Tensor,
        n_centers: int,
        method: str,
        gen: torch.Generator,
        chunk: int = 10_000,
    ) -> torch.Tensor:
        """
        Select ``n_centers`` rows from ``all_x`` as RBF centers.

        Args:
            all_x (torch.Tensor): All training inputs, shape ``(N, D)``.
            n_centers (int): Number of centers to select.
            method (str):
                ``"random"``  — uniform random over indices (original behaviour).
                ``"uniform"`` — voxel-grid subsampling: divides the bounding box
                    into a 3-D grid and picks one random point per occupied cell,
                    then adjusts to return exactly ``n_centers`` points. Works
                    directly on the surface geometry without requiring mesh
                    connectivity or cell areas.
            gen (torch.Generator): RNG state (for reproducibility).
            chunk (int): Unused, kept for API consistency.

        Returns:
            torch.Tensor: Selected center rows, shape ``(n_centers, D)``.
        """
        N = all_x.shape[0]

        if method == "random":
            idx = torch.randperm(N, generator=gen)[:n_centers]
            return all_x[idx].detach().clone()

        elif method == "uniform":
            # Work entirely on CPU to avoid device mismatch issues;
            # caller moves the result to the target device.
            x_cpu = all_x.cpu()
            N     = x_cpu.shape[0]
            K     = math.ceil(n_centers ** (1 / 3)) + 1

            mins  = x_cpu.min(dim=0).values
            maxs  = x_cpu.max(dim=0).values
            span  = maxs - mins
            span  = torch.where(span > 0, span, torch.ones_like(span))

            coords    = ((x_cpu - mins) / span * K).long().clamp(0, K - 1)
            voxel_ids = coords[:, 0] * K * K + coords[:, 1] * K + coords[:, 2]

            order      = torch.randperm(N, generator=gen)   # CPU generator → CPU tensor
            voxel_ids  = voxel_ids[order]

            sorted_v, sort_idx = voxel_ids.sort()
            first_occ = torch.cat([
                torch.tensor([True]),                        # CPU, matches sorted_v
                sorted_v[1:] != sorted_v[:-1]
            ])
            cell_representatives = order[sort_idx[first_occ]]

            n_occupied = cell_representatives.shape[0]

            if n_occupied >= n_centers:
                chosen = cell_representatives[
                    torch.randperm(n_occupied, generator=gen)[:n_centers]
                ]
            else:
                missing = n_centers - n_occupied
                mask    = torch.ones(N, dtype=torch.bool)
                mask[cell_representatives] = False
                extra   = torch.where(mask)[0]
                extra   = extra[torch.randperm(extra.shape[0], generator=gen)[:missing]]
                chosen  = torch.cat([cell_representatives, extra])

            return x_cpu[chosen].detach().clone()

        else:
            raise ValueError(
                f"Unknown center_sampling method: '{method}'. "
                "Choose 'random' or 'uniform'."
            )
    
    @staticmethod
    def _estimate_local_gamma(
        centers: torch.Tensor,
        k: int = 10,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Estimate a per-center gamma value based on the local spacing between centers.

        For each center, the mean distance to its ``k`` nearest neighbours is used
        as a local length scale ``sigma``.  The resulting gamma is:

        .. math::

            \\gamma_l = \\frac{1}{(\\alpha \\cdot \\sigma_l)^2}

        Args:
            centers (torch.Tensor): RBF centers, shape ``(L, D)``.
            k (int): Number of nearest neighbours used to estimate local spacing
                (default: ``10``).
            alpha (float): Scaling factor for the bandwidth (default: ``1.0``).
                Values < 1 produce narrower kernels; > 1 produce wider kernels.

        Returns:
            torch.Tensor: Per-center gamma values, shape ``(L,)``, on the same
            device and dtype as ``centers``.
        """
        centers_np    = centers.cpu().numpy()
        nbrs          = NearestNeighbors(n_neighbors=k + 1).fit(centers_np)
        distances, _  = nbrs.kneighbors(centers_np)
        local_spacing = distances[:, 1:].mean(axis=1)
        sigma         = local_spacing
        sigma         = np.where(sigma > 0, sigma, np.finfo(np.float32).eps)
        gamma         = alpha / (sigma ** 2)
        return torch.tensor(gamma, device=centers.device, dtype=centers.dtype)
        
    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        batch_size: int = 10_000,
        save_logs_path: Optional[str] = None,
        print_rate_batch: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        r"""
        Fit the RBF-ELM using block-wise accumulation of the normal equations.

        Steps
        -----
        1. Randomly sample ``n_centers`` rows from the training set as RBF centers.
        2. Stream data in blocks to accumulate :math:`H^T H` and :math:`H^T Y`
           without loading everything into memory at once.
        3. Solve the ``(L × L)`` regularised system for ``beta``.

        Args:
            train_dataset (torch.utils.data.Dataset): Training data; items must
                yield ``(x, y)`` pairs.
            batch_size (int, optional): Block size for accumulation
                (default: ``10 000``).
            save_logs_path (str, optional): Directory to save the checkpoint
                after fitting (default: ``None``).
            print_rate_batch (int, optional): Print progress every N blocks;
                ``0`` to suppress (default: ``1``).
            kwargs: Ignored (API compatibility).
        """
        dtype = torch.float32

        # ---- Pass 1: collect all X to sample centers -----------------
        loader0 = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        all_x = torch.cat(
            [b[0] for b in loader0], dim=0
        ).to(self.device, dtype=dtype)

        N = all_x.shape[0]
        gen = torch.Generator(device="cpu")
        if self.seed is not None:
            gen.manual_seed(self.seed)

        # Assign centers into the RBFLayer buffer
        selected_centers = self._sample_centers(
            all_x, self.n_centers, self.center_sampling, gen
        ).to(self.device)

        self.hidden.centers = selected_centers

        if self.gamma_mode == "local":
            local_gamma = self._estimate_local_gamma(centers=selected_centers, k=self.gamma_k, alpha=self.gamma_alpha)
            self.hidden.set_gamma(local_gamma)
        else:
            self.hidden.set_gamma(self.gamma)

        del all_x

        # ---- Pass 2: block-wise accumulation of H^T H, H^T Y --------
        L   = self.n_centers
        HtH = torch.zeros((L, L),                device=self.device, dtype=dtype)
        Hty = torch.zeros((L, self.output_size),  device=self.device, dtype=dtype)

        loader1       = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        total_blocks  = math.ceil(N / batch_size)

        for b_idx, batch in enumerate(loader1):
            x_blk, y_blk = batch[0].to(self.device, dtype=dtype), batch[1].to(self.device, dtype=dtype)
            if y_blk.dim() == 1:
                y_blk = y_blk.unsqueeze(-1)

            H_blk  = self.hidden(x_blk)   # (B, L)
            HtH   += H_blk.T @ H_blk
            Hty   += H_blk.T @ y_blk

            # Cuántos elementos de H_blk son no-cero
            threshold = 1e-6
            nnz_H = (H_blk.abs() > threshold).sum().item()
            sparsity_H = 1 - nnz_H / (H_blk.numel())

            # Calcular H^T H y ver su sparsidad
            HtH_blk = H_blk.T @ H_blk
            nnz_HtH = (HtH_blk.abs() > threshold).sum().item()
            sparsity_HtH = 1 - nnz_HtH / (HtH_blk.numel())

            print(f"\nH sparsity:     {sparsity_H:.3%}")
            print(f"H^T H sparsity: {sparsity_HtH:.3%}")


            if verbose:
                if print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                    print(
                        f"\r\tFitting block {b_idx + 1}/{total_blocks}",
                        end="", flush=True,
                    )

        # ---- Solve normal equations ----------------------------------
        I         = torch.eye(L, device=self.device, dtype=dtype)
        self.beta = torch.linalg.solve(HtH + self.reg_lambda * I, Hty)
        if verbose:
            print(f"\tDone.  beta shape: {tuple(self.beta.shape)}")

        if save_logs_path is not None:
            self.save(os.path.join(save_logs_path, f"{self._model_name}.pth"))

    def fit_cholmod(
        self,
        train_dataset: torch.utils.data.Dataset,
        batch_size: int = 10_000,
        sparsity_threshold: float = 1e-6,
        save_logs_path: Optional[str] = None,
        print_rate_batch: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        r"""
        Fit the RBF-ELM by accumulating :math:`H^T H` as a sparse matrix and
        solving the normal equations with a sparse Cholesky factorisation
        (CHOLMOD via ``scikit-sparse``).
 
        This is the recommended method when ``n_centers`` is large (e.g.
        :math:`L = 200\,000`) and the RBF kernels are narrow enough that both
        :math:`H` and :math:`H^T H` are sparse.  Memory usage is
        :math:`O(\text{nnz}(H^T H))` rather than :math:`O(L^2)`, making it
        feasible where :meth:`fit` runs out of memory.
 
        Steps
        -----
        1. Sample centers and estimate gamma (identical to :meth:`fit`).
        2. Stream data in blocks.  For each block:
 
           a. Evaluate :math:`H_\text{blk}` on the GPU.
           b. Zero out entries below ``sparsity_threshold``.
           c. Compute :math:`H_\text{blk}^T H_\text{blk}` and sparsify it.
           d. Accumulate into a running ``scipy`` sparse matrix (CPU).
           e. Accumulate :math:`H_\text{blk}^T y_\text{blk}` (dense, small).
 
        3. Add regularisation: :math:`A = H^T H + \lambda I`.
        4. Factorise :math:`A` with CHOLMOD and solve for each output column.
 
        Args:
            train_dataset (torch.utils.data.Dataset): Training data; items
                must yield ``(x, y)`` pairs.
            batch_size (int, optional): Block size for accumulation
                (default: ``10 000``).  Larger values speed up the GPU work
                but use more intermediate memory.
            sparsity_threshold (float, optional): Entries of :math:`H_\text{blk}`
                and :math:`H_\text{blk}^T H_\text{blk}` whose absolute value
                falls below this threshold are treated as zero
                (default: ``1e-6``).
            save_logs_path (str, optional): Directory to save a checkpoint
                after fitting (default: ``None``).
            print_rate_batch (int, optional): Print progress every N blocks;
                ``0`` to suppress (default: ``1``).
            verbose (bool, optional): Print summary statistics
                (default: ``True``).
            kwargs: Ignored (API compatibility).
 
        Notes
        -----
        * Requires ``scikit-sparse`` (``pip install scikit-sparse`` or
          ``conda install -c conda-forge scikit-sparse``).
        * :math:`H_\text{blk}` is evaluated on ``self.device`` (GPU) and
          immediately moved to CPU before sparsification, so GPU memory usage
          per block is :math:`O(B \times L)` in float32.
        * The accumulated :math:`H^T H` lives in CPU RAM as a
          ``scipy.sparse.csc_matrix``.  For :math:`L = 200\,000` with ~5%
          fill, this is roughly 8 GB.
        * CHOLMOD exploits the sparsity of :math:`H^T H` for both the
          symbolic and numeric factorisation, making the solve much faster
          than a dense :math:`O(L^3)` factorisation.
        """
        try:
            import scipy.sparse as sp
            from sksparse.cholmod import cholesky as cholmod_cholesky
        except ImportError:
            raise ImportError(
                "scikit-sparse is required for fit_cholmod.\n"
                "Install with:  conda install -c conda-forge scikit-sparse\n"
                "           or: pip install scikit-sparse"
            )
 
        dtype  = torch.float32
        device = self.device
 
        # ----------------------------------------------------------------
        # Pass 1 — collect all X, sample centers, estimate gamma
        # ----------------------------------------------------------------
        loader0 = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        all_x = torch.cat(
            [b[0] for b in loader0], dim=0
        ).to(device, dtype=dtype)
 
        N = all_x.shape[0]
        L = self.n_centers
 
        gen = torch.Generator(device="cpu")
        if self.seed is not None:
            gen.manual_seed(self.seed)
 
        selected_centers = self._sample_centers(
            all_x, L, self.center_sampling, gen
        ).to(device)
        self.hidden.centers = selected_centers
 
        if self.gamma_mode == "local":
            local_gamma = self._estimate_local_gamma(
                centers=selected_centers, k=self.gamma_k, alpha=self.gamma_alpha
            )
            self.hidden.set_gamma(local_gamma)
        else:
            self.hidden.set_gamma(self.gamma)
 
        del all_x
 
        # ----------------------------------------------------------------
        # Pass 2 — accumulate H^T H (sparse) and H^T y (dense) by blocks
        # ----------------------------------------------------------------
        loader1      = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        total_blocks = math.ceil(N / batch_size)
 
        # Running sparse accumulator for H^T H  (COO triplets, flushed per block)
        HtH_sparse = sp.csc_matrix((L, L), dtype=np.float32)
        Hty        = np.zeros((L, self.output_size), dtype=np.float32)
 
        total_nnz_HtH = 0
 
        if verbose:
            print(f"\t[fit_cholmod] Accumulating H^T H  (N={N}, L={L}, "
                  f"threshold={sparsity_threshold})")
 
        for b_idx, batch in enumerate(loader1):
            x_blk, y_blk = batch[0].to(self.device, dtype=dtype), batch[1].to(self.device, dtype=dtype)
            if y_blk.dim() == 1:
                y_blk = y_blk.unsqueeze(-1)
 
            # ---- Evaluate H_blk on GPU, move to CPU --------------------
            with torch.no_grad():
                H_blk = self.hidden(x_blk).cpu()           # (B, L) float32
 
            # ---- Sparsify H_blk ----------------------------------------
            mask_H        = H_blk.abs() >= sparsity_threshold
            H_blk_sparse  = H_blk * mask_H                 # zero out small entries
 
            # ---- H^T H block (dense on CPU, then sparsify) -------------
            HtH_blk = H_blk_sparse.T @ H_blk_sparse        # (L, L)
            mask_HtH = HtH_blk.abs() >= sparsity_threshold
            nz_rows, nz_cols = mask_HtH.nonzero(as_tuple=True)
            nz_vals          = HtH_blk[nz_rows, nz_cols].numpy().astype(np.float32)
 
            # Accumulate into sparse matrix (convert to scipy COO then CSC)
            block_sparse = sp.csc_matrix(
                (nz_vals, (nz_rows.numpy(), nz_cols.numpy())),
                shape=(L, L),
                dtype=np.float32,
            )
            HtH_sparse = HtH_sparse + block_sparse
            total_nnz_HtH += len(nz_vals)
 
            # ---- H^T y block (dense, cheap) ----------------------------
            Hty += (H_blk_sparse.T @ y_blk.cpu()).numpy()
 
            if verbose and print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                sparsity_H   = 1.0 - mask_H.sum().item()   / H_blk.numel()
                sparsity_HtH = 1.0 - mask_HtH.sum().item() / (L * L)
                print(
                    f"\r\tBlock {b_idx + 1}/{total_blocks}  "
                    f"H sparsity={sparsity_H:.2%}  "
                    f"H^T H sparsity={sparsity_HtH:.2%}  "
                    f"nnz(H^T H accumulated)={HtH_sparse.nnz:,}",
                    end="", flush=True,
                )
 
        if verbose:
            final_sparsity = 1.0 - HtH_sparse.nnz / (L * L)
            mem_mb = HtH_sparse.nnz * 4 / 1024 ** 2
            print(f"\n\t[fit_cholmod] H^T H assembled — "
                  f"nnz={HtH_sparse.nnz:,}  "
                  f"sparsity={final_sparsity:.3%}  "
                  f"mem≈{mem_mb:.1f} MB")
 
        # ----------------------------------------------------------------
        # Add regularisation:  A = H^T H + lambda * I
        # ----------------------------------------------------------------
        A = HtH_sparse + self.reg_lambda * sp.eye(L, format="csc", dtype=np.float32)
        A = sp.csc_matrix(A)
 
        del HtH_sparse
 
        # ----------------------------------------------------------------
        # Cholesky factorisation (done once, reused for all output dims)
        # ----------------------------------------------------------------
        if verbose:
            print(f"\t[fit_cholmod] Factorising with CHOLMOD...")
 
        factor = cholmod_cholesky(A)
 
        del A
 
        # ----------------------------------------------------------------
        # Solve for each output dimension
        # ----------------------------------------------------------------
        if verbose:
            print(f"\t[fit_cholmod] Solving for {self.output_size} output dim(s)...")
 
        beta_np = np.empty((L, self.output_size), dtype=np.float32)
        for o in range(self.output_size):
            beta_np[:, o] = factor(Hty[:, o])
 
        self.beta = torch.tensor(beta_np, dtype=dtype, device=device)
 
        if verbose:
            print(f"\t[fit_cholmod] Done.  beta shape: {tuple(self.beta.shape)}")
 
        if save_logs_path is not None:
            self.save(os.path.join(save_logs_path, f"{self._model_name}.pth"))

    @cr('RBFELM.predict')
    def predict(
        self,
        X: torch.utils.data.Dataset,
        return_targets: bool = False,
        dataloader_kwargs: dict = {},
        **kwargs,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        r"""
        Predict target values for the input data of a dataset. The dataset is loaded to a DataLoader with the provided keyword arguments. 
        The model is set to evaluation mode and the predictions are made using the input data. 
        To make a prediction from a torch tensor, use the `__call__` method directly.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            return_targets (bool, optional): If ``True``, the true target values will be returned along with the predictions (default: ``False``).
                        dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size=16_384`` ,``shuffle=False``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).

        Returns:
            ``np.ndarray`` of shape ``(N, output_size)``, or a ``(predictions, targets)`` tuple if ``return_targets=True``.
        """
        _dataloader_kwargs = {
            "batch_size": kwargs.get("batch_size", 16_384),
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }

        predict_dataloader = DataLoader(X, **_dataloader_kwargs)
        total_rows = len(predict_dataloader.dataset)
        total_cols = self.output_size
        all_predictions = np.empty((total_rows, total_cols), dtype=np.float32)
        all_targets = np.empty((total_rows, total_cols), dtype=np.float32)

        self.eval()
        start_idx = 0
        with torch.no_grad():
            for x, y in predict_dataloader:
                output = self(x.to(self.device))
                batch_size = x.size(0)
                end_idx = start_idx + batch_size
                all_predictions[start_idx:end_idx, :] = output.cpu().numpy()
                if return_targets:
                    all_targets[start_idx:end_idx, :] = y.cpu().numpy()
                start_idx = end_idx

        return (all_predictions, all_targets) if return_targets else all_predictions

    def _define_checkpoint(self) -> Dict:
        return {
            "input_size":       self.input_size,
            "output_size":      self.output_size,
            "n_centers":        self.n_centers,
            "gamma":            self.gamma,
            "reg_lambda":       self.reg_lambda,
            "center_sampling":  self.center_sampling,
            "gamma_mode":       self.gamma_mode,
            "gamma_k":          self.gamma_k,
            "gamma_alpha":      self.gamma_alpha,
            "device":           self.device,
            "seed":             self.seed,
            "model_name":       self._model_name,
            "centers":          self.hidden.centers,
            "gamma_tensor":     self.hidden.gamma,
            "beta":             self.beta,
        }

    def save(self, path: str) -> None:
        r"""
        Save the model to a ``.pth`` checkpoint.

        Args:
            path (str): File path or directory.  A directory receives the
                automatic filename ``{model_name}.pth``.
        """
        if os.path.isdir(path):
            path = os.path.join(path, f"{self._model_name}.pth")
        torch.save(self._define_checkpoint(), path)
        print(f"\tModel saved → {path}")

    @classmethod
    def load(cls, path, device=torch.device("cpu"), verbose=True):
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_size      = ckpt["input_size"],
            output_size     = ckpt["output_size"],
            n_centers       = ckpt["n_centers"],
            gamma           = ckpt["gamma"],
            reg_lambda      = ckpt["reg_lambda"],
            center_sampling = ckpt.get("center_sampling", "random"),
            gamma_mode      = ckpt.get("gamma_mode", "fixed"),
            gamma_k         = ckpt.get("gamma_k", 10),
            gamma_alpha     = ckpt.get("gamma_alpha", 1.0),
            device          = device,
            seed            = ckpt["seed"],
            model_name      = ckpt["model_name"],
            verbose         = verbose,
        )
        if ckpt["centers"] is not None:
            model.hidden.centers = ckpt["centers"].to(device)
        if ckpt.get("gamma_tensor") is not None:
            model.hidden.set_gamma(ckpt["gamma_tensor"].to(device))
        if ckpt["beta"] is not None:
            model.beta = ckpt["beta"].to(device)
        return model

    # ------------------------------------------------------------------
    # create_optimized_model
    # ------------------------------------------------------------------

    @classmethod
    def create_optimized_model(
        cls,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset:  torch.utils.data.Dataset,
        optuna_optimizer,
        **kwargs,
    ) -> Tuple["RBFELM", Dict]:
        r"""
        Hyperparameter search with Optuna; returns the best unfitted model.

        Each trial instantiates a fresh ``RBFELM``, calls :meth:`fit`, and
        scores it on ``eval_dataset`` using MSE.  After all trials the best
        hyperparameters are used to build the returned model.

        Searchable hyperparameters
        --------------------------
        Supply ``(low, high)`` tuples for parameters you want Optuna to search,
        or a fixed scalar to hold a parameter constant:

        ================  =======  =========================================
        Key               Type     Description
        ================  =======  =========================================
        ``n_centers``     int      Number of hidden RBF neurons
        ``gamma``         float    Gaussian kernel width
        ``reg_lambda``    float    Tikhonov regularisation coefficient
        ``batch_size``    int      Block size used during fitting
        ================  =======  =========================================

        Ranges spanning more than 2 orders of magnitude are sampled on a
        **log scale** automatically.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Validation dataset for scoring each trial.
            optuna_optimizer (OptunaOptimizer): Must expose
                ``.optimization_params`` (dict) and
                ``.optimize(objective_function)`` → best_params dict.
            kwargs: Ignored.

        Returns:
            Tuple[RBFELM, Dict]: Best unfitted model and resolved parameter
            dict.  Fit the model with:

            .. code-block:: python

                model.fit(train_dataset, **best_params)

        Example
        -------
        >>> optimization_params = {
        ...     "n_centers":  (500, 5000),      # searched (log-scale int)
        ...     "gamma":      (1e-3, 10.0),     # searched (log-scale float)
        ...     "reg_lambda": (1e-10, 1e-4),    # searched (log-scale float)
        ...     "batch_size": 10_000,           # fixed
        ... }
        >>> optimizer = OptunaOptimizer(
        ...     optimization_params=optimization_params,
        ...     n_trials=30,
        ...     direction="minimize",
        ... )
        >>> model, best_params = RBFELM.create_optimized_model(
        ...     train_dataset, eval_dataset, optimizer
        ... )
        >>> model.fit(train_dataset, **best_params)
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for create_optimized_model. "
                "Install it with:  pip install optuna"
            )

        optimization_params = optuna_optimizer.optimization_params
        input_dim  = train_dataset[0][0].shape[0]
        first_y    = train_dataset[0][1]
        output_dim = first_y.shape[0] if first_y.dim() > 0 else 1

        # ---- Helper: suggest or pass through a single value -----------
        def suggest_value(name, space, trial):
            if isinstance(space, (tuple, list)) and len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    use_log = (high / max(1, low)) >= 100
                    return trial.suggest_int(name, low, high, log=use_log)
                if isinstance(low, float) and isinstance(high, float):
                    use_log = (high / max(1e-12, low)) >= 100
                    return trial.suggest_float(name, low, high, log=use_log)
            return space  # fixed scalar

        # ---- Objective ------------------------------------------------
        def objective(trial) -> float:
            model = None
            try:
                training_params: Dict = {
                    key: suggest_value(key, space, trial)
                    for key, space in optimization_params.items()
                }
                model = cls(
                    input_size  = input_dim,
                    output_size = output_dim,
                    verbose     = False,
                    **training_params,
                )
                model.fit(train_dataset, print_rate_batch=0, verbose=False, **training_params)
                y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                return float(((y_pred - y_true) ** 2).mean())

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"Trial {trial.number}: OOM — pruning.")
                    raise TrialPruned()
                raise

            finally:
                del model

        # ---- Run optimisation ----------------------------------------
        best_params = optuna_optimizer.optimize(objective_function=objective)

        # Resolve final parameter dict (best values override search spaces)
        resolved: Dict = {}
        for key, space in optimization_params.items():
            if key in best_params:
                resolved[key] = best_params[key]
            elif not isinstance(space, (tuple, list)):
                resolved[key] = space   # was fixed all along

        final_model = cls(
            input_size  = input_dim,
            output_size = output_dim,
            **resolved,
        )
        return final_model, resolved


class MultiRBFELM:
    r"""
    Ensemble of local RBF-ELM models trained on spatial clusters.

    For large datasets where a single RBF-ELM would require too many centers
    to capture local structure, ``MultiRBFELM`` partitions the input space
    using :class:`~sklearn.cluster.MiniBatchKMeans`, trains one
    :class:`RBFELM` per cluster, and at inference routes each point to its
    nearest-cluster model.

    The public interface mirrors :class:`RBFELM` (``fit``, ``predict``,
    ``save``, ``load``) so both classes can be used interchangeably.

    Persistence layout (``save`` / ``load``)
    -----------------------------------------
    All artefacts are stored under a single directory::

        <path>/
          meta.json          ← n_clusters + shared RBFELM hyperparameters
          kmeans.pkl         ← fitted MiniBatchKMeans router
          scalers.pkl        ← per-cluster (input_scaler, output_scaler) pairs
          model_0.pth
          model_1.pth
          ...

    Args:
        n_clusters (int): Number of spatial clusters (and local models).
        n_centers (int): RBF centers per local model.  Capped at the cluster
            size automatically.
        reg_lambda (float): Tikhonov regularisation for each local model
            (default: ``1e-2``).
        center_sampling (str): Center-sampling strategy passed to each
            :class:`RBFELM` (default: ``"uniform"``).
        gamma_mode (str): Gamma estimation mode (default: ``"local"``).
        gamma_k (int): Neighbours used for local gamma estimation
            (default: ``30``).
        gamma_alpha (float): Bandwidth scaling factor (default: ``1.0``).
        kmeans_batch_size (int): ``batch_size`` for
            :class:`~sklearn.cluster.MiniBatchKMeans` (default: ``100_000``).
        kmeans_n_init (int): ``n_init`` for MiniBatchKMeans (default: ``5``).
        fit_batch_size (int): DataLoader block size used during each local
            :class:`RBFELM` fit (default: ``100_000``).
        device (torch.device): Computation device (default: global ``DEVICE``).
        seed (int, optional): Master seed for reproducibility.
        model_name (str): Base name used when printing / saving
            (default: ``"multirbfelm"``).
        verbose (bool): Print progress during fit (default: ``True``).
    """

    def __init__(
        self,
        n_clusters: int,
        n_centers: int = 10_000,
        reg_lambda: float = 1e-2,
        center_sampling: str = "uniform",
        gamma_mode: str = "local",
        gamma_k: int = 30,
        gamma_alpha: float = 1.0,
        kmeans_batch_size: int = 100_000,
        kmeans_n_init: int = 5,
        fit_batch_size: int = 100_000,
        device: torch.device = DEVICE,
        seed: Optional[int] = None,
        model_name: str = "multirbfelm",
        verbose: bool = True,
    ):
        self.n_clusters       = n_clusters
        self.n_centers        = n_centers
        self.reg_lambda       = reg_lambda
        self.center_sampling  = center_sampling
        self.gamma_mode       = gamma_mode
        self.gamma_k          = gamma_k
        self.gamma_alpha      = gamma_alpha
        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_n_init    = kmeans_n_init
        self.fit_batch_size   = fit_batch_size
        self.device           = device
        self.seed             = seed
        self.model_name       = model_name
        self.verbose          = verbose

        # Populated by fit() / load()
        self._kmeans:         Optional[MiniBatchKMeans] = None
        self._models:         List[Optional[RBFELM]]   = [None] * n_clusters
        self._input_scalers:  List[Optional[object]]   = [None] * n_clusters
        self._output_scalers: List[Optional[object]]   = [None] * n_clusters

        if verbose:
            print(f"MultiRBFELM — {n_clusters} clusters, up to {n_centers} centers each")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _is_fitted(self) -> bool:
        return self._kmeans is not None and all(m is not None for m in self._models)

    def _assign_clusters(self, X: torch.Tensor) -> np.ndarray:
        """Return cluster label for every row in X (shape N,)."""
        return self._kmeans.predict(X.cpu().numpy())

    def _build_kmeans(self) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters   = self.n_clusters,
            random_state = self.seed,
            batch_size   = self.kmeans_batch_size,
            n_init       = self.kmeans_n_init,
        )

    @staticmethod
    def _subset_dataset(
        dataset,
        mask: np.ndarray,
        input_scaler,
        output_scaler,
    ):
        """
        Build an NNDataset for the rows selected by boolean *mask*, using
        already-fitted scalers so train/valid/test share the same scaling.
        """
        if NNDataset is None or RobustScaler is None:
            raise ImportError(
                "NNDataset and RobustScaler must be importable to use MultiRBFELM.fit()."
            )

        indices  = torch.where(torch.tensor(mask))[0]
        X_full, y_full = dataset[:]
        X_sub = X_full[indices]
        y_sub = y_full[indices]

        return NNDataset(
            variables_out  = (y_sub,),
            variables_in   = X_sub,
            parameters     = None,
            inputs_scaler  = input_scaler,
            outputs_scaler = output_scaler,
        ), indices

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset,
        valid_dataset,
        save_dir: Optional[str] = None,
        reload_existing: bool = True,
    ) -> None:
        r"""
        Cluster the training set and fit one :class:`RBFELM` per cluster.

        The KMeans model is fitted **only on training data**; validation and
        test points are then assigned to the nearest cluster centroid.

        Args:
            train_dataset: Training :class:`NNDataset`.
            valid_dataset: Validation :class:`NNDataset` (used for progress
                reporting only; not used for fitting).
            save_dir (str, optional): If given, each local model is saved to
                ``<save_dir>/model_<i>.pth`` as it is trained, and reloaded
                from disk on subsequent calls when ``reload_existing=True``.
            reload_existing (bool): Skip retraining a cluster whose ``.pth``
                file already exists in ``save_dir`` (default: ``True``).
        """
        if NNDataset is None or RobustScaler is None:
            raise ImportError(
                "NNDataset and RobustScaler are required for MultiRBFELM.fit()."
            )

        X_train, y_train = train_dataset[:]
        X_valid, y_valid = valid_dataset[:]

        input_size  = X_train.shape[1]
        output_size = y_train.shape[1] if y_train.dim() > 1 else 1

        # ----------------------------------------------------------------
        # 1. KMeans clustering on training set
        # ----------------------------------------------------------------
        self._log("\n[MultiRBFELM] Fitting KMeans on training set...")
        self._kmeans = self._build_kmeans()
        train_labels = self._kmeans.fit_predict(X_train.cpu().numpy())
        valid_labels = self._assign_clusters(X_valid)

        self._log(f"[MultiRBFELM] Cluster sizes (train): "
                  f"{np.bincount(train_labels, minlength=self.n_clusters).tolist()}")

        # ----------------------------------------------------------------
        # 2. Train one RBFELM per cluster
        # ----------------------------------------------------------------
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for i in range(self.n_clusters):
            self._log(f"\n[MultiRBFELM] ── Cluster {i + 1}/{self.n_clusters} ──")

            model_path = (
                os.path.join(save_dir, f"model_{i}.pth") if save_dir else None
            )

            # Optionally reload an already-trained model
            if reload_existing and model_path and os.path.exists(model_path):
                self._log(f"  Loading existing model from {model_path}")
                self._models[i] = RBFELM.load(model_path, device=self.device, verbose=False)
                # Scalers must have been saved separately — loaded in load()
                continue

            # ---- Build per-cluster scalers (fit on train split) --------
            train_mask = train_labels == i
            valid_mask = valid_labels == i

            n_train_i = int(train_mask.sum())
            n_valid_i = int(valid_mask.sum())
            self._log(f"  Train: {n_train_i}  |  Valid: {n_valid_i}")

            if n_train_i == 0:
                self._log("  ⚠ Empty cluster — skipping.")
                continue

            input_scaler_i  = RobustScaler()
            output_scaler_i = RobustScaler()

            train_sub, _ = self._subset_dataset(
                train_dataset, train_mask, input_scaler_i, output_scaler_i
            )
            valid_sub, _ = self._subset_dataset(
                valid_dataset, valid_mask, input_scaler_i, output_scaler_i
            )

            self._input_scalers[i]  = input_scaler_i
            self._output_scalers[i] = output_scaler_i

            # ---- Instantiate and fit local RBFELM ----------------------
            n_centers_i = min(self.n_centers, n_train_i)

            model = RBFELM(
                input_size      = input_size,
                output_size     = output_size,
                n_centers       = n_centers_i,
                reg_lambda      = self.reg_lambda,
                center_sampling = self.center_sampling,
                gamma_mode      = self.gamma_mode,
                gamma_k         = self.gamma_k,
                gamma_alpha     = self.gamma_alpha,
                device          = self.device,
                seed            = self.seed,
                model_name      = f"{self.model_name}_cluster_{i}",
                verbose         = self.verbose,
            )

            model.fit(
                train_dataset  = train_sub,
                batch_size     = self.fit_batch_size,
                save_logs_path = None,
            )

            if model_path:
                model.save(model_path)

            self._models[i] = model

        self._log("\n[MultiRBFELM] All clusters trained.")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        dataset,
        return_targets: bool = False,
        batch_size: int = 50_000,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        r"""
        Predict target values by routing each point to its local model.

        Args:
            dataset: :class:`NNDataset` whose inputs are used for routing and
                prediction.
            return_targets (bool): Also return true targets (default: ``False``).
            batch_size (int): Unused; kept for API parity with
                :class:`RBFELM`.

        Returns:
            ``np.ndarray`` of shape ``(N, output_size)``, or a
            ``(predictions, targets)`` tuple if ``return_targets=True``.
        """
        if not self._is_fitted():
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        X_all, y_all = dataset[:]
        N = X_all.shape[0]

        output_size = (
            y_all.shape[1] if y_all.dim() > 1 else 1
        )

        labels    = self._assign_clusters(X_all)
        preds_out = np.empty((N, output_size), dtype=np.float32)
        tgts_out  = np.empty((N, output_size), dtype=np.float32)

        for i in range(self.n_clusters):
            mask    = labels == i
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            model           = self._models[i]
            output_scaler_i = self._output_scalers[i]

            if model is None:
                raise RuntimeError(f"Model for cluster {i} is not fitted.")

            # Build a minimal NNDataset using the cluster's scalers
            X_sub = X_all[indices]
            y_sub = y_all[indices]

            sub_dataset, _ = self._subset_dataset(
                _TensorDatasetWrapper(X_all, y_all),
                mask,
                self._input_scalers[i],
                self._output_scalers[i],
            )

            y_pred_scaled, y_tgt_scaled = model.predict(sub_dataset, return_targets=True)

            # Inverse-transform back to original scale
            y_pred = output_scaler_i.inverse_transform(y_pred_scaled)
            y_tgt  = output_scaler_i.inverse_transform(y_tgt_scaled)

            preds_out[indices] = y_pred
            tgts_out[indices]  = y_tgt

        return (preds_out, tgts_out) if return_targets else preds_out

    # ------------------------------------------------------------------
    # evaluate  (convenience, mirrors common usage pattern)
    # ------------------------------------------------------------------

    def evaluate(self, dataset, batch_size: int = 50_000) -> Dict[str, float]:
        r"""
        Compute MAE, MSE, and R² on a dataset.

        Args:
            dataset: :class:`NNDataset`.
            batch_size (int): Passed to :meth:`predict`.

        Returns:
            dict with keys ``"mae"``, ``"mse"``, ``"r2"``.
        """
        y_pred, y_true = self.predict(dataset, return_targets=True, batch_size=batch_size)
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)

        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        mse = torch.mean((y_pred - y_true) ** 2).item()
        var = torch.var(y_true, correction=0).item()
        r2  = 1.0 - mse / var if var > 0 else float("nan")

        return {"mae": mae, "mse": mse, "r2": r2}

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        r"""
        Persist the full ensemble to a directory.

        Saved artefacts::

            <path>/
              meta.json
              kmeans.pkl
              scalers.pkl
              model_0.pth … model_{N-1}.pth

        Args:
            path (str): Target directory (created if it does not exist).
        """
        os.makedirs(path, exist_ok=True)

        # meta.json — all hyperparameters needed to reconstruct the object
        meta = {
            "n_clusters":        self.n_clusters,
            "n_centers":         self.n_centers,
            "reg_lambda":        self.reg_lambda,
            "center_sampling":   self.center_sampling,
            "gamma_mode":        self.gamma_mode,
            "gamma_k":           self.gamma_k,
            "gamma_alpha":       self.gamma_alpha,
            "kmeans_batch_size": self.kmeans_batch_size,
            "kmeans_n_init":     self.kmeans_n_init,
            "fit_batch_size":    self.fit_batch_size,
            "seed":              self.seed,
            "model_name":        self.model_name,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # KMeans router
        with open(os.path.join(path, "kmeans.pkl"), "wb") as f:
            pickle.dump(self._kmeans, f)

        # Per-cluster scalers
        with open(os.path.join(path, "scalers.pkl"), "wb") as f:
            pickle.dump(
                {
                    "input_scalers":  self._input_scalers,
                    "output_scalers": self._output_scalers,
                },
                f,
            )

        # Local models
        for i, model in enumerate(self._models):
            if model is not None:
                model.save(os.path.join(path, f"model_{i}.pth"))

        self._log(f"[MultiRBFELM] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE, verbose: bool = True) -> "MultiRBFELM":
        r"""
        Restore a persisted ensemble from a directory.

        Args:
            path (str): Directory previously created by :meth:`save`.
            device (torch.device): Device to load models onto.
            verbose (bool): Print progress (default: ``True``).

        Returns:
            A fully restored :class:`MultiRBFELM` instance.
        """
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)

        obj = cls(
            n_clusters        = meta["n_clusters"],
            n_centers         = meta["n_centers"],
            reg_lambda        = meta["reg_lambda"],
            center_sampling   = meta["center_sampling"],
            gamma_mode        = meta["gamma_mode"],
            gamma_k           = meta["gamma_k"],
            gamma_alpha       = meta["gamma_alpha"],
            kmeans_batch_size = meta["kmeans_batch_size"],
            kmeans_n_init     = meta["kmeans_n_init"],
            fit_batch_size    = meta["fit_batch_size"],
            device            = device,
            seed              = meta["seed"],
            model_name        = meta["model_name"],
            verbose           = verbose,
        )

        with open(os.path.join(path, "kmeans.pkl"), "rb") as f:
            obj._kmeans = pickle.load(f)

        with open(os.path.join(path, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)
        obj._input_scalers  = scalers["input_scalers"]
        obj._output_scalers = scalers["output_scalers"]

        for i in range(obj.n_clusters):
            model_path = os.path.join(path, f"model_{i}.pth")
            if os.path.exists(model_path):
                obj._models[i] = RBFELM.load(model_path, device=device, verbose=False)

        if verbose:
            loaded = sum(m is not None for m in obj._models)
            print(f"[MultiRBFELM] Loaded {loaded}/{obj.n_clusters} models from {path}")

        return obj


# ---------------------------------------------------------------------------
# Internal helper — wraps raw tensors so _subset_dataset can index them
# ---------------------------------------------------------------------------

class _TensorDatasetWrapper:
    """Minimal dataset wrapper around two tensors (X, y)."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self._X = X
        self._y = y

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]

    def __iter__(self):
        return iter((self._X, self._y))