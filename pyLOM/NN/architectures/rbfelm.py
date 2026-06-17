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
from ..optimizer            import OptunaOptimizer
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
            raiseError("Model has not been fitted yet. Call fit() first.")
        
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
    ) -> torch.Tensor:
        """
        Select ``n_centers`` rows from ``all_x`` as RBF centers.

        Args:
            all_x (torch.Tensor): All training inputs, shape ``(N, D)``.
            n_centers (int): Number of centers to select.
            method (str): Sampling method, either ``"random"`` or ``"uniform"``.
            gen (torch.Generator): RNG state (for reproducibility).

        Returns:
            torch.Tensor: Selected center rows, shape ``(n_centers, D)``.
        """
        N = all_x.shape[0]

        if method == "random":
            idx = torch.randperm(N, generator=gen)[:n_centers]
            return all_x[idx].detach().clone()

        elif method == "uniform":
            x_cpu = all_x.cpu()
            K = math.ceil(n_centers ** (1 / 3)) + 1

            mins = x_cpu.min(dim=0).values
            maxs = x_cpu.max(dim=0).values
            span = maxs - mins
            span = torch.where(span > 0, span, torch.ones_like(span))

            coords = ((x_cpu - mins) / span * K).long().clamp(0, K - 1)
            voxel_ids = coords[:, 0] * K * K + coords[:, 1] * K + coords[:, 2]

            order = torch.randperm(N, generator=gen)
            voxel_ids = voxel_ids[order]

            sorted_v, sort_idx = voxel_ids.sort()
            first_occ = torch.cat([torch.tensor([True]), sorted_v[1:] != sorted_v[:-1]])
            cell_representatives = order[sort_idx[first_occ]]

            n_occupied = cell_representatives.shape[0]

            if n_occupied >= n_centers:
                chosen = cell_representatives[torch.randperm(n_occupied, generator=gen)[:n_centers]]

            else:
                missing = n_centers - n_occupied
                mask = torch.ones(N, dtype=torch.bool)
                mask[cell_representatives] = False
                extra = torch.where(mask)[0]
                extra = extra[torch.randperm(extra.shape[0], generator=gen)[:missing]]
                chosen = torch.cat([cell_representatives, extra])

            return x_cpu[chosen].detach().clone()

        else:
            raiseError(f"Unknown center_sampling method: '{method}'. Choose either 'random' or 'uniform'.")
    
    @staticmethod
    def _estimate_local_gamma(
        centers: torch.Tensor,
        k: int = 10,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Estimate a per-center gamma value based on the local spacing between centers.
        For each center, the mean distance to its ``k`` nearest neighbours is used as a local length scale ``sigma``.

        Args:
            centers (torch.Tensor): RBF centers, shape ``(L, D)``.
            k (int): Number of nearest neighbours used to estimate local spacing (default: ``10``).
            alpha (float): Scaling factor for the bandwidth (default: ``1.0``). Values < 1 produce narrower kernels; > 1 produce wider kernels.

        Returns:
            torch.Tensor: Per-center gamma values, shape ``(L,)``, on the same device and dtype as ``centers``.
        """
        centers_np = centers.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors = k+1).fit(centers_np)
        distances, _ = nbrs.kneighbors(centers_np)
        local_spacing = distances[:, 1:].mean(axis=1)
        sigma = local_spacing
        sigma = np.where(sigma > 0, sigma, np.finfo(np.float32).eps)
        gamma = alpha / (sigma ** 2)
        return torch.tensor(gamma, device=centers.device, dtype=centers.dtype)

    @cr('RBFELM.fit')
    def fit(
        self,
        train_dataset:      torch.utils.data.Dataset,
        batch_size:         int = 16_384,
        dataloader_kwargs:  dict = {},
        use_cholmod:        bool = False,
        sparsity_threshold: float = 1e-6,
        save_logs_path:     Optional[str] = None,
        print_rate_batch:   int = 1,
        verbose:            bool = True,
        **kwargs,
    ) -> None:
        r"""
        Fit the RBF-ELM using block-wise accumulation of the normal equations.
        Supports both classical dense PyTorch solver and sparse CHOLMOD solver.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset to fit the model.
            batch_size (int, optional): Block size for DataLoader during fitting (default: ``10_000``).
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size`` (taken from the ``batch_size`` argument),``shuffle=True``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).
            use_cholmod (bool, optional): If ``True``, uses sparse CHOLMOD solver for the normal equations. Requires scikit-sparse to be installed. If ``False`` (default), uses classical dense PyTorch solver.
            sparsity_threshold (float, optional): Threshold for zeroing small values in H and H^T H when using CHOLMOD (default: ``1e-6``). Higher values produce sparser matrices but may affect accuracy.
            save_logs_path (str, optional): Path to save the training results. If ``None``, no results will be saved (default: ``None``).
            print_rate_batch (int, optional): Print progress every N batches (default: ``1``). Set to ``0`` to disable intermediate prints.
            verbose (bool, optional): If ``True``, prints detailed progress and sparsity metrics during fitting (default: ``True``).
        
        Returns:
            dict: A dictionary containing a "check" key with a list of boolean values indicating the success of the fitting process. Always returns ``{"check": [True]}`` if no exceptions are raised.

        """
        if use_cholmod: 
            try:
                import scipy.sparse as sp
                from sksparse.cholmod import cholesky as cholmod_cholesky
            except ImportError:
                raise ImportError(
                    "scikit-sparse is required for use_cholmod=True.\n"
                    "Install with: conda install -c conda-forge scikit-sparse\n"
                    "          or: pip install scikit-sparse"
                )

        _dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }

        dtype = torch.float32

        # Pass 1: sample centers and set up the RBFLayer
        all_x = train_dataset[:][0].to(self.device, dtype=dtype)
        N = all_x.shape[0]
        L = self.n_centers
        
        gen = torch.Generator(device="cpu")
        if self.seed is not None:
            gen.manual_seed(self.seed)

        selected_centers = self._sample_centers(
            all_x = all_x, 
            n_centers = L, 
            method = self.center_sampling, 
            gen = gen
        ).to(self.device)

        self.hidden.centers = selected_centers
        if self.gamma_mode == "local":
            local_gamma = self._estimate_local_gamma(centers=selected_centers, k=self.gamma_k, alpha=self.gamma_alpha)
            self.hidden.set_gamma(local_gamma)
        else:
            self.hidden.set_gamma(self.gamma)

        # Pass 2: block-wise accumulation
        total_blocks = math.ceil(N / batch_size)
        if not hasattr(self, "train_dataloader"):
            self.train_dataloader = DataLoader(train_dataset, **_dataloader_kwargs)

        if use_cholmod:
            HtH_sparse = sp.csc_matrix((L, L), dtype=np.float32)
            Hty_np = np.zeros((L, self.output_size), dtype=np.float32)
            if verbose:
                pprint(0, f"\n[fit_cholmod] Accumulating H^T H  (N={N}, L={L}, threshold={sparsity_threshold})")
        else:
            HtH_dense = torch.zeros((L, L), device=self.device, dtype=dtype)
            Hty_dense = torch.zeros((L, self.output_size), device=self.device, dtype=dtype)
            if verbose:
                pprint(0, f"\n[fit] Accumulating H^T H  (N={N}, L={L})")

        for b_idx, batch in enumerate(self.train_dataloader):
            x_blk, y_blk = batch[0].to(self.device, dtype=dtype), batch[1].to(self.device, dtype=dtype)
            if y_blk.dim() == 1:
                y_blk = y_blk.unsqueeze(-1)

            if use_cholmod:
                with torch.no_grad():
                    H_blk = self.hidden(x_blk).cpu()
                
                mask_H = H_blk.abs() >= sparsity_threshold
                H_blk_sparse = H_blk * mask_H
                
                HtH_blk = H_blk_sparse.T @ H_blk_sparse
                mask_HtH = HtH_blk.abs() >= sparsity_threshold
                nz_rows, nz_cols = mask_HtH.nonzero(as_tuple=True)
                nz_vals = HtH_blk[nz_rows, nz_cols].numpy().astype(np.float32)

                block_sparse = sp.csc_matrix(
                    (nz_vals, (nz_rows.numpy(), nz_cols.numpy())),
                    shape=(L, L),
                    dtype=np.float32,
                )
                HtH_sparse = HtH_sparse + block_sparse
                Hty_np += (H_blk_sparse.T @ y_blk.cpu()).numpy()

                if verbose and print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                    sparsity_H = 1.0 - mask_H.sum().item() / H_blk.numel()
                    sparsity_HtH = 1.0 - mask_HtH.sum().item() / (L * L)
                    pprint(
                        0, 
                        f"\r[fit_cholmod] Block {b_idx + 1}/{total_blocks}  "
                        f"H sparsity={sparsity_H:.3%}  "
                        f"H^T H sparsity={sparsity_HtH:.3%}  "
                        f"nnz(H^T H accumulated)={HtH_sparse.nnz:,}",
                        end="", flush=True,
                    )
            else:
                H_blk = self.hidden(x_blk)
                HtH_dense += H_blk.T @ H_blk
                Hty_dense += H_blk.T @ y_blk

                if verbose and print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                    threshold = 1e-6
                    sparsity_H = 1 - (H_blk.abs() > threshold).sum().item() / H_blk.numel()
                    HtH_blk = H_blk.T @ H_blk
                    sparsity_HtH = 1 - (HtH_blk.abs() > threshold).sum().item() / HtH_blk.numel()
                    pprint(
                        0,
                        f"\r[fit] Block {b_idx + 1}/{total_blocks}  "
                        f"H sparsity={sparsity_H:.3%}  "
                        f"H^T H sparsity={sparsity_HtH:.3%}",
                        end="", flush=True,
                    )

        # Solve normal equations
        if use_cholmod:
            if verbose:
                final_sparsity = 1.0 - HtH_sparse.nnz / (L * L)
                mem_mb = HtH_sparse.nnz * 4 / 1024 ** 2
                pprint(0, f"\n[fit_cholmod] H^T H assembled — nnz={HtH_sparse.nnz:,}  "
                      f"sparsity={final_sparsity:.3%}  mem≈{mem_mb:.1f} MB")
                pprint(0, f"[fit_cholmod] Factorising with CHOLMOD...")

            A = HtH_sparse + self.reg_lambda * sp.eye(L, format="csc", dtype=np.float32)
            A = sp.csc_matrix(A)
            del HtH_sparse

            factor = cholmod_cholesky(A)
            del A

            if verbose:
                pprint(0, f"[fit_cholmod] Solving for {self.output_size} output dim(s)...")

            beta_np = np.empty((L, self.output_size), dtype=np.float32)
            for o in range(self.output_size):
                beta_np[:, o] = factor(Hty_np[:, o])

            self.beta = torch.tensor(beta_np, dtype=dtype, device=self.device)
        else:
            I = torch.eye(L, device=self.device, dtype=dtype)
            self.beta = torch.linalg.solve(HtH_dense + self.reg_lambda * I, Hty_dense)

        if verbose:
            pprint(0, f"\n[fit] Done. Beta shape: {tuple(self.beta.shape)}")

        results = {"check": [True]}

        if save_logs_path is not None:
            pprint(0, f"\nPrinting losses on path: {save_logs_path}")
            if save_logs_path.endswith(".npy"):
                fn = save_logs_path
            else:
                fn = os.path.join(save_logs_path,f"training_results_{self.mname}.npy")
            np.save(fn, results)
                
        return results
    
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
        Save the model to a ``.pth`` checkpoint file.

        Args:
            path (str): File path or directory. A directory receives the automatic filename ``{model_name}.pth``.
        """
        self.checkpoint = self._define_checkpoint()
        if os.path.isdir(path):
            path = os.path.join(path, f"{self._model_name}.pth")
        torch.save(self._define_checkpoint(), path)
        pprint(0, f"\tModel saved at: {path}")

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict, device: torch.device, verbose: bool):
        model = cls(
            input_size      = checkpoint["input_size"],
            output_size     = checkpoint["output_size"],
            n_centers       = checkpoint["n_centers"],
            gamma           = checkpoint["gamma"],
            reg_lambda      = checkpoint["reg_lambda"],
            center_sampling = checkpoint.get("center_sampling", "random"),
            gamma_mode      = checkpoint.get("gamma_mode", "fixed"),
            gamma_k         = checkpoint.get("gamma_k", 10),
            gamma_alpha     = checkpoint.get("gamma_alpha", 1.0),
            device          = device,
            seed            = checkpoint["seed"],
            model_name      = checkpoint["model_name"],
            verbose         = verbose,
        )
        if checkpoint["centers"] is not None:
            model.hidden.centers = checkpoint["centers"].to(device)
        if checkpoint.get("gamma_tensor") is not None:
            model.hidden.set_gamma(checkpoint["gamma_tensor"].to(device))
        if checkpoint["beta"] is not None:
            model.beta = checkpoint["beta"].to(device)
        return model

    @classmethod
    def load(cls, path, device=DEVICE, verbose=True) -> "RBFELM":
        r"""
        Load the model from a checkpoint file. Does not require the model to be instantiated.

        Args:
            path (str): Path to the file to load the model from.
            device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
            verbose (bool, optional): If ``True``, prints detailed information about the loaded model (default: ``True``).

        Returns:
            model (RBFELM): The loaded model instance.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return cls._from_checkpoint(checkpoint, device, verbose)

    @classmethod
    def create_optimized_model(
        cls,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset:  torch.utils.data.Dataset,
        optuna_optimizer,
        **kwargs,
    ) -> Tuple["RBFELM", Dict]:
        r"""
        Create an optimized model using Optuna.
        Each trial instantiates a fresh ``RBFELM``, calls :meth:`fit`, and scores it on ``eval_dataset`` using MSE.  After all trials the best hyperparameters are used to build the returned model.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[RBFELM, Dict]: The optimized model and the optimization parameters.

        Example:
        >>> from pyLOM.NN import RBFELM, OptunaOptimizer
        >>> train_dataset, eval_dataset = dataset.get_splits([0.8, 0.2])
        >>> optimization_params = {
        ...     "n_centers":  (500, 5000),
        ...     "gamma":      (1e-3, 10.0),
        ...     "reg_lambda": (1e-10, 1e-4),
        ...     "batch_size": 10_000,
        ... }
        >>> optimizer = OptunaOptimizer(
        ...     optimization_params=optimization_params,
        ...     n_trials=30,
        ...     direction="minimize",
        ...     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
        ...     save_dir=None,
        ... )
        >>> model, best_params = RBFELM.create_optimized_model(
        ...     train_dataset, eval_dataset, optimizer
        ... )
        >>> model.fit(train_dataset, **best_params)
        """
        if not _OPTUNA_AVAILABLE:
            raiseError("Optuna is required for create_optimized_model. Install it with: pip install optuna")

        optimization_params = optuna_optimizer.optimization_params
        input_dim, output_dim = train_dataset[0][0].shape[0], train_dataset[0][1].shape[0]

        def suggest_value(name, space, trial):
            if isinstance(space, dict):
                suggested_dict = {}
                for key, subspace in space.items():
                    full_name = f"{name}.{key}"
                    suggested_dict[key] = suggest_value(full_name, subspace, trial)
                return suggested_dict
            
            if isinstance(space, (tuple, list)):
                low, high = space

                if isinstance(low, int) and isinstance(high, int):
                    def is_power_of_2(n):
                        return n > 0 and (n & (n - 1)) == 0
                    
                    if is_power_of_2(low) and is_power_of_2(high):
                        power_low = int(np.log2(low))
                        power_high = int(np.log2(high))
                        power_diff = power_high - power_low
                        
                        if power_diff > 1:
                            choices = [2**p for p in range(power_low, power_high + 1)]
                            return trial.suggest_categorical(name, choices)
                    
                    use_log = (high / max(1, low)) >= 1000
                    return trial.suggest_int(name, low, high, log=use_log)

                if isinstance(low, float) and isinstance(high, float):
                    use_log = (high / max(1e-12, low)) >= 1000
                    return trial.suggest_float(name, low, high, log=use_log)

            return space

        def optimization_function(trial) -> float:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = None

            try:
                training_params = {}
                for key, params in optimization_params.items():
                    training_params[key] = suggest_value(key, params, trial)
                training_params["save_logs_path"] = None

                model = cls(input_dim, output_dim, verbose=False, **training_params)
                results = model.fit(train_dataset, print_rate_batch=0, verbose=False, **training_params)
                y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                return float(((y_pred - y_true) ** 2).mean())

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"Trial {trial.number} failed due to out of memory error. Pruning the trial.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise TrialPruned()
                raise

            finally:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        # Update params with best ones
        OptunaOptimizer.apply_to(optimization_params, optimized_params=best_params)

        return cls(input_dim, output_dim, **optimization_params), optimization_params


class MultiRBFELM:
    r"""
    Ensemble of local RBF-ELM models trained on spatial clusters with interface overlap and blended inference.
    Useful for large datasets where a single RBF-ELM would require too many centers to capture local structure.

    Args:
        n_clusters (int): Number of spatial clusters (and local models).
        n_centers (int): RBF centers per local model. Capped at the cluster size automatically.
        overlap_factor (float): Controls the width of the shared zone. A point is added to cluster *i* if its distance to centroid *i* is less than ``overlap_factor`` times its distance to its primary centroid. ``1.0`` disables overlap (default: ``1.25``).
        reg_lambda (float): Tikhonov regularisation for each local model (default: ``1e-2``).
        center_sampling (str): Center-sampling strategy passed to each :class:`RBFELM` (default: ``"random"``).
        gamma_mode (str): Gamma estimation mode (default: ``"local"``).
        gamma_k (int): Neighbours used for local gamma estimation (default: ``30``).
        gamma_alpha (float): Bandwidth scaling factor (default: ``1.0``).
        blend_eps (float): Small constant added to squared distances when computing blending weights to avoid division by zero (default: ``1e-12``).
        kmeans_batch_size (int): ``batch_size`` for :class:`~sklearn.cluster.MiniBatchKMeans` (default: ``100_000``).
        kmeans_n_init (int): ``n_init`` for :class:`~sklearn.cluster.MiniBatchKMeans` (default: ``5``).
        fit_batch_size (int): DataLoader block size used during each local :class:`RBFELM` fit (default: ``100_000``).
        device (torch.device): Computation device (default: global ``DEVICE``).
        seed (int, optional): Master seed for reproducibility.
        model_name (str): Base name used when printing and saving (default: ``"multirbfelm"``).
        verbose (bool): Print progress logs (default: ``True``).
    """

    def __init__(
        self,
        n_clusters:         int,
        n_centers:          int = 10_000,
        overlap_factor:     float = 1.25,
        reg_lambda:         float = 1e-2,
        center_sampling:    str = "random",
        gamma_mode:         str = "local",
        gamma:              float = 1.0,
        gamma_k:            int = 30,
        gamma_alpha:        float = 1.0,
        blend_eps:          float = 1e-12,
        kmeans_batch_size:  int = 100_000,
        kmeans_n_init:      int = 5,
        fit_batch_size:     int = 100_000,
        device:             torch.device = DEVICE,
        seed:               Optional[int] = None,
        model_name:         str = "multirbfelm",
        verbose:            bool= True,
    ):
        self.n_clusters        = n_clusters
        self.n_centers         = n_centers
        self.overlap_factor    = overlap_factor
        self.reg_lambda        = reg_lambda
        self.center_sampling   = center_sampling
        self.gamma_mode        = gamma_mode
        self.gamma             = gamma
        self.gamma_k           = gamma_k
        self.gamma_alpha       = gamma_alpha
        self.blend_eps         = blend_eps
        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_n_init     = kmeans_n_init
        self.fit_batch_size    = fit_batch_size
        self.device            = device
        self.seed              = seed
        self.model_name        = model_name
        self.verbose           = verbose

        # Populated by fit() or load()
        self._kmeans:         Optional[MiniBatchKMeans] = None
        self._models:         List[Optional[RBFELM]] = [None] * n_clusters
        self._input_scalers:  List[Optional[object]] = [None] * n_clusters
        self._output_scalers: List[Optional[object]] = [None] * n_clusters

        self._log(f"MultiRBFELM — {n_clusters} clusters, up to {n_centers} centers each, overlap_factor={overlap_factor}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            pprint(0, msg)

    def _is_fitted(self) -> bool:
        return self._kmeans is not None and all(m is not None for m in self._models)

    def _build_kmeans(self) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters   = self.n_clusters,
            random_state = self.seed,
            batch_size   = self.kmeans_batch_size,
            n_init       = self.kmeans_n_init,
        )

    def _cluster_distances(self, X_np: np.ndarray) -> np.ndarray:
        centroids = self._kmeans.cluster_centers_                       # (K, D)
        diff = X_np[:, np.newaxis, :] - centroids[np.newaxis, :, :]     # Expand: (N,1,D) - (1,K,D) = (N,K,D)
        return (diff ** 2).sum(axis=-1)                                 # (N, K)

    def _active_clusters(self, dist_sq: np.ndarray) -> List[np.ndarray]:
        primary_dist_sq = dist_sq.min(axis=1, keepdims=True)            # (N, 1)
        threshold_sq = (self.overlap_factor ** 2) * primary_dist_sq     # (N, 1)
        active_mask = dist_sq <= threshold_sq                           # (N, K)
        return [np.where(row)[0] for row in active_mask]                # list of arrays

    @staticmethod
    def _subset_dataset(
        dataset,
        indices: np.ndarray,
        input_scaler,
        output_scaler,
    ):
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        X_full, y_full = dataset[:]
        X_sub = X_full[idx_tensor]
        y_sub = y_full[idx_tensor]

        return NNDataset(
            variables_out  = (y_sub,),
            variables_in   = X_sub,
            parameters     = None,
            inputs_scaler  = input_scaler,
            outputs_scaler = output_scaler,
        ), idx_tensor

    @cr('MultiRBFELM.fit')
    def fit(
        self,
        train_dataset,
        valid_dataset,
        save_dir: Optional[str] = None,
        reload_existing: bool = True,
    ) -> None:
        r"""
        Cluster the training set and fit one :class:`RBFELM` per cluster, including overlap points from neighbouring clusters.

        Args:
            train_dataset: Training :class:`pyLOM.NN.Dataset`.
            valid_dataset: Validation :class:`pyLOM.NN.Dataset`.
            save_dir (str, optional): If given, each local model is saved to ``<save_dir>/model_<i>.pth`` as it is trained, and reloaded from disk on subsequent calls when ``reload_existing=True``.
            reload_existing (bool): Skip retraining a cluster whose ``.pth`` file already exists in ``save_dir`` (default: ``True``).
        """
        X_train, y_train = train_dataset[:]
        X_valid, y_valid = valid_dataset[:]

        input_size  = X_train.shape[1]
        output_size = y_train.shape[1] if y_train.dim() > 1 else 1

        X_train_np = X_train.cpu().numpy()
        X_valid_np = X_valid.cpu().numpy()

        # KMeans clustering on training set
        self._log("\n[MultiRBFELM] Fitting KMeans on training set...")
        self._kmeans = self._build_kmeans()
        self._kmeans.fit(X_train_np)

        train_dist_sq = self._cluster_distances(X_train_np)    # (N_train, K)
        valid_dist_sq = self._cluster_distances(X_valid_np)    # (N_valid, K)

        train_primary = train_dist_sq.argmin(axis=1)           # (N_train,)
        valid_primary = valid_dist_sq.argmin(axis=1)           # (N_valid,)

        train_active = self._active_clusters(train_dist_sq)    # list of arrays

        primary_counts = np.bincount(train_primary, minlength=self.n_clusters)
        self._log(f"[MultiRBFELM] Primary cluster sizes (train): {primary_counts.tolist()}")

        # Build per-cluster lists of training indices (with overlap)
        cluster_train_indices = [[] for _ in range(self.n_clusters)]
        for n, active in enumerate(train_active):
            for i in active:
                cluster_train_indices[i].append(n)

        overlap_counts = [len(idxs) for idxs in cluster_train_indices]
        self._log(f"[MultiRBFELM] Effective cluster sizes after overlap (train): {overlap_counts}")

        # Train one RBFELM model per cluster
        for i in range(self.n_clusters):
            self._log(f"\n[MultiRBFELM] ── Cluster {i + 1}/{self.n_clusters} ──")

            model_path = os.path.join(save_dir, f"model_{i}.pth") if save_dir else None
            if reload_existing and model_path and os.path.exists(model_path):
                self._log(f"  Loading existing model from {model_path}")
                self._models[i] = RBFELM.load(model_path, device=self.device, verbose=False)
                continue

            # Find primary and overlap indices for this cluster
            all_indices_i = np.array(cluster_train_indices[i], dtype=np.int64)
            primary_mask_i = train_primary[all_indices_i] == i
            primary_indices = all_indices_i[primary_mask_i]

            n_total_i   = len(all_indices_i)
            n_primary_i = len(primary_indices)
            n_overlap_i = n_total_i - n_primary_i

            valid_indices_i = np.where(valid_primary == i)[0]
            n_valid_i = len(valid_indices_i)

            self._log(
                f"  Primary: {n_primary_i}  |  Overlap: {n_overlap_i}  "
                f"(total train: {n_total_i})  |  Valid: {n_valid_i}"
            )

            if n_primary_i == 0:
                self._log("  Empty primary cluster — skipping.")
                continue

            # Fit scalers on primary members
            input_scaler_i  = RobustScaler()
            output_scaler_i = RobustScaler()

            train_sub, _ = self._subset_dataset(train_dataset, all_indices_i, input_scaler_i, output_scaler_i)
            valid_sub, _ = self._subset_dataset(valid_dataset, valid_indices_i, input_scaler_i, output_scaler_i)

            self._input_scalers[i]  = input_scaler_i
            self._output_scalers[i] = output_scaler_i

            # Instantiate and fit local RBF-ELM
            n_centers_i = min(self.n_centers, n_total_i)

            model = RBFELM(
                input_size      = input_size,
                output_size     = output_size,
                n_centers       = n_centers_i,
                reg_lambda      = self.reg_lambda,
                center_sampling = self.center_sampling,
                gamma_mode      = self.gamma_mode,
                gamma           = self.gamma,
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

    @cr('MutiRBFELM.predict')
    def predict(
        self,
        dataset: torch.utils.data.Dataset,
        return_targets: bool = False,
        dataloader_kwargs: dict = {},
        **kwargs,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        r"""
        Predict target values by routing each point to its local model(s).
        Points that fall exclusively within one cluster's zone are handled by hard assignment. 
        Points that lie within the overlap zone of multiple clusters receive a weighted average prediction.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            return_targets (bool, optional): If ``True``, the true target values will be returned along with the predictions (default: ``False``).
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size=16_384`` ,``shuffle=False``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).

        Returns:
            ``np.ndarray`` of shape ``(N, output_size)``, or a ``(predictions, targets)`` tuple if ``return_targets=True``.
        """
        if not self._is_fitted():
            raiseError(f"Model has not been fitted. Call fit() or load() first.")

        X_all, y_all = dataset[:]
        N = X_all.shape[0]
        output_size = y_all.shape[1] if y_all.dim() > 1 else 1

        X_all_np = X_all.cpu().numpy()
        dist_sq = self._cluster_distances(X_all_np)
        active = self._active_clusters(dist_sq)

        # Numerator and denominator accumulators for the weighted blend
        weighted_sum = np.zeros((N, output_size), dtype=np.float64)
        weight_total = np.zeros((N, 1), dtype=np.float64)

        # Group points by their active-cluster sets to minimise repeated model calls
        cluster_to_points: List[List[int]] = [[] for _ in range(self.n_clusters)]
        for n, active_i in enumerate(active):
            for i in active_i:
                cluster_to_points[i].append(n)

        for i in range(self.n_clusters):
            point_indices_i = np.array(cluster_to_points[i], dtype=np.int64)
            if len(point_indices_i) == 0:
                continue

            model_i = self._models[i]
            input_scaler_i = self._input_scalers[i]
            output_scaler_i = self._output_scalers[i]

            if model_i is None:
                raiseError(f"Model for cluster {i} is not fitted.")

            sub_dataset, _ = self._subset_dataset(
                dataset,
                point_indices_i,
                input_scaler_i,
                output_scaler_i,
            )

            if return_targets:
                y_pred_scaled, y_true_scaled = model_i.predict(sub_dataset, return_targets=True, dataloader_kwargs=dataloader_kwargs, **kwargs)
                y_pred = output_scaler_i.inverse_transform(y_pred_scaled)
                y_true = output_scaler_i.inverse_transform(y_true_scaled)

            else:
                y_pred_scaled = model_i.predict(sub_dataset, return_targets=False, dataloader_kwargs=dataloader_kwargs, **kwargs)
                y_pred = output_scaler_i.inverse_transform(y_pred_scaled)

            # Weights: inverse squared distance to centroid i
            d_sq_i = dist_sq[point_indices_i, i]
            w_i = 1.0 / (d_sq_i + self.blend_eps)

            weighted_sum[point_indices_i] += w_i[:, np.newaxis] * y_pred
            weight_total[point_indices_i] += w_i[:, np.newaxis]

        preds_out = (weighted_sum / weight_total).astype(np.float32)

        if return_targets:
            return preds_out, y_true

        return preds_out

    def save(self, path: str) -> None:
        r"""
        Save the model to a ``.pth`` checkpoint file.

        Args:
            path (str): File path or directory. A directory receives the automatic filename ``{model_name}.pth``.
        """
        if os.path.isdir(path):
            path = os.path.join(path, f"{self.model_name}.pth")
        torch.save({
            "meta": {
                "n_clusters":        self.n_clusters,
                "n_centers":         self.n_centers,
                "overlap_factor":    self.overlap_factor,
                "reg_lambda":        self.reg_lambda,
                "center_sampling":   self.center_sampling,
                "gamma_mode":        self.gamma_mode,
                "gamma_k":           self.gamma_k,
                "gamma_alpha":       self.gamma_alpha,
                "blend_eps":         self.blend_eps,
                "kmeans_batch_size": self.kmeans_batch_size,
                "kmeans_n_init":     self.kmeans_n_init,
                "fit_batch_size":    self.fit_batch_size,
                "seed":              self.seed,
                "model_name":        self.model_name,
            },
            "kmeans":         self._kmeans,
            "input_scalers":  self._input_scalers,
            "output_scalers": self._output_scalers,
            "models":         [m._define_checkpoint() if m is not None else None for m in self._models],
        }, path)
        self._log(f"[MultiRBFELM] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE, verbose: bool = True) -> "MultiRBFELM":
        r"""
        Load the model from a checkpoint file. Does not require the model to be instantiated.
        
        Args:
            path (str): Path to the file to load the model from.
            device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
            verbose (bool, optional): If ``True``, prints detailed information about the loaded model (default: ``True``).

        Returns:
            model (MultiRBFELM): The loaded model instance.
        """
        data = torch.load(path, map_location=device, weights_only=False)
        meta = data["meta"]

        obj = cls(
            n_clusters        = meta["n_clusters"],
            n_centers         = meta["n_centers"],
            overlap_factor    = meta.get("overlap_factor", 1.0),
            reg_lambda        = meta["reg_lambda"],
            center_sampling   = meta["center_sampling"],
            gamma_mode        = meta["gamma_mode"],
            gamma_k           = meta["gamma_k"],
            gamma_alpha       = meta["gamma_alpha"],
            blend_eps         = meta.get("blend_eps", 1e-12),
            kmeans_batch_size = meta["kmeans_batch_size"],
            kmeans_n_init     = meta["kmeans_n_init"],
            fit_batch_size    = meta["fit_batch_size"],
            device            = device,
            seed              = meta["seed"],
            model_name        = meta["model_name"],
            verbose           = verbose,
        )

        obj._kmeans         = data["kmeans"]
        obj._input_scalers  = data["input_scalers"]
        obj._output_scalers = data["output_scalers"]

        obj._models = []
        for ckpt in data["models"]:
            if ckpt is None:
                obj._models.append(None)
            else:
                model = RBFELM._from_checkpoint(ckpt, device=device, verbose=False)
                obj._models.append(model)

        if verbose:
            loaded = sum(m is not None for m in obj._models)
            pprint(0, f"[MultiRBFELM] Loaded {loaded}/{obj.n_clusters} models from {path}")

        return obj