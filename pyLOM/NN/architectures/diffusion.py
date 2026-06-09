
from .. import DiT, Unet1D
from fluidFlow.flow_matching import create_flow_matching
from fluidFlow.trainer import Trainer

import shutil
from typing import Union, Optional
from pathlib import Path

from torch.utils.data import Dataset
import torch
import numpy as np


class Diffusion:
    r"""
    Diffusion model that uses the implementaion of fluidFlow (https://github.com/DavidRamosArchilla/FluidFlow)
    Args:
        neural_net: the neural network to be trained
        input_size: the size of the input data
        cond_scale: the scale of the conditioning (if any)
        shifted_mu: the amount of shift for the mean in the loss calculation
        sampler_atol: the absolute tolerance for the sampler
        sampler_rtol: the relative tolerance for the sampler
        num_sampling_steps: the number of sampling steps to use during training
        sampler_timestep_shift: the amount of shift for the timestep during sampling
        sampling_method: the method to use for sampling (euler or dopri5)
        reverse_sampling: whether to reverse the sampling direction (from x_T to x_0 instead of x_0 to x_T)
        transport_kwargs: the keyword arguments for creating the transport object 
    """
    def __init__(
        self,
        model: Union[DiT, Unet1D],
        input_size: int,
        cond_scale: float = 1,
        shifted_mu: float = 0,
        sampler_atol: float = 1e-6,
        sampler_rtol: float = 1e-3,
        num_sampling_steps: int = 50,
        sampler_timestep_shift: float = 0.0,
        sampling_method: str = "euler",
        reverse_sampling: bool = False,
        results_folder: str = "./results",
        use_cpu=False,
        **transport_kwargs,
    ):
        
        self.flow_matching = create_flow_matching(
            model,
            input_size=input_size,
            cond_scale=cond_scale,
            shifted_mu=shifted_mu,
            sampler_atol=sampler_atol,
            sampler_rtol=sampler_rtol,
            num_sampling_steps=num_sampling_steps,
            sampler_timestep_shift=sampler_timestep_shift,
            sampling_method=sampling_method,
            reverse_sampling=reverse_sampling,
            **transport_kwargs,
        )
        self.results_folder = results_folder
        self.use_cpu = use_cpu
        self.trainer =  Trainer(self.flow_matching, [0], results_folder=self.results_folder, use_cpu=use_cpu)  # placeholder until fit() or load() is called

    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: Dataset,
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100_000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        amp: bool = False,
        mixed_precision_type: str = "bf16",
        split_batches: bool = True,
        max_grad_norm: Optional[float] = None,
        dataset_test: Optional[Dataset] = None,
        eta_min_scheduler: Optional[float] = None,
        compile_model: bool = False,
        use_fsdop: bool = False,
        use_muon: bool = False,
    ):
        """
        Train the flow matching model on the provided dataset.
        
        Args:
            dataset (Dataset): The training dataset.
            train_batch_size (int, optional): Batch size for training. Defaults to 16.
            gradient_accumulate_every (int, optional): Number of steps to accumulate gradients. Defaults to 1.
            train_lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            train_num_steps (int, optional): Total number of training steps. Defaults to 100,000.
            ema_update_every (int, optional): Frequency of EMA updates in steps. Defaults to 10.
            ema_decay (float, optional): Decay rate for exponential moving average. Defaults to 0.995.
            adam_betas (tuple, optional): Beta coefficients for the Adam optimizer. Defaults to (0.9, 0.99).
            save_and_sample_every (int, optional): Frequency of saving checkpoints and sampling in steps. Defaults to 1000.
            amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
            mixed_precision_type (str, optional): Type of mixed precision ('bf16' or 'fp16'). Defaults to "bf16".
            split_batches (bool, optional): Whether to split batches across devices. Defaults to True.
            max_grad_norm (Optional[float], optional): Maximum gradient norm for clipping. Defaults to None.
            dataset_test (Optional[Dataset], optional): Optional test dataset for validation. Defaults to None.
            eta_min_scheduler (Optional[float], optional): Minimum learning rate for the scheduler. Defaults to None.
            compile_model (bool, optional): Whether to compile the model. Defaults to False.
            use_fsdop (bool, optional): Whether to use fully sharded data parallel. Defaults to False.
            use_muon (bool, optional): Whether to use Muon optimizer. Defaults to False.
        
        Returns:
            dict: A dictionary containing training history with keys:
                - 'train_loss': List of training loss values.
                - 'test_loss': List of test loss values (if dataset_test is provided).
        """
        self.trainer = Trainer(
            self.flow_matching,
            dataset,
            results_folder=self.results_folder,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            train_lr=train_lr,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            amp=amp,
            mixed_precision_type=mixed_precision_type,
            split_batches=split_batches,
            max_grad_norm=max_grad_norm,
            dataset_test=dataset_test,
            eta_min_scheduler=eta_min_scheduler,
            compile_model=compile_model,
            use_fsdop=use_fsdop,
            use_muon=use_muon,
            use_cpu=self.use_cpu,
        )
        self.trainer.train()
        return {"train_loss": self.trainer.loss_history, "test_loss": self.trainer.test_loss_history}

    def predict(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        **sampling_kwargs,
    ) -> np.ndarray:
        """Generate samples conditioned on the classes in ``dataset``.

        Returns
            np.ndarray  shape (N, channels, seq_len) — only meaningful on rank 0.
        """
        if self.trainer is None:
            raise RuntimeError("Call fit() or load() before predict().")

        bs = batch_size or self.trainer.batch_size
        samples, _ = self.trainer.eval_model(dataset, batch_size=bs, use_autocast=True, **sampling_kwargs)

        if samples is None:
            return np.array([])  # non-main process in distributed run

        return samples.cpu().numpy()

 # ------------------------------------------------------------------
    # Internal milestone label used to bridge Trainer's naming convention
    # ------------------------------------------------------------------
    _MILESTONE = "checkpoint"
 
    def save(self, path: str) -> None:
        """Save a checkpoint to an arbitrary file path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        staging_folder = path.parent / f"_tmp_{path.stem}"
        staging_folder.mkdir(parents=True, exist_ok=True)
        original_folder = self.trainer.results_folder
        try:
            self.trainer.results_folder = staging_folder
            model_state_dict = self.trainer.accelerator.get_state_dict(self.trainer.model)
            self.trainer.save(self._MILESTONE, model_state_dict)
            shutil.move(str(staging_folder / f"model-{self._MILESTONE}.pt"), str(path))
        finally:
            self.trainer.results_folder = original_folder
            if staging_folder.exists():
                shutil.rmtree(staging_folder)
 
    def load(self, path: str) -> "Diffusion":
        """Restore a checkpoint from an arbitrary file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
 
        staging_folder = path.parent / f"_tmp_{path.stem}"
        staging_folder.mkdir(parents=True, exist_ok=True)
        original_folder = self.trainer.results_folder
        try:
            shutil.copy(str(path), str(staging_folder / f"model-{self._MILESTONE}.pt"))
            self.trainer.results_folder = staging_folder
            self.trainer.load(self._MILESTONE)
        except RuntimeError:
            self.trainer.model = torch.compile(self.trainer.model)
            self.trainer.load(self._MILESTONE)
        finally:
            self.trainer.results_folder = original_folder
            if staging_folder.exists():
                shutil.rmtree(staging_folder)
        return self
 
