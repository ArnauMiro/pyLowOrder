import os
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple, Callable
from ..optimizer import OptunaOptimizer, TrialPruned
from .. import DEVICE, set_seed  # pyLOM/NN/__init__.py
from ... import pprint, cr  # pyLOM/__init__.py
from ...utils.errors import raiseWarning, raiseError


class MLP(nn.Module):
    r"""
    Multi-layer perceptron model for regression tasks. The model is based on the PyTorch library `torch.nn` 
    (detailed documentation can be found at https://pytorch.org/docs/stable/nn.html).

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        n_layers (int): Number of hidden layers.
        hidden_size (int): Number of neurons in each hidden layer.
        p_dropouts (float, optional): Dropout probability for the hidden layers (default: ``0.0``).
        activation (torch.nn.Module, optional): Activation function to use (default: ``torch.nn.functional.relu``).
        device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
        initialization (Callable, optional): Initialization function for the weights (default: ``torch.nn.init.xavier_uniform_``).
        initialization_kwargs (Dict, optional): Additional keyword arguments for the initialization function (default: ``{}``).
        seed (int, optional): Seed for reproducibility (default: ``None``).
        model_name (str, optional): Name of the model used as a base for the model name (default: ``"mlp"``).
        verbose (bool, optional): If ``True``, prints the model parameters and total size (default: ``True``).
        kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        p_dropouts: float = 0.0,
        activation: torch.nn.Module = torch.nn.functional.relu,
        device: torch.device = DEVICE,
        initialization: Callable = torch.nn.init.xavier_uniform_,
        initialization_kwargs: Dict = {},
        seed: int = None,
        model_name: str = "mlp",
        verbose: bool = True,
        **kwargs: Dict,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.p_dropouts = p_dropouts
        self.activation = activation
        self.device = device
        self.initialization = initialization
        self.initialization_kwargs = initialization_kwargs
        self.seed = seed
        self.model_name = model_name

        super().__init__()
        if seed is not None:
            set_seed(seed)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = hidden_size
            self.layers.append(nn.Linear(in_size, out_size))
            if p_dropouts > 0:
                self.layers.append(nn.Dropout(p_dropouts))
        self.oupt = nn.Linear(hidden_size, output_size)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.initialization(layer.weight, **self.initialization_kwargs)
                nn.init.zeros_(layer.bias)
        self.initialization(self.oupt.weight, **self.initialization_kwargs)
        nn.init.zeros_(self.oupt.bias)

        self.to(self.device)
        if verbose:
            pprint(0, f"Creating model: {self._model_name}")
            keys_print = [
                "input_size",
                "output_size",
                "n_layers",
                "hidden_size",
                "p_dropouts",
                "activation",
                "device",
                "initialization",
                "initialization_kwargs",
                "seed",
                "model_name",
            ]
            for key in keys_print:
                value = getattr(self, key)
                if callable(value):
                    value = value.__name__
                pprint(0, f"\t{key}: {value}")
            pprint(
                0,
                f"\ttotal_size (trainable parameters): {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n"
            )
    
    def forward(self, x):
        for layer in self.layers:
            z = self.activation(layer(x))
            x = z
        z = self.oupt(x)
        return z
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str) -> None:
        if not isinstance(value, str):
            raiseError("model_name must be a string")
        value = value.strip()
        if not value:
            raiseError("model_name cannot be empty")
        self._model_name = value
    
    @cr('MLP.fit')
    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        optimizer_kwargs: dict = {},
        scheduler_kwargs: dict = {},
        save_logs_path: str = None,
        print_rate_batch: int = 0,
        print_rate_epoch: int = 1,
        save_best: bool = False,
        ddp: bool = False,
        ddp_backend: str = "nccl",
        local_rank: int = None,
        find_unused_parameters: bool = False,
        **kwargs,
    )-> Dict[str, List[float]]:
        r"""
        Fit the model to the training data. If eval_set is provided, the model will be evaluated on this set after each epoch. 
        
        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset to fit the model.
            eval_dataset (torch.utils.data.Dataset, optional): Evaluation dataset to evaluate the model after each epoch (default: ``None``).
            epochs (int, optional): Number of epochs to train the model (default: ``100``).
            batch_size (int, optional): Batch size for training (default: ``32``).
            lr (float, optional): Learning rate for the optimizer (default: ``0.001``).
            loss_fn (torch.nn.Module, optional): Loss function to optimize (default: ``torch.nn.MSELoss()``).
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class to use (default: ``torch.optim.Adam``).
            scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler class to use. If ``None``, no scheduler will be used (default: ``None``).
            optimizer_kwargs (dict, optional): Additional keyword arguments to pass to the optimizer (default: ``{}``).
            scheduler_kwargs (dict, optional): Additional keyword arguments to pass to the scheduler (default: ``{}``).
            save_logs_path (str, optional): Path to save the training results. If ``None``, no results will be saved (default: ``None``).
            print_rate_batch (int, optional): Print loss every ``print_rate_batch`` batches (default: ``1``). If set to ``0``, no print will be done.
            print_rate_epoch (int, optional): Print loss every ``print_rate_epoch`` epochs (default: ``1``). If set to ``0``, no print will be done.
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                - shuffle (bool, optional): Shuffle the data (default: ``True``).
                - num_workers (int, optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, optional): Pin memory (default: ``True``).
            ddp (bool, optional): Enable DistributedDataParallel for multi-GPU training (default: ``False``).
            ddp_backend (str, optional): Backend for torch.distributed (default: ``"nccl"``).
            local_rank (int, optional): Local GPU rank. If ``None``, inferred from ``LOCAL_RANK`` env (default: ``None``).
            find_unused_parameters (bool, optional): Passed to DDP wrapper (default: ``False``).

        Returns:
            Dict[str, List[float]]: Dictionary containing the training and evaluation results:
                - "train_loss": List of training losses for each epoch.
                - "test_loss": List of evaluation losses for each epoch (if eval_dataset is provided
                - "lr": List of learning rates for each epoch.
                - "loss_iterations_train": List of training losses for each iteration.
                - "loss_iterations_test": List of evaluation losses for each iteration (if eval_dataset is provided).
                - "grad_norms": List of gradient norms for each iteration.
                - "check": List with a single boolean indicating successful training.
        """
        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
        }

        # --- DDP setup (minimal) ---
        if ddp:
            # Infer local rank from torch.run/torch.distributed.launch if not provided
            if local_rank is None:
                env_lr = (
                    os.environ.get("LOCAL_RANK")
                    or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
                    or os.environ.get("SLURM_LOCALID")
                )
                local_rank = int(env_lr) if env_lr is not None else 0

            # Initialize process group if needed
            if not (dist.is_available() and dist.is_initialized()):
                try:
                    # Populate env:// variables if launched with MPI/SLURM
                    if os.environ.get("RANK") is None:
                        mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("SLURM_PROCID")
                        if mpi_rank is not None:
                            os.environ["RANK"] = str(mpi_rank)
                    if os.environ.get("WORLD_SIZE") is None:
                        mpi_world = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("SLURM_NTASKS")
                        if mpi_world is not None:
                            os.environ["WORLD_SIZE"] = str(mpi_world)
                    if os.environ.get("LOCAL_RANK") is None:
                        mpi_local = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
                        if mpi_local is not None:
                            os.environ["LOCAL_RANK"] = str(mpi_local)
                    # Reasonable single-node defaults if missing
                    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                    os.environ.setdefault("MASTER_PORT", "29500")
                    dist.init_process_group(backend=ddp_backend)
                except Exception as e:
                    raiseError(f"Failed to init DDP process group: {e}")

            # Set device and move model
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cpu")
            self.to(self.device)

        # World/rank info and main process check for logging/saving
        rank = 0
        world_size = 1
        is_main_process = True
        if ddp and dist.is_available() and dist.is_initialized():
            try:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                is_main_process = rank == 0
            except Exception:
                is_main_process = True

        # Build DataLoaders (use DistributedSampler when ddp=True)
        if ddp:
            # Samplers override shuffle
            dataloader_params["shuffle"] = False
            for key in dataloader_params.keys():
                if key in kwargs:
                    dataloader_params[key] = kwargs[key]
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            self.train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler, **dataloader_params)
        
            if eval_dataset is not None:
                self.eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
                self.eval_dataloader = DataLoader(eval_dataset, sampler=self.eval_sampler, **dataloader_params)
        else:
            if not hasattr(self, "train_dataloader"):
                for key in dataloader_params.keys():
                    if key in kwargs:
                        dataloader_params[key] = kwargs[key]
                self.train_dataloader = DataLoader(train_dataset, **dataloader_params)
            if not hasattr(self, "eval_dataloader") and eval_dataset is not None:
                for key in dataloader_params.keys():
                    if key in kwargs:
                        dataloader_params[key] = kwargs[key]
                self.eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

        # Wrap model with DDP if requested
        model = self
        if ddp:
            try:
                model = DDP(
                    self,
                    device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=find_unused_parameters,
                )
            except Exception as e:
                raiseError(f"Failed to wrap model with DDP: {e}")

        if not hasattr(self, "optimizer"):
            self.optimizer = optimizer_class(
                model.parameters(),
                lr=lr,
                **optimizer_kwargs
            )
        if not hasattr(self, "scheduler"):
            if scheduler_class is not None:
                self.scheduler = scheduler_class(
                    self.optimizer,
                    **scheduler_kwargs
                )
            else:
                self.scheduler = None
        
        if hasattr(self, "checkpoint"):
            results_fn = os.path.join(save_logs_path,f"training_results_{self._model_name}.npy")
            if "optimizer" in self.checkpoint:
                self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            if "scheduler" in self.checkpoint and self.scheduler is not None: 
                self.scheduler.load_state_dict(self.checkpoint["scheduler"])
            if os.path.isfile(results_fn):
                results = np.load(results_fn, allow_pickle=True).item()
                train_losses = results["train_loss"].tolist()
                test_losses = results["test_loss"].tolist()
                loss_iterations_train = results["loss_iterations_train"].tolist()
                loss_iterations_test = results["loss_iterations_test"].tolist()
                current_lr_vec = results["lr"].tolist()
                grad_norms = results["grad_norms"].tolist()
                epoch_times = results.get("epoch_time_s", []).tolist()
                throughput_local = results.get("throughput_samples_per_sec_local", []).tolist()
                throughput_global = results.get("throughput_samples_per_sec_global", []).tolist()
            else:
                train_losses = []
                test_losses = []
                loss_iterations_train = []
                loss_iterations_test = []
                current_lr_vec = []
                grad_norms = []
                epoch_times = []
                throughput_local = []
                throughput_global = []
        else:
            train_losses = []
            test_losses = []
            loss_iterations_train = []
            loss_iterations_test = []
            current_lr_vec = []
            grad_norms = []
            epoch_times = []
            throughput_local = []
            throughput_global = []

        total_epochs = len(train_losses) + epochs
        self.mname = f"{self._model_name}_{total_epochs:05d}"
        for epoch in range(1+len(train_losses), 1+total_epochs):
            train_loss = 0.0
            model.train()

            # Ensure sampler shuffles differently each epoch under DDP
            if ddp:
                try:
                    self.train_sampler.set_epoch(epoch)
                except Exception:
                    pass

            # Timing and throughput accounting
            epoch_start_time = time.perf_counter()
            samples_train_epoch_local = 0

            def closure():
                self.optimizer.zero_grad()
                oupt = model(x_train)
                loss_val = loss_fn(oupt, y_train)
                loss_val.backward()
                loss_iterations_train.append(loss_val.item())
                return loss_val

            for b_idx, batch in enumerate(self.train_dataloader):
                x_train, y_train = batch[0].to(self.device), batch[1].to(self.device)
                samples_train_epoch_local += x_train.size(0)
                train_loss += self.optimizer.step(closure).item()
                total_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
                grad_norms.append(total_norm.item())
        
                if is_main_process and print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                    pprint(
                        0,
                        f"\tBatch {b_idx+1}/{len(self.train_dataloader)}, Train Loss: {train_loss:.4e}",
                        flush=True
                    )
                    
            # Global average train loss across ranks
            train_batches_local = b_idx + 1
            train_loss_sum_local = train_loss
            if ddp and dist.is_available() and dist.is_initialized():
                agg = torch.tensor([train_loss_sum_local, float(train_batches_local)], device=self.device)
                dist.all_reduce(agg, op=dist.ReduceOp.SUM)
                train_loss = (agg[0] / agg[1]).item()
            else:
                train_loss = train_loss_sum_local / train_batches_local
            train_losses.append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            
            test_loss = 0.0
            if eval_dataset is not None:
                model.eval()
                with torch.no_grad():
                    test_loss_sum_local = 0.0
                    test_batches_local = 0
                    eval_samples_epoch_local = 0
                    for n_idx, sample in enumerate(self.eval_dataloader):
                        x_test, y_test = sample[0].to(self.device), sample[1].to(self.device)
                        test_output = model(x_test)
                        loss_val = loss_fn(test_output, y_test)
                        loss_iterations_test.append(loss_val.item())
                        test_loss_sum_local += loss_val.item()
                        test_batches_local += 1
                        eval_samples_epoch_local += x_test.size(0)

                # Compute global eval loss (mean over all ranks and batches)
                if ddp and dist.is_available() and dist.is_initialized():
                    agg_eval = torch.tensor([test_loss_sum_local, float(test_batches_local)], device=self.device)
                    dist.all_reduce(agg_eval, op=dist.ReduceOp.SUM)
                    test_loss = (agg_eval[0] / agg_eval[1]).item()
                else:
                    test_loss = test_loss_sum_local / max(1, test_batches_local)
                test_losses.append(test_loss)

                if save_best and (len(test_losses) == 1 or test_loss < min(test_losses[:-1])):
                    if is_main_process and save_logs_path is not None:
                        best_model_file = os.path.join(save_logs_path, f"best_model_{self._model_name}.pth")
                        self.save(best_model_file)
                    elif is_main_process and save_best:
                        raiseWarning("The argument save_best is set to True but no save_logs_path is provided. The best model will not be saved.")
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            current_lr_vec.append(current_lr)
            
            if is_main_process and print_rate_epoch != 0 and (epoch % print_rate_epoch) == 0:
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / (1024**2)  # Memory usage in MB
                    memory_usage_str = f", MEM: {mem_used:.2f} MB"
                else:
                    memory_usage_str = ""
                test_log = f", Test Loss: {test_loss:.4e}" if eval_dataset is not None else ""
                pprint(
                    0,
                    f"\tEpoch {epoch}/{total_epochs}, Train Loss: {train_loss:.4e}{test_log}, "
                    f"LR: {current_lr:.2e}{memory_usage_str}",
                    flush=True
                )

            # End epoch timing and throughput
            epoch_time = time.perf_counter() - epoch_start_time
            epoch_times.append(epoch_time)
            # Local throughput
            throughput_local.append(samples_train_epoch_local / max(1e-12, epoch_time))
            # Global throughput: sum samples across ranks, divide by max epoch time
            if ddp and dist.is_available() and dist.is_initialized():
                # Sum samples
                t_samp = torch.tensor([float(samples_train_epoch_local)], device=self.device)
                dist.all_reduce(t_samp, op=dist.ReduceOp.SUM)
                total_samples_all = t_samp.item()
                # Max time
                t_time = torch.tensor([epoch_time], device=self.device)
                dist.all_reduce(t_time, op=dist.ReduceOp.MAX)
                epoch_time_max = t_time.item()
                throughput_global.append(total_samples_all / max(1e-12, epoch_time_max))
            else:
                throughput_global.append(samples_train_epoch_local / max(1e-12, epoch_time))

        results = {
            "train_loss": np.array(train_losses),
            "test_loss": np.array(test_losses),
            "lr": np.array(current_lr_vec),
            "loss_iterations_train": np.array(loss_iterations_train),
            "loss_iterations_test": np.array(loss_iterations_test),
            "grad_norms": np.array(grad_norms),
            "check": [True],
            # Extra metrics for DDP/normal comparison
            "epoch_time_s": np.array(epoch_times),
            "throughput_samples_per_sec_local": np.array(throughput_local),
            "throughput_samples_per_sec_global": np.array(throughput_global),
            "mode": "ddp" if (ddp and dist.is_available() and dist.is_initialized()) else "single",
            "ddp_info": {
                "enabled": bool(ddp),
                "backend": ddp_backend if ddp else None,
                "world_size": int(world_size),
                "rank": int(rank),
                "local_rank": int(local_rank) if local_rank is not None else 0,
            },
        }

        if is_main_process and save_logs_path is not None:
            pprint(0, f"\nPrinting losses on path: {save_logs_path}")
            fn = os.path.join(save_logs_path,f"training_results_{self._model_name}.npy")
            np.save(fn, results)

        return results

    @staticmethod
    def summarize_results(results: Dict) -> Dict:
        """
        Produce a compact summary of a training run's key metrics.
        Returns a dict with final losses, average epoch time and throughputs.
        """
        summary = {}
        try:
            train_loss = results.get("train_loss")
            test_loss = results.get("test_loss")
            epoch_time = results.get("epoch_time_s")
            thr_glob = results.get("throughput_samples_per_sec_global")
            thr_loc = results.get("throughput_samples_per_sec_local")
            summary.update({
                "mode": results.get("mode", "single"),
                "final_train_loss": float(train_loss[-1]) if train_loss is not None and len(train_loss)>0 else None,
                "final_test_loss": float(test_loss[-1]) if test_loss is not None and len(test_loss)>0 else None,
                "avg_epoch_time_s": float(np.mean(epoch_time)) if epoch_time is not None and len(epoch_time)>0 else None,
                "avg_throughput_global": float(np.mean(thr_glob)) if thr_glob is not None and len(thr_glob)>0 else None,
                "avg_throughput_local": float(np.mean(thr_loc)) if thr_loc is not None and len(thr_loc)>0 else None,
            })
            ddp_info = results.get("ddp_info", {})
            summary.update({
                "world_size": ddp_info.get("world_size", 1),
                "rank": ddp_info.get("rank", 0),
                "backend": ddp_info.get("backend", None),
            })
        except Exception:
            pass
        return summary

    @staticmethod
    def compare_results(results_a: Dict, results_b: Dict) -> Dict:
        """
        Compare two training runs (e.g., single vs DDP) and return key ratios.
        Keys returned include speedup and loss ratios for quick reporting.
        """
        sa = MLP.summarize_results(results_a)
        sb = MLP.summarize_results(results_b)
        comp = {
            "mode_a": sa.get("mode"),
            "mode_b": sb.get("mode"),
            "speedup_global": None,
            "speedup_local": None,
            "final_train_loss_ratio": None,
            "final_test_loss_ratio": None,
        }
        try:
            if sa.get("avg_throughput_global") and sb.get("avg_throughput_global"):
                comp["speedup_global"] = float(sb["avg_throughput_global"]) / float(sa["avg_throughput_global"])  # b over a
        except Exception:
            pass
        try:
            if sa.get("avg_throughput_local") and sb.get("avg_throughput_local"):
                comp["speedup_local"] = float(sb["avg_throughput_local"]) / float(sa["avg_throughput_local"])  # b over a
        except Exception:
            pass
        try:
            if sa.get("final_train_loss") and sb.get("final_train_loss"):
                comp["final_train_loss_ratio"] = float(sb["final_train_loss"]) / float(sa["final_train_loss"])  # b over a
        except Exception:
            pass
        try:
            if sa.get("final_test_loss") and sb.get("final_test_loss"):
                comp["final_test_loss_ratio"] = float(sb["final_test_loss"]) / float(sa["final_test_loss"])  # b over a
        except Exception:
            pass
        return comp

    @cr('MLP.predict')
    def predict(
        self, 
        X: torch.utils.data.Dataset, 
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded to a DataLoader with the provided keyword arguments. 
        The model is set to evaluation mode and the predictions are made using the input data. 
        To make a prediction from a torch tensor, use the `__call__` method directly.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            return_targets (bool, optional): If ``True``, the true target values will be returned along with the predictions (default: ``False``).
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                
                - batch_size (int, optional): Batch size (default: ``256``).
                - shuffle (bool, optional): Shuffle the data (default: ``False``).
                - num_workers (int, optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, optional): Pin memory (default: ``True``).
 
        Returns:
            Tuple [np.ndarray, np.ndarray]: The predictions and the true target values.
        """
        dataloader_params = {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
        }
        
        for key in dataloader_params.keys():
            if key in kwargs:
                dataloader_params[key] = kwargs[key]

        predict_dataloader = DataLoader(X, **dataloader_params)

        total_rows = len(predict_dataloader.dataset)
        num_columns = self.output_size
        all_predictions = np.empty((total_rows, num_columns))
        all_targets = np.empty((total_rows, num_columns))

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

        if return_targets:
            return all_predictions, all_targets
        else:
            return all_predictions

    def _define_checkpoint(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "p_dropouts": self.p_dropouts,
            "activation": self.activation,
            "device": self.device,
            "initialization": self.initialization,
            "initialization_kwargs": self.initialization_kwargs,
            "seed": self.seed,
            "model_name": self._model_name,
            "state_dict": self.state_dict(),
        }

    def save(
        self, 
        path: str,
        save_only_model: bool = False
    ):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name. 
            save_only_model (bool, optional): If ``True``, only the model state will be saved. If ``False``, the optimizer and scheduler states will also be saved (default: ``False``).
        """
        self.checkpoint = self._define_checkpoint()

        if not save_only_model:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.checkpoint["optimizer"] = self.optimizer.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.checkpoint["scheduler"] = self.scheduler.state_dict()
        
        if os.path.isdir(path):
            filename = "/" + str(self.mname) + ".pth"
            path = path + filename
        torch.save(self.checkpoint, path)

    @classmethod
    def load(
        cls, 
        path: str,
        device: torch.device = DEVICE,
    ):
        r"""
        Load the model from a checkpoint file. Does not require the model to be instantiated.

        Args:
            path (str): Path to the file to load the model from.
            device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).

        Returns:
            model (MLP): The loaded model with the trained weights.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        raiseWarning("The model has been loaded with weights_only set to False. According with torch documentation, this is not recommended if you do not trust the source of your saved model, as it could lead to arbitrary code execution.")
        checkpoint['device'] = device
        model = cls(
            checkpoint["input_size"],
            checkpoint["output_size"],
            checkpoint["n_layers"],
            checkpoint["hidden_size"],
            checkpoint["p_dropouts"],
            checkpoint["activation"],
            checkpoint["device"],
            checkpoint["initialization"],
            checkpoint["initialization_kwargs"],
            checkpoint["seed"],
            checkpoint["model_name"]
        )
        
        model.load_state_dict(checkpoint["state_dict"])
        model.checkpoint = checkpoint
        return model
    
    @classmethod
    @cr('MLP.create_optimized_model')
    def create_optimized_model(
        cls, 
        train_dataset: torch.utils.data.Dataset, 
        eval_dataset: torch.utils.data.Dataset, 
        optuna_optimizer: OptunaOptimizer,
        **kwargs,
    ) -> Tuple[nn.Module, Dict]:
        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.
        
        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple [MLP, Dict]: The optimized model and the optimization parameters.

        Example:
            >>> from pyLOM.NN import MLP, OptunaOptimizer
            >>> # Split the dataset
            >>> train_dataset, eval_dataset = dataset.get_splits([0.8, 0.2])
            >>> 
            >>> # Define the optimization parameters
            >>> optimization_params = {
            >>>     "lr": (0.00001, 0.01), # optimizable parameter
            >>>     "epochs": 50, # fixed parameter
            >>>     "n_layers": (1, 4),
            >>>     "batch_size": (128, 512),
            >>>     "hidden_size": (200, 400),
            >>>     "p_dropouts": (0.1, 0.5),
            >>>     "num_workers": 0,
            >>>     'print_rate_epoch': 5
            >>> }
            >>>
            >>> # Define the optimizer
            >>> optimizer = OptunaOptimizer(
            >>>     optimization_params=optimization_params,
            >>>     n_trials=5,
            >>>     direction="minimize",
            >>>     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
            >>>     save_dir=None,
            >>> )
            >>>
            >>> # Create the optimized model
            >>> model, optimization_params = MLP.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>> 
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        input_dim, output_dim = train_dataset[0][0].shape[0], train_dataset[0][1].shape[0]
        
        def suggest_value(name, space, trial):
            if isinstance(space, (tuple, list)):
                use_log = (space[1] / max(1e-12, space[0])) >= 1000 if isinstance(space[0], (int, float)) else False
                if isinstance(space[0], int):
                    return trial.suggest_int(name, int(space[0]), int(space[1]), log=use_log)
                elif isinstance(space[0], float):
                    return trial.suggest_float(name, float(space[0]), float(space[1]), log=use_log)
            else:
                return space
        
        def optimization_function(trial) -> float:
            training_params = {}       
            for key, params in optimization_params.items():
                training_params[key] = suggest_value(key, params, trial)
            training_params["save_logs_path"] = None
            
            model = cls(input_dim, output_dim, verbose=False, **training_params)
            if optuna_optimizer.pruner is not None:
                epochs = training_params["epochs"]
                training_params["epochs"] = 1
                for epoch in range(epochs):
                    model.fit(train_dataset, **training_params)
                    y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                    loss_val = ((y_pred - y_true)**2).mean()
                    trial.report(loss_val, epoch)
                    if trial.should_prune(): 
                        raise TrialPruned()
            else:
                model.fit(train_dataset, **training_params)
                y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                loss_val = ((y_pred - y_true)**2).mean()
            
            return loss_val
        
        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        # Update params with best ones
        for param in best_params.keys():
            if param in optimization_params:
                optimization_params[param] = best_params[param]
        
        return cls(input_dim, output_dim, **optimization_params), optimization_params
