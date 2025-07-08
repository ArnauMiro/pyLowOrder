#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

from typing import Dict, Tuple, Union, Any
import os
import json
import warnings
import gc

import torch
from torch import Tensor
from torch.nn import ELU
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data

from .. import Dataset as NNDataset
from .. import DEVICE, set_seed
from ..utils import count_trainable_params, cleanup_tensors, get_optimizing_value, hyperparams_serializer
from ..gns import GNSMLP, MessagePassingLayer, Graph, InputsInjector, ManualNeighborLoader, _ShapeValidator, _GNSHelpers
from ..optimizer import OptunaOptimizer, TrialPruned
from ... import pprint, cr





class GNS(nn.Module):
    r"""
    Graph Neural Solver class for predicting aerodynamic variables on RANS meshes.
    The model uses a message-passing architecture with MLPs for the message and update functions.

    Args:
        input_dim (int): The dimension of the operational parameters.
        latent_dim (int): The number of latent features.
        output_dim (int): The number of output features.
        hidden_size (int): The number of hidden units in the MLPs.
        num_msg_passing_layers (int): The number of GNN layers.
        encoder_hidden_layers (int): The number of hidden layers in the encoder.
        decoder_hidden_layers (int): The number of hidden layers in the decoder.
        message_hidden_layers (int): The number of hidden layers in the message MLP.
        update_hidden_layers (int): The number of hidden layers in the update MLP.
        graph (Union[torch_geometric.data.Data, Graph]): The graph object with node and edge features.
        p_dropouts (float, optional): The dropout probability. Default is ``0``.
        checkpoint_file (str, optional): The path to the checkpoint file. Default is ``None``.
        activation (Union[str, nn.Module]): The activation function to use.
        device (Union[str, torch.device]): The device to use for training. Default is ``'cuda'`` if available, otherwise ``'cpu'``.
        seed (int): The random seed for reproducibility. Default is None.
    """

    def __init__(self,
                graph: Graph,
                input_dim: int,
                latent_dim: int,
                output_dim: int,
                hidden_size: int,
                num_msg_passing_layers: int,
                encoder_hidden_layers: int,
                decoder_hidden_layers: int,
                message_hidden_layers: int,
                update_hidden_layers: int,
                **kwargs) -> None:
        super().__init__()

        # --- Parse kwargs ---
        self.device = torch.device(kwargs.get("device", DEVICE))
        self.seed = kwargs.get("seed", None)
        self.p_dropouts = kwargs.get("p_dropouts", 0.0)
        
        # --- Activation setup ---
        activation = kwargs.get("activation", ELU())
        if isinstance(activation, str):
            if not hasattr(nn, activation):
                raise ValueError(f"Activation function '{activation}' not found in torch.nn")
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        # --- Device setup ---
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(0)
        pprint(0, f"Using device: {self.device}", flush=True)

        # --- Seed ---
        if self.seed is not None:
            set_seed(self.seed)

        # --- Save config ---
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_msg_passing_layers = num_msg_passing_layers
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.message_hidden_layers = message_hidden_layers
        self.update_hidden_layers = update_hidden_layers

        self.model_dict = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "hidden_size": hidden_size,
            "num_msg_passing_layers": num_msg_passing_layers,
            "encoder_hidden_layers": encoder_hidden_layers,
            "decoder_hidden_layers": decoder_hidden_layers,
            "message_hidden_layers": message_hidden_layers,
            "update_hidden_layers": update_hidden_layers,
            "p_dropouts": self.p_dropouts,
            "activation": activation.__class__.__name__,
            "seed": self.seed,
        }

        # --- Graph setup ---
        self._graph = None
        self._encoder_input_dim = None
        self._edge_dim = None
        self.graph = graph  # sets _encoder_input_dim and _edge_dim

        # --- Inputs injector (prepares subgraphs for forward pass) ---
        self.injector = InputsInjector(device=self.device)

        # --- Optimizer and training state ---
        self.state = {}
        self.optimizer = None
        self.scheduler = None
        self.checkpoint = None

        # --- Validator setup ---
        self.validator = _ShapeValidator(self.input_dim, self.output_dim, self.graph.num_nodes)

        # --- Helper setup ---
        self._helpers = _GNSHelpers(
            device=self.device,
            graph=self.graph,
            num_msg_passing_layers=self.num_msg_passing_layers
        )

        # --- Build modules ---
        self._build_encoder()
        self._build_message_passing_layers()
        self._build_decoder()
        self.groupnorm = nn.GroupNorm(num_groups=min(2, self.latent_dim), num_channels=self.latent_dim).to(self.device)

    def _build_encoder(self):
        self.encoder = GNSMLP(
            input_size=self._encoder_input_dim,
            output_size=self.latent_dim,
            hidden_size=self.hidden_size,
            activation=self.activation,
            drop_p=self.p_dropouts
        ).to(self.device)

    def _build_decoder(self):
        self.decoder = GNSMLP(
            input_size=self.latent_dim,
            output_size=self.output_dim,
            hidden_size=self.hidden_size,
            activation=self.activation,
            drop_p=self.p_dropouts
        ).to(self.device)

    def _build_message_passing_layers(self):
        self.conv_layers_list = nn.ModuleList([
            MessagePassingLayer(
                in_channels=2 * self.latent_dim + self._edge_dim,
                out_channels=self.latent_dim,
                hidden_size=self.hidden_size,
                message_hidden_layers=self.message_hidden_layers,
                update_hidden_layers=self.update_hidden_layers,
                activation=self.activation,
                drop_p=self.p_dropouts,
            ) for _ in range(self.num_msg_passing_layers)
        ]).to(self.device)

    @property
    def graph(self) -> Graph:
        r"""Graph property to get the graph object."""
        return self._graph
    
    @graph.setter
    def graph(self, graph: Graph) -> None:
        r"""Graph property to set the graph object."""
        if not isinstance(graph, Graph):
            raise TypeError("Graph must be of type Graph.")
        if self._graph is not None:
            warnings.warn("Graph is already set. Overwriting.")
        
        graph.validate()
        
        if graph.device != self.device:
            graph = graph.to(self.device)


        self._encoder_input_dim = graph.x.shape[1] + self.input_dim # Update node features dimension
        self._edge_dim = graph.edge_attr.shape[1] # Update edge features dimension

        self._graph = graph

    @cr('GNS.forward')
    def forward(self, graph: Union[Data, Graph]) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            graph (Data or Graph): Prepared input graph with operational parameters embedded, useful for internal calls or external debugging.

        Returns:
            Tensor: Predicted values for all nodes in the graph.
        """
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        h = self.activation(self.encoder(x))
        for conv in self.conv_layers_list:
            h = self.groupnorm(self.activation(conv(h, edge_index, edge_attr)))
        y_hat = self.decoder(h)
        return y_hat

    @cr('GNS.predict')
    def predict(
        self,
        X: Union[Tensor, TorchDataset],
        batch_size: int = 1,
        node_batch_size: int = 256,
        **kwargs
    ) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            X (Tensor or Dataset): Operational inputs. If Tensor, shape [B, D].
            batch_size (int): Used only if X is a Dataset.
            node_batch_size (int): Number of seed nodes per subgraph batch.
        Returns:
            Tensor: Predictions of shape [B * N, F].
        """
        self._validate_shapes(X)
        self.eval()

        with torch.no_grad():
            input_dataloader = self._helpers.init_dataloader(X, batch_size)
            subgraph_loader = self._helpers.init_subgraph_loader(
                batch_size=node_batch_size,
                input_nodes=kwargs.get("input_nodes", None)
            )
            return self._run_epoch(input_dataloader, subgraph_loader, is_train=False, return_loss=False)

    @cr('GNS.fit')
    def fit(self, train_dataset:Union[TensorDataset, NNDataset], eval_dataset: Union[TensorDataset, NNDataset]=None, **kwargs) -> Dict:
        """
        Train the model using subgraph batching over both the training inputs and the node space.

        Supports evaluation on a separate dataset and configurable training parameters.

        Args:
            train_dataset: Dataset of input parameters and labels. At each iteration, it yields:
                - inputs (Tensor): Shape [B, D] where B=batch_size and D=input_dim.
                - targets (Tensor): Shape [B, N, F] where B=batch_size, N=num_nodes and F=output_dim.
            eval_dataset: Optional dataset for evaluation. Same format as `train_dataset`.
            **kwargs: Training hyperparameters such as:
                - epochs (int)
                - lr (float)
                - lr_gamma (float)
                - lr_scheduler_step (int)
                - loss_fn (callable)
                - optimizer (class)
                - scheduler (class)
                - batch_size (int)
                - node_batch_size (int)
                - input_nodes (Tensor or None)

        Returns:
            Dict: Dictionary with keys `"train_loss"` and `"test_loss"` listing per-epoch values.
        """
        # --- Validate inputs ---
        self._validate_shapes(train_dataset)
        if eval_dataset is not None:
            self._validate_shapes(eval_dataset)
        
        # --- Parse kwargs ---
        epochs = kwargs.get("epochs", 100)
        lr = kwargs.get("lr", 1e-4)
        lr_gamma = kwargs.get("lr_gamma", 0.1)
        lr_scheduler_step = kwargs.get("lr_scheduler_step", 1)
        loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss(reduction='mean'))
        optimizer = kwargs.get("optimizer", torch.optim.Adam)
        scheduler = kwargs.get("scheduler", torch.optim.lr_scheduler.StepLR)
        print_rate_epoch = kwargs.get("print_rate_epoch", 1)

        # --- Instantiate dataloaders ---        
        input_dataloader = self._helpers.init_dataloader(train_dataset, **kwargs)
        eval_dataloader = self._helpers.init_dataloader(eval_dataset, **kwargs) if eval_dataset is not None else None
        subgraph_loader = self._helpers.init_subgraph_loader(
            batch_size=kwargs.get("node_batch_size", 256),
            input_nodes=kwargs.get("input_nodes", None)
        )

        # --- Set optimizer and scheduler ---
        if self.optimizer is None:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        if self.scheduler is None:
            self.scheduler = scheduler(self.optimizer, step_size=lr_scheduler_step, gamma=lr_gamma) if scheduler is not None else None

        # --- Initialize state ---
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint["state"][0])
            if self.scheduler is not None and len(self.checkpoint["state"][1]) > 0:
                self.scheduler.load_state_dict(self.checkpoint["state"][1])
                self.scheduler.gamma = lr_gamma
                self.scheduler.step_size = lr_scheduler_step
            epoch_list = self.checkpoint["state"][2]
            train_loss_list = self.checkpoint["state"][3]
            test_loss_list = self.checkpoint["state"][4]
        else:
            epoch_list = []
            train_loss_list = []
            test_loss_list = []

        # --- Start training loop ---
        total_epochs = len(epoch_list) + epochs
        for epoch in range(1 + len(epoch_list), 1 + total_epochs):

            train_loss = self._run_epoch(input_dataloader, subgraph_loader, return_loss=True, loss_fn=loss_fn, is_train=True)
            train_loss_list.append(train_loss)

            if eval_dataloader is not None:
                test_loss = self._run_epoch(eval_dataloader, subgraph_loader, return_loss=True, loss_fn=loss_fn, is_train=False)
                test_loss_list.append(test_loss)

            if print_rate_epoch and epoch % print_rate_epoch == 0:
                test_log = f" | Eval loss:{test_loss:.4e}" if eval_dataloader else ""
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss: {train_loss:.4e}{test_log}", flush=True)
                if self.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 2
                    reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 2
                    pprint(0, f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB", flush=True)

            epoch_list.append(epoch)
            self.state = (
                self.optimizer.state_dict(),
                self.scheduler.state_dict() if self.scheduler is not None else {},
                epoch_list,
                train_loss_list,
                test_loss_list,
            )

        return {"train_loss": train_loss_list, "test_loss": test_loss_list}

    @cr('GNS._run_epoch')
    def _run_epoch(self, input_dataloader, subgraph_loader, loss_fn=None, return_loss: bool = False, is_train: bool = False) -> Union[float, Tensor]:
        """
        Core routine for both training and inference, handling subgraph batching and device transfer.

        Args:
            input_dataloader: Dataloader yielding input parameter batches.
            node_dataloader: Dataloader yielding seed node indices for subgraph batching.
            loss_fn (callable, optional): Loss function to apply if return_loss is True.
            return_loss (bool): Whether to compute and return loss instead of predictions.
            is_train (bool): Whether to run in training mode (with gradient updates).

        Returns:
            float or Tensor:
                - If return_loss: Average loss over the full dataset.
                - Else: Concatenated predictions over all batches.
        """

        self.train() if is_train else self.eval()
        outputs = []
        total_loss = 0.0
        context = torch.enable_grad() if is_train else torch.no_grad()

        last_graph = None
        last_output = None
        last_targets = None
        last_loss = None

        with context:
            num_batches = 0
            for batch in input_dataloader:
                inputs_batch = batch[0].to(self.device)  # Input parameters
                try:
                    targets_batch = batch[1].to(self.device)  # Target values
                except IndexError:
                    targets_batch = None

                for subgraph in subgraph_loader:
                    G_batch_prepared = self.injector.replicate_inject(subgraph, inputs_batch, targets_batch)

                    if is_train:
                        self.optimizer.zero_grad()

                    output = self.forward(G_batch_prepared)[G_batch_prepared.seed_mask] # We only want the seed nodes to compute the loss.

                    if return_loss:
                        targets = G_batch_prepared.y[G_batch_prepared.seed_mask]
                        loss = loss_fn(output, targets)
                        if is_train:
                            loss.backward()
                            self.optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1
                        last_graph = G_batch_prepared
                        last_output = output
                        last_targets = targets
                        last_loss = loss
                    else:
                        outputs.append(output)

            if is_train and self.scheduler is not None:
                self.scheduler.step()

        if return_loss:
            cleanup_tensors({
                "graph": last_graph,
                "output": last_output,
                "targets": last_targets,
                "loss": last_loss,
            })
            return total_loss / num_batches
        else:
            return torch.cat(outputs, dim=0)
    
    def _validate_shapes(self, X):
        try:
            self.validator.validate(X)
        except Exception as e:
            raise ValueError(f"Invalid dataset for {self.__class__.__name__}: {e}") from e


    def save(self, path: str) -> None:
        """
        Save the model to a checkpoint file.

        If the given path is a directory, appends a filename with the number of trained epochs.

        Args:
            path (str): Directory or full file path to save the model.
        """
        checkpoint = {
            **self.model_dict,
            "state_dict": self.state_dict(),
            "state": self.state,
            "graph": self.graph,
        }

        if os.path.isdir(path):
            filename = f"/trained_model_{len(self.state[2]):06d}.pth"
            path = os.path.join(path, filename)

        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = DEVICE) -> "GNS":
        """
        Load a model from a checkpoint file.

        Args:
            path (str): Path to the `.pth` checkpoint file.
            device (Union[str, torch.device]): Target device for model (default: inferred).

        Returns:
            GNS: Loaded model instance.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file '{path}' not found.")
        if not path.endswith(".pth"):
            raise ValueError("Model file must have a '.pth' extension.")

        device = torch.device(device or DEVICE)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. Use CPU instead.")
            torch.cuda.set_device(0)

        checkpoint = torch.load(path, map_location='cpu')

        required_keys = ["graph", "state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing key '{key}' in checkpoint.")
        
        model = cls(
            graph=checkpoint["graph"],
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_size=checkpoint["hidden_size"],
            num_msg_passing_layers=checkpoint["num_msg_passing_layers"],
            encoder_hidden_layers=checkpoint["encoder_hidden_layers"],
            decoder_hidden_layers=checkpoint["decoder_hidden_layers"],
            message_hidden_layers=checkpoint["message_hidden_layers"],
            update_hidden_layers=checkpoint["update_hidden_layers"],
            p_dropouts=checkpoint.get("p_dropouts"),
            activation=checkpoint.get("activation"),
            seed=checkpoint.get("seed"),
            device=device,
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.state = checkpoint["state"]
        model.eval()

        return model

    @classmethod
    def create_optimized_model(
        cls,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        optuna_optimizer: OptunaOptimizer,
    ) -> Tuple["GNS", Dict]:

        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.
        
        Args:
            train_dataset (Dataset): The training dataset.
            eval_dataset (Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The Optuna optimizer object.
        Returns:
            Tuple[nn.Module, Dict]: The optimized model and the optimization parameters.

        Example:
            >>> from pyLOM.NN import GNS, OptunaOptimizer
            >>> # Split the dataset
            >>> train_dataset, eval_dataset = dataset.get_splits([0.8, 0.2])
            >>> # Create the graph
            >>> graph = Graph.from_mesh(mesh, x=x, y=y, scaler=scaler)
            >>> 
            >>> # Define the optimization parameters
            >>> optimization_params = {
            >>>     "graph": graph,
            >>>     "input_dim": 2,
            >>>     "latent_dim": 16,
            >>>     "output_dim": 1,
            >>>     "hidden_size": (64, 512),
            >>>     "num_msg_passing_layers": (1, 10),
            >>>     "encoder_hidden_layers": (1, 10),
            >>>     "decoder_hidden_layers": (1, 10),
            >>>     "message_hidden_layers": (1, 10),
            >>>     "update_hidden_layers": (1, 10),
            >>>     "activation": "ELU",
            >>>     "epochs": 1000,
            >>>     "lr": (1e-5, 1e-2),
            >>>     "lr_gamma": (0.1, 1),
            >>>     "lr_scheduler_step": (1, 10),
            >>>     "loss_fn": nn.MSELoss(reduction='mean'),
            >>>     "optimizer": torch.optim.Adam,
            >>>     "scheduler": torch.optim.lr_scheduler.StepLR,
            >>>     "batch_size": (1, 16),
            >>>     "node_batch_size": (2**3, 2**16),
            >>> }
            >>>
            >>> # Define the optimizer
            >>> optimizer = OptunaOptimizer(
            >>>     optimization_params=optimization_params,
            >>>     n_trials=100,
            >>>     direction="minimize",
            >>>     pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=5),
            >>>     save_dir=None,
            >>> )
            >>>
            >>> # Create the optimized model
            >>> model, optimization_params = GNS.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>> 
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        seed_base = optimization_params.get("seed", 42)

        MODEL_KWARGS = {
            'graph', 'input_dim', 'latent_dim', 'output_dim',
            'hidden_size', 'num_msg_passing_layers',
            'encoder_hidden_layers', 'decoder_hidden_layers',
            'message_hidden_layers', 'update_hidden_layers',
            'seed', 'activation', 'p_dropouts', 'device'
        }
        TRAINING_KWARGS = {
            'batch_size', 'node_batch_size', 'num_workers', 'pin_memory',
            'epochs', 'lr', 'lr_gamma', 'lr_scheduler_step',
            'loss_fn', 'optimizer', 'scheduler',
            'print_rate_epoch', 'print_rate_batch'
        }
        
        @cr('GNS.optimization_function')
        def optimization_function(trial) -> float:
            hyperparams = {
                key: get_optimizing_value(key, val, trial)
                for key, val in optimization_params.items()
            }

            model_kwargs = {k: v for k, v in hyperparams.items() if k in MODEL_KWARGS}
            fit_kwargs = {k: v for k, v in hyperparams.items() if k in TRAINING_KWARGS}

            model = cls(
                **model_kwargs
            )

            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}. Training with hyperparams:\n",
                json.dumps(hyperparams, indent=4, default=hyperparams_serializer), flush=True)

            try:
                if optuna_optimizer.pruner is not None:
                    original_epochs = fit_kwargs.get("epochs", 100)
                    fit_kwargs_epoch = dict(fit_kwargs)
                    fit_kwargs_epoch.pop("epochs", None)

                    for epoch in range(original_epochs):
                        losses = model.fit(train_dataset, eval_dataset, epochs=1, **fit_kwargs_epoch)
                        loss_val = losses["test_loss"][-1]
                        trial.report(loss_val, epoch)
                        pprint(0, f"Epoch {epoch + 1}/{original_epochs}", flush=True)
                        if trial.should_prune():
                            pprint(0, f"Trial pruned at epoch {epoch + 1}. Calling torch.cuda.empty_cache()", flush=True)
                            torch.cuda.empty_cache()  # Clear GPU memory before pruning
                            pprint(0, f"calling del model", flush=True)
                            del model
                            pprint(0, f"\nTrial pruned at epoch {epoch + 1}", flush=True)
                            raise TrialPruned()
                else:
                    losses = model.fit(train_dataset, eval_dataset, **fit_kwargs)
                    loss_val = losses["test_loss"][-1]
                    trial.report(loss_val, step=9999)


            except RuntimeError as e:
                pprint(0, f"RuntimeError during trial {trial.number + 1}: {e}. Calling torch.cuda.empty_cache()", flush=True)
                torch.cuda.empty_cache()  # Clear GPU memory before raising
                pprint(0, f"calling del model", flush=True)
                del model
                raise e

            pprint(0, f"\nTrial {trial.number + 1} completed. Calling torch.cuda.empty_cache()", flush=True)
            torch.cuda.empty_cache()  # Clear GPU memory after each trial
            pprint(0, f"calling del model", flush=True)
            del model
            return loss_val

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)
        pprint(0, f"\nBest hyperparameters found: {json.dumps(best_params, indent=4, default=hyperparams_serializer)}\n", flush=True)
        optimization_params.update(best_params)

        final_model = cls(**{
            k: v for k, v in optimization_params.items() if k in MODEL_KWARGS
        })

        return final_model, optimization_params

    def __repr__(self):
        return (
            f"<GNSModel: {self.input_dim} → {self.latent_dim} → {self.output_dim}>\n"
            f" Layers: encoder({self.encoder_hidden_layers}), message({self.num_msg_passing_layers}), decoder({self.decoder_hidden_layers})\n"
            f" MLPs: message({self.message_hidden_layers}), update({self.update_hidden_layers})\n"
            f" Activation: {self.activation.__class__.__name__}, Dropout: {self.p_dropouts}, Device: {self.device}\n"
            f" Graph: {repr(self.graph)}\n"
            f" Params: {count_trainable_params(self):,} trainable\n>"
        )
