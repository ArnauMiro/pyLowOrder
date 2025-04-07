#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.nn import ELU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import k_hop_subgraph


from pyLOM import Mesh
from pyLOM.vmmath.geometric import edge_to_cells, wall_normals
from pyLOM.NN.optimizer import OptunaOptimizer, TrialPruned
from pyLOM.NN import DEVICE, set_seed  # pyLOM/NN/__init__.py
from pyLOM import pprint, cr  # pyLOM/__init__.py

from typing import Protocol, Optional, Dict, Tuple, Union

class MLP(torch.nn.Module):
    '''Simple MLP with dropout and activation function
    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        hidden_sizes (list): List of hidden sizes.
        drop_p (float): Dropout probability.
        activation (torch.nn.Module): Activation function.
    '''
    def __init__(self, input_size, output_size, hidden_sizes, drop_p=0.5, activation=ELU()):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        self.activation = activation
        self.dropout = torch.nn.Dropout(p=drop_p)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MessagePassingLayer(MessagePassing):
    '''Message passing layer for the GNN. Uses mean aggregation and MLPs for message and update functions.
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        hiddim (int): Hidden dimension.
        drop_p (float): Dropout probability.
    '''
    def __init__(self, in_channels, out_channels, hiddim, activation=ELU(), drop_p=0.):
        # Message passing with "mean" aggregation.
        super().__init__(aggr='mean')
        self.dropout = torch.nn.Dropout(p=drop_p)

        # MLP for the message function
        self.phi = MLP(in_channels, out_channels, 1*[hiddim], drop_p=0, activation=activation)
        
        # MLP for the update function
        self.gamma = MLP(2*out_channels, out_channels, 1*[hiddim], drop_p=0, activation=activation)


    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i defines the features of central nodes as shape [num_edges, in_channels-6]
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels-6]
        # edge_attr defines the attributes of intersecting edges as shape [num_edges, 6]

        input = torch.cat([x_i, x_j, edge_attr], dim=1)

        return self.phi(input)  # Apply MLP phi
    
    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]

        input = torch.cat([x, aggr_out], dim=1)

        # Apply MLP gamma
        return self.gamma(input)
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)






class ScalerProtocol(Protocol):
    '''
    Abstract protocol for scalers. Must include:
        - fit: Fit the scaler to the data.
        - transform: Transform the data using the fitted scaler.
        - fit_transform: Fit the scaler to the data and transform it.
    '''
    def fit(self, X: np.ndarray, y=None) -> "ScalerProtocol":
        """ Ajusta el escalador a los datos. """
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Transforma los datos utilizando el escalador ajustado. """
        ...

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """ Ajusta el escalador a los datos y los transforma. """
        ...


class pyLOMGraph(Data):
    '''
    Custom class derived from torch.geometric.Data to handle graphs for GNN.
    
    Custom features:
        - from_pyLOM_mesh: Create a torch_geometric Data object from a pyLOM Mesh object.
        - filter: Filter graph by eliminating nodes not in node_mask.
    '''

    @classmethod
    def from_pyLOM_mesh(cls,
                        mesh: Mesh,
                        y: Optional[np.ndarray] = None,
                        scaler: Optional[ScalerProtocol] = None,
                        name: Optional[str] = None,
                        device: Optional[Union[str, torch.device]] = DEVICE,
                                ) -> "pyLOMGraph":
        r"""
        Create a torch_geometric Data object from a pyLOM Mesh object.

        Args:
            mesh (pyLOM.Mesh): The input mesh.
            y (Optional[np.ndarray]): Optional node target values. Must have dimension (n_nodes, :).
            scaler (Optional[ScalerProtocol]): Optional scaler to normalize node and edge features. Must include:
                - fit: Fit the scaler to the data.
                - transform: Transform the data using the fitted scaler.
                -   fit_transform: Fit the scaler to the data and transform it.
            operational_parameters_size (int): The number of operational parameters (e.g.: 2 for Mach and alpha)

        Returns:
            pyLOMGraph: The graph structure.
        """
        xyzc = mesh.xyzc  # Cell centers coordinates
        print("Computing surface normals")
        surface_normals = mesh.normal
        print("Surface normals computed")

        print("Computing dual edges and wall normals")
        edge_index, wall_normals = cls._dual_edges_and_wall_normals(mesh)

        # Create the edge features
        c_i = xyzc[edge_index[0, :]]
        c_j = xyzc[edge_index[1, :]]
        d_ij = c_j - c_i
        # Transform to spherical coordinates
        r = np.linalg.norm(d_ij, axis=1)
        theta = np.arccos(d_ij[:, 2] / r)
        phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])
        edge_attr = np.concatenate((r[:, None], theta[:, None], phi[:, None], wall_normals), axis=1)  # Ensure correct shape

        # Scale node and edge features if scaler is provided
        if scaler is not None:
            xyzc = scaler.fit_transform(xyzc)
            surface_normals = scaler.fit_transform(surface_normals)
            edge_attr = scaler.fit_transform(edge_attr)

        y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # Return the class instance with the necessary attributes. Leave x as None as the final edge attributes need to be computed dynamically during training.
        graph = cls(
            x=None, # To be completed with the operational parameters during training
            pos=torch.tensor(xyzc, dtype=torch.float32),
            surf_norms = torch.tensor(surface_normals, dtype=torch.float32),
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr
            )

        # Set the graph name
        if name is not None:
            graph.name = name
        
        # Set the device
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            graph.to(device)

        return graph
    

    def filter(self,
        node_mask: Optional[Union[list, torch.Tensor, np.ndarray]]=None,
        node_indices: Optional[Union[list, torch.Tensor, np.ndarray]]=None
    ):
        '''
        Filter graph by providing either a boolean mask or a list of node indices to keep.

        Args:
            node_mask: Boolean mask to filter nodes.
            node_indices: List of node indices to keep.
        '''
        

        if node_mask is None and node_indices is None:
            raise ValueError("Either node_mask or node_indices must be provided.")
        elif node_mask is not None and node_indices is not None:
            raise ValueError("Only one of node_mask or node_indices must be provided.")
        elif node_mask is not None:
            node_mask = torch.tensor(node_mask, dtype=torch.bool)
        elif node_indices is not None:
            node_mask = torch.zeros(self.x.shape[0], dtype=torch.bool)
            node_mask[node_indices] = True

        for attr in self.node_attrs():
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[node_mask])

        self.y = self.y[node_mask] if self.y is not None else None


        self.edge_attr = self.edge_attr[torch.logical_and(node_mask[self.edge_index[0]], node_mask[self.edge_index[1]])]
        self.edge_index = self.edge_index[:, torch.logical_and(node_mask[self.edge_index[0]], node_mask[self.edge_index[1]])]
        self.edge_index -= torch.min(self.edge_index)
     

    
    @staticmethod
    def _dual_edges_and_wall_normals(mesh):
        '''Computes the directed edges in the dual graph and the unitary wall normals of each cell (only for 2D cells) in the way its needed to build an atributed graph
        as described in
            Hines, D., & Bekemeyer, P. (2023). Graph neural networks for the prediction of aircraft surface pressure distributions.
            Aerospace Science and Technology, 137, 108268.
            https://doi.org/10.1016/j.ast.2023.108268
        
        Edges in the dual graph connect cells in the primal graph and are given as pairs of cell indices.
        Wall unitary normals are orthogonal to the cell walls and point outwards. They are contained in the cell plane, so are orthogonal themselves to the cell normal.
        The order in which dual edges and wall normals are saved is coherent, meaning that if the i-th dual edge is (a, b), then the i-th wall normal is ortogonal to the wall shared by cells a and b and points outwards of cell a. 
        As a convention the dual graph is bidirectional, so if the i-th dual edge is (a, b), then there is some edge (b, a) and the wall normal at that position is equal to - the wall normal at the i-th position.

        As boundary walls lack a corresponging edge in the dual graph, their wall normals are not saved as a convention.
        '''

        # Check whether the cells are 2D
        if not np.all(np.isin(mesh.eltype, [2, 3, 4, 5])):
            raise ValueError("The mesh must contain only 2D cells in order to compute the wall normals.")
        

        # Dictionary that maps each edge to the cells that share it
        edge_dict = edge_to_cells(mesh.connectivity)
        # List storying directed edges in the dual graph
        dual_edges_list = []
        # List to store the wall normals.
        wall_normals_list = []

        # Iterate over each cell
        for i, cell_id in enumerate(range(mesh.ncells)):
            cell_normal = mesh.normal[cell_id]
            cell_nodes = mesh.connectivity[cell_id]
            nodes_xyz = mesh.xyz[cell_nodes]  # Get the nodes of the cell

            cell_edges, cell_wall_normals = wall_normals(cell_nodes, nodes_xyz, cell_normal)  # Compute the edge normals of the cell
            
            # Directed dual edges: tuples of the form (cell_id, neighbor_id)
            dual_edges = [
                (cell_id, (edge_dict[edge] - {cell_id}).pop()) if len(edge_dict[edge]) == 2 else None # If the edge is not a boundary edge, get the neighbor cell
                for edge in cell_edges
            ]

            dual_edges_list.extend(dual_edges)
            wall_normals_list.extend(cell_wall_normals)

            if i%1e5 == 0:
                print(f"Processing mesh. {i} cells out of {mesh.ncells} processed.")

        # Remove the wall normals and dual edges at the boundary walls
        dual_edges_list, wall_normals_list = zip(*[
                (x, y) for x, y in zip(dual_edges_list, wall_normals_list) if x is not None
            ])

        return np.array(dual_edges_list, dtype=np.int32).T, np.array(wall_normals_list, dtype=np.float64)
    





class GNS(nn.Module):
    """
    Graph Neural Solver class for predicting aerodynamic variables on RANS meshes.
    The model uses a message-passing architecture with MLPs for the message and update functions.

    Args:
        input_dim (int): The dimension of the operational parameters.
        latent_dim (int): The number of latent features.
        output_dim (int): The number of output features.
        hidden_size (int): The number of hidden units in the MLPs.
        num_gnn_layers (int): The number of GNN layers.
        encoder_hidden_layers (int): The number of hidden layers in the encoder.
        decoder_hidden_layers (int): The number of hidden layers in the decoder.
        message_hidden_layers (int): The number of hidden layers in the message MLP.
        update_hidden_layers (int): The number of hidden layers in the update MLP.
        # graph (Union[Data, pyLOMGraph]: The pyLOMGraph object used to train the GNN..
        p_dropouts (float, optional): The dropout probability. Default is ``0``.
        checkpoint_file (str, optional): The path to the checkpoint file. Default is ``None``.
        activation (Union[str, nn.Module]): The activation function to use.
        device (Union[str, torch.device]): The device to use for training. Default is ``'cuda'`` if available, otherwise ``'cpu'``.
        seed (int): The random seed for reproducibility. Default is None.
    """

    def __init__(self,
                 graph: Union[Data, pyLOMGraph],
                 input_dim: int,
                 latent_dim: int,
                 output_dim: int,
                 hidden_size: int,
                 num__gnn_layers: int,
                 encoder_hidden_layers: int,
                 decoder_hidden_layers: int,
                 message_hidden_layers: int,
                 update_hidden_layers: int,
                 **kwargs
                 ):
        
        super().__init__()

        p_dropouts = kwargs.get("p_dropouts")
        activation = kwargs.get("activation")
        device = kwargs.get("device")
        seed = kwargs.get("seed")
        if p_dropouts is None:
            p_dropouts = 0.0
        if activation is None:
            activation = 'ELU'
        if device is None:
            device = DEVICE
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(0)
        if seed is not None:
            set_seed(seed)

        # Save the model parameters
        self._graph = None  # Placeholder for the graph object.
        self._edge_dim = None # To be determined from self.graph
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.encoder_input_dim = None # To be determined from self.graph
        self.hidden_size = hidden_size
        self.num__gnn_layers = num__gnn_layers
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.message_hidden_layers = message_hidden_layers
        self.update_hidden_layers = update_hidden_layers
        self.activation = None
        self.p_dropouts = p_dropouts
        self.device = device
        self.seed = seed
        self.state = {} # Save the state of the optimizer, scheduler and epoch list
        self.checkpoint = None # Placeholder for the checkpoint object.

        self.graph = graph  # Set the graph object

        # Activation function
        if isinstance(activation, str):
            if hasattr(nn, activation):  
                self.activation = getattr(nn, activation)()
            else:
                raise ValueError(f"Activation function {activation} not found in torch.nn")
        else:
            self.activation = activation

        # Save a dictionary with the model parameters
        self.model_dict = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "hidden_size": hidden_size,
            "num__gnn_layers": num__gnn_layers,
            "encoder_hidden_layers": encoder_hidden_layers,
            "decoder_hidden_layers": decoder_hidden_layers,
            "message_hidden_layers": message_hidden_layers,
            "update_hidden_layers": update_hidden_layers,
            "p_dropouts": p_dropouts,
            "activation": activation,
            "device": device,
            "seed": seed
        }

        # Encoder: from graph node features to latent space
        self.encoder = MLP(
            input_size=self.encoder_input_dim,
            output_size=self.latent_dim,
            hidden_sizes=[self.hidden_size] * self.encoder_hidden_layers,
            activation=self.activation,
            drop_p=self.p_dropouts
        )

        # Decoder: from latent space to output features
        self.decoder = MLP(
            input_size=self.latent_dim,
            output_size=self.output_dim,
            hidden_sizes=[self.hidden_size] * self.decoder_hidden_layers,
            activation=self.activation,
            drop_p=self.p_dropouts
        )

        # Message-passing layers
        self.conv_layers_list = nn.ModuleList([
            MessagePassingLayer(
                in_channels=2 * self.latent_dim + self._edge_dim,
                out_channels=self.latent_dim,
                hiddim=self.hidden_size,
                activation=self.activation,
                drop_p=self.p_dropouts,
            )
            for _ in range(self.num_gnn_layers)
        ])

        # Normalization layer
        self.groupnorm = nn.GroupNorm(2, self.latent_dim)

    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, graph):
        """
        Forward pass of the model.

        Args:
            graph (Data): The input graph.

        Returns:
            torch.Tensor: The predicted target values.
        """

        # Get node and edge features
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        # 1. Encode node features
        h = self.encoder(x)
        h = self.activation(h)

        # 2. Message-passing layers
        for conv in self.conv_layers_list:
            h = conv(h, edge_index, edge_attr)
            h = self.activation(h)
            h = self.groupnorm(h)

        # 3. Decode node features
        y_hat = self.decoder(h)

        return y_hat
    

    @property
    def graph(self):
        '''Graph property to get the graph object.'''
        return self._graph
    
    @graph.setter
    def graph(self, graph: pyLOMGraph):
        '''Graph property to set the graph object.'''
        if self._graph is not None:
            raise Warning("Graph is already set! Graph name is: {}".format(self._graph.name))
        if not isinstance(graph, pyLOMGraph):
            if not isinstance(graph, Data):
                raise TypeError("Graph must be of type torch_geometric.data.Data or pyLOMGraph.")
        if not graph.is_undirected():
            raise Warning("Graph is not undirected. This may lead to unexpected results.")
        if getattr(graph, "edge_index", None) is None:
            raise ValueError("Graph must have edge_index attribute.")
        if getattr(graph, "edge_attr", None) is None:
            raise ValueError("Graph must have edge_attr attribute.")
        if getattr(graph, "pos", None) is None:
            raise ValueError("Graph must have pos attribute.")
        if getattr(graph, "surf_norms", None) is None:
            raise ValueError("Graph must have surf_norms attribute.")
        
        
        graph = graph.to(self.device)

        # Precompute the graph node features. Concatenate all node attrs in a single tensor with preallocated memory for the operational parameters
        graph.x = torch.cat(
            [
                torch.zeros((graph.num_nodes, self.input_dim), dtype=torch.float32, device=self.device),
                *[getattr(graph, attr) for attr in graph.node_attrs()]
            ],
            dim=1
        )

        self.encoder_input_dim = graph.x.shape[1] # Update node features dimension
        self._edge_dim = graph.edge_attr.shape[1] # Update edge features dimension

        self._graph = graph



    @cr('GNS._train')
    def _train(self, op_dataloader, node_dataloader, loss_fn):
        '''Train for 1 epoch. Used in the fit method inside a loop with ``epochs`` iterations.
        Args:
            op_dataloader (DataLoader): The DataLoader for the operational parameters. Should contain also target values.
            node_dataloader (DataLoader): The DataLoader for the nodes.
            loss_fn (torch.nn.Module): The loss function to use.
        Returns:
            float: The average loss for the epoch.
        '''

        # Set the model to training mode
        if self.seed is not None:
            set_seed(self.seed)

        self.train()
        total_loss = 0

        for params_batch, y_batch in op_dataloader:
            params_batch = params_batch.to(self.device) # [B, 3]
            y_batch = y_batch.to(self.device) # [B, N]
            
            for seed_nodes in node_dataloader:
                # Compute the k-hop subgraph
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(seed_nodes, num_hops=1, edge_index=self.graph.edge_index, relabel_nodes=True)

                # Complete the subgraph data and append it to a subgraphs batch
                x = self.graph.x[subset]
                y_batch = y_batch[:, subset]
                # Create a boolean mask for the seed nodes (original tensor is not useful bc of the relabeling)
                seed_nodes = torch.zeros(subset.shape[0], dtype=torch.bool, device=self.device)
                seed_nodes[mapping] = True # Store this mapping as we only care about the seed nodes for the loss
                G_list = []
                for p, y in zip(params_batch, y_batch):
                    x[:, :self.op_dim] = p # Prepend the operational parameters to node features
                    y = y.reshape(-1,1)
                    G = Data(x=x.clone(), y=y.clone(), seed_nodes=seed_nodes, edge_index=edge_index, edge_attr=self.graph.edge_attr[edge_mask])
                    G_list.append(G)

                G_batch = Batch.from_data_list(G_list)
                G_batch = G_batch.to(self.device)
                
                self.optimizer.zero_grad()
                # Forward pass: only look at the seed nodes for the loss
                output = self(G_batch)[G_batch.seed_nodes]
                targets = G_batch.y[G_batch.seed_nodes]
                loss = loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

        return total_loss / (len(self.op_dataloader) * len(self.node_dataloader))

    @cr('GNS._eval')
    def _eval(self, eval_dataloader, loss_fn):
        '''Evaluate the model on a validation set. Used in the fit method inside a loop with ``epochs`` iterations.
        Args:
            eval_dataloader (DataLoader): The DataLoader for the evaluation set.
            loss_fn (torch.nn.Module): The loss function to use.
        Returns:
            float: The average loss for the evaluation set.
        '''
        
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for params_batch, y_batch in eval_dataloader:
                params_batch = params_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                for (p, y) in zip(params_batch, y_batch):
                    self.graph.x[:, :self.op_dim] = p
                    targets = y.reshape(-1, self.output_dim)
                    output = self(self.graph)
                    loss = loss_fn(output, targets)
                    total_loss += loss.item()

        return total_loss / eval_dataloader.dataset.__len__()




    @cr('GNS.fit')
    def fit(self,
            train_dataset,
            eval_dataset=None,
            **kwargs
            ):
        """
        Fit the model to the training data.

        Args:
            train_dataset (Dataset): The training dataset.
            eval_dataset (Dataset, optional): The evaluation dataset. Default is None.
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                - epochs (int): Number of epochs to train the model. Default is 100.
                - lr (float): Learning rate. Default is 1e-4.
                - lr_gamma (float): Learning rate decay factor. Default is 0.1.
                - lr_scheduler_step (int): Learning rate scheduler step size. Default is 1.
                - loss_fn (torch.nn.Module): Loss function. Default is nn.MSELoss(reduction='mean').
                - optimizer (torch.optim.Optimizer): Optimizer class. Default is torch.optim.Adam.
                - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler class. Default is torch.optim.lr_scheduler.StepLR.
                - print_rate_batch (int): Print rate for batch loss. Default is 0.
                - print_rate_epoch (int): Print rate for epoch loss. Default is 1.

        Returns:
            Dict[str, List[float]]: A dictionary with the training and validation losses for each epoch.
        """

        epochs = kwargs.get("epochs", 100)
        lr = kwargs.get("lr", 1e-4)
        lr_gamma = kwargs.get("lr_gamma", 0.1)
        lr_scheduler_step = kwargs.get("lr_scheduler_step", 1)
        loss_fn = kwargs.get("loss_fn", nn.MSELoss(reduction='mean'))
        optimizer = kwargs.get("optimizer", torch.optim.Adam)
        scheduler = kwargs.get("scheduler", torch.optim.lr_scheduler.StepLR)
        print_rate_batch = kwargs.get("print_rate_batch", 0)
        print_rate_epoch = kwargs.get("print_rate_epoch", 1)


        op_dataloader_params = {
            "batch_size": kwargs.get("batch_size", 15),
            "shuffle": kwargs.get("shuffle", True),
            "num_workers": kwargs.get("num_workers", 0),
            "pin_memory": kwargs.get("pin_memory", True)
        }

        node_dataloader_params = {
            "batch_size": kwargs.get("node_batch_size", 256),
            "shuffle": kwargs.get("shuffle", True),
            "num_workers": kwargs.get("num_workers", 0),
            "pin_memory": kwargs.get("pin_memory", True)
        }

        # Create the DataLoader for the operational parameters
        if not hasattr(self, "op_dataloader"):
            op_dataloader = DataLoader(train_dataset, **op_dataloader_params)
        # Create the DataLoader for the nodes
        if not hasattr(self, "node_dataloader"):
            node_indices = np.arange(self.graph.num_nodes)
            node_indices = torch.tensor(node_indices, dtype=torch.long)
            node_dataloader = DataLoader(node_indices, **node_dataloader_params)

        eval_dataloader = DataLoader(eval_dataset, **op_dataloader_params) if eval_dataset is not None else None

        if not hasattr(self, "optimizer"):
            self.optimizer = optimizer(self.parameters(), lr=lr)
        if not hasattr(self, "scheduler"):
            self.scheduler = scheduler(self.optimizer, step_size=lr_scheduler_step, gamma=lr_gamma) if scheduler is not None else None

        if hasattr(self, "checkpoint"):
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



        total_epochs = len(epoch_list) + epochs
        for epoch in range(1+len(epoch_list), 1+total_epochs):
            
            train_loss = self._train(op_dataloader, node_dataloader, loss_fn)
            train_loss_list.append(train_loss)
            if print_rate_batch != 0 and (epoch % print_rate_batch) == 0:
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss (x1e5) {train_loss * 1e5:.4f}", flush=True)
            
            test_loss = 0.0
            if eval_dataloader is not None:
                test_loss = self._eval(eval_dataloader)
                test_loss_list.append(test_loss)
            
            if print_rate_epoch != 0 and (epoch % print_rate_epoch) == 0:
                test_log = f" | Test loss (x1e5) {test_loss * 1e5:.4f}" if eval_dataloader is not None else ""
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss (x1e5) {train_loss * 1e5:.4f} {test_log}", flush=True)

            epoch_list.append(epoch)
            self.state = (
                self.optimizer.state_dict(),
                self.scheduler.state_dict() if self.scheduler is not None else {},
                epoch_list,
                train_loss_list,
                test_loss_list,
                )
            
        return {"train_loss": train_loss_list, "test_loss": test_loss_list}

        

    def predict(self, 
        X, 
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded to a DataLoader with the provided keyword arguments. 
        The model is set to evaluation mode and the predictions are made using the input data. 
        To make a prediction from a torch tensor, use the `__call__` method directly.

        Args:
            X: The dataset whose target values are to be predicted using the input data.
            rescale_output (bool): Whether to rescale the output with the scaler of the dataset (default: ``True``).
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                - batch_size (int, optional): Batch size (default: ``256``).
                - shuffle (bool, optional): Shuffle the data (default: ``False``).
                - num_workers (int, optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, optional): Pin memory (default: ``True``).
 
        Returns:
            np.ndarray: The predicted target values if return_targets is False.
            Tuple[np.ndarray, np.ndarray]: The predicted target values and the target values from the dataset if return_targets is True.
        """
        dataloader_params = {
            "batch_size": 15,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
        }
        
        for key in dataloader_params.keys():
            if key in kwargs:
                dataloader_params[key] = kwargs[key]

        predict_dataloader = DataLoader(X, **dataloader_params)

        num_rows = len(predict_dataloader.dataset)
        num_columns = self.output_dim
        all_predictions = np.empty((num_rows, num_columns))
        all_targets = np.empty((num_rows, num_columns))


        with torch.no_grad():
            self.eval()
            for params_batch, y_batch in predict_dataloader:
                params_batch = params_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                for i, (p, y) in enumerate(zip(params_batch, y_batch)):
                    self.graph.x[:, :self.op_dim] = p
                    targets = y.reshape(-1, self.output_dim)
                    output = self(self.graph)
                    all_predictions[i] = output.cpu().numpy()
                    all_targets[i] = targets.cpu().numpy()    

        if return_targets:
            return all_predictions, all_targets
        else:
            return all_predictions


    def save(self, path: str):
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.
        """
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name. 
            If it is a directory, the model will be saved with a filename that includes the number of epochs trained.
        """
        checkpoint = {
            **self.model_dict,
            "state_dict": self.state_dict(),
            "state": self.state,
            'graph': self.graph,
        }
        
        if os.path.isdir(path):
            filename = "/trained_model_{:06d}".format(len(self.state[2])) + ".pth"
            path = path + filename
        torch.save(checkpoint, path)

    @classmethod
    def load(cls,
             path: str,
             device: Union[str, torch.device] = DEVICE,
             ):
        """
        Load a model from a file.

        Args:
            path (str): The path to load the model from.

        Returns:
            Model (GNS): The loaded model.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        if not path.endswith(".pth"):
            raise ValueError(f"Model file {path} must be a .pth file.")
        if device is None:
            device = DEVICE
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(0)
        
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            graph=checkpoint["graph"],
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_size=checkpoint["hidden_size"],
            num_gnn_layers=checkpoint["num_gnn_layers"],
            encoder_hidden_layers=checkpoint["encoder_hidden_layers"],
            decoder_hidden_layers=checkpoint["decoder_hidden_layers"],
            message_hidden_layers=checkpoint["message_hidden_layers"],
            update_hidden_layers=checkpoint["update_hidden_layers"],
            p_dropouts=checkpoint.get("p_dropouts"),
            activation=checkpoint.get("activation"),
            seed=checkpoint.get("seed"),
            device=checkpoint.get("device"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.state = checkpoint["state"]
        model.graph = checkpoint["graph"]
        model.to(model.device)
        model.eval()
        
        return model
    

    @classmethod
    @cr('MLP.create_optimized_model')
    def create_optimized_model(
        cls,
        # graph: Union[Data, pyLOMGraph], 
        train_dataset, 
        eval_dataset, 
        optuna_optimizer: OptunaOptimizer,
    ) -> Tuple[nn.Module, Dict]:
        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.
        
        Args:
            graph (Graph): The graph object used to train the GNN.
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
            >>>     "num__gnn_layers": (1, 10),
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
            >>> model, optimization_params = MLP.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>> 
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        def optimization_function(trial) -> float:
            hyperparams = {}       
            for key, params in optimization_params.items():
                hyperparams[key] = cls._get_optimizing_value(key, params, trial)

            
            model = cls(
                graph=hyperparams["graph"],
                input_dim=hyperparams["input_dim"],
                latent_dim=hyperparams["latent_dim"],
                output_dim=hyperparams["output_dim"],
                hidden_size=hyperparams["hidden_size"],
                num__gnn_layers=hyperparams["num__gnn_layers"],
                encoder_hidden_layers=hyperparams["encoder_hidden_layers"],
                decoder_hidden_layers=hyperparams["decoder_hidden_layers"],
                message_hidden_layers=hyperparams["message_hidden_layers"],
                update_hidden_layers=hyperparams["update_hidden_layers"],
                **hyperparams
            )
            if optuna_optimizer.pruner is not None:
                # prune epoch-wise
                epochs = hyperparams["epochs"]
                hyperparams["epochs"] = 1
                for epoch in range(epochs):
                    losses = model.fit(train_dataset, eval_dataset, **hyperparams)
                    loss_val = losses["test_loss"][-1]
                    # Report the loss to Optuna
                    trial.report(loss_val, epoch)
                    if trial.should_prune(): 
                        raise TrialPruned()
            else:
                losses = model.fit(
                    train_dataset,
                    epochs = hyperparams.get("epochs"),
                    lr = hyperparams.get("lr"),
                    lr_gamma = hyperparams.get("lr_gamma"),
                    lr_scheduler_step = hyperparams.get("lr_scheduler_step"),
                    loss_fn = hyperparams.get("loss_fn"),
                    optimizer = hyperparams.get("optimizer"),
                    scheduler = hyperparams.get("scheduler"),
                    print_rate_batch = hyperparams.get("print_rate_batch"),
                    print_rate_epoch = hyperparams.get("print_rate_epoch"),
                    **hyperparams
                    )
                loss_val = losses["test_loss"][-1]
                # Report the loss to Optuna
                trial.report(loss_val, 0)
            
            return loss_val
        
        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        # Update params with best ones
        for param in best_params.keys():
            if param in optimization_params:
                optimization_params[param] = best_params[param]
        
        return cls(
            graph=optimization_params["graph"],
            input_dim=optimization_params["input_dim"],
            latent_dim=optimization_params["latent_dim"],
            output_dim=optimization_params["output_dim"],
            hidden_size=optimization_params["hidden_size"],
            num_gnn_layers=optimization_params["num__gnn_layers"],
            encoder_hidden_layers=optimization_params["encoder_hidden_layers"],
            decoder_hidden_layers=optimization_params["decoder_hidden_layers"],
            message_hidden_layers=optimization_params["message_hidden_layers"],
            update_hidden_layers=optimization_params["update_hidden_layers"],
            **best_params
        ), optimization_params

    def _get_optimizing_value(name, value, trial):
        if isinstance(value, tuple) or isinstance(value, list):
            use_log = value[1] / value[0] >= 1000
            if isinstance(value[0], int):
                return trial.suggest_int(name, value[0], value[1], log=use_log)
            elif isinstance(value[0], float):
                return trial.suggest_float(name, value[0], value[1], log=use_log)
        else:
            return value
