#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 21/03/2025

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.nn import ELU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from pyLOM import Mesh
from pyLOM.vmmath.geometric import edge_to_cells, wall_normals

from typing import Protocol, Optional, Dict, Tuple, Union

class MLP(torch.nn.Module):
    '''Simple MLP with dropout and activation function'''
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
    def __init__(self, in_channels, out_channels, hiddim, drop_p=0.):
        # Message passing with "mean" aggregation.
        super().__init__(aggr='mean')
        self.dropout = torch.nn.Dropout(p=drop_p)

        # MLP for the message function
        self.phi = MLP(in_channels, out_channels, 1*[hiddim], drop_p=0, activation=ELU())
        
        # MLP for the update function
        self.gamma = MLP(2*out_channels, out_channels, 1*[hiddim], drop_p=0, activation=ELU())


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




# class GNS(nn.Module):
#     """
#     Graph Neural Network model for predicting aerodynamic variables on RANS meshes.
#     The model uses a message-passing architecture with MLPs for the message and update functions.

#     Args:
#         input_dim (int): The number of input features.
#         latent_dim (int): The number of latent features.
#         output_dim (int): The number of output features.
#         hidden_size (int): The number of hidden units in the MLPs.
#         GNN_layers (int): The number of GNN layers.
#         encoder_hidden_layers (int): The number of hidden layers in the encoder.
#         decoder_hidden_layers (int): The number of hidden layers in the decoder.
#         message_hidden_layers (int): The number of hidden layers in the message MLP.
#         update_hidden_layers (int): The number of hidden layers in the update MLP.
#         activation (Union[str, nn.Module]): The activation function to use.
#         drop_p (float): The dropout probability.
#         edge_dim (int): The number of edge features.
#         graph (Optional[Data]): torch-geometric Data object with the graph structure.
#     """

#     def __init__(self,
#                  input_dim: int,
#                  latent_dim: int,
#                  output_dim: int,
#                  hidden_size: int,
#                  GNN_layers: int,
#                  encoder_hidden_layers: int,
#                  decoder_hidden_layers: int,
#                  message_hidden_layers: int,
#                  update_hidden_layers: int,
#                  activation: Union[str, nn.Module] = nn.ELU(),
#                  drop_p: float = 0.,
#                  edge_dim: int = 6,
#                  graph: Optional[Data] = None):
#         super().__init__()
#         torch.manual_seed(11235)

#         # Save the model parameters
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.output_dim = output_dim
#         self.hidden_size = hidden_size
#         self.GNN_layers = GNN_layers
#         self.encoder_hidden_layers = encoder_hidden_layers
#         self.decoder_hidden_layers = decoder_hidden_layers
#         self.drop_p = drop_p
#         self.edge_dim = edge_dim
#         self.graph = graph

#         # Activation function
#         if isinstance(activation, str):
#             if hasattr(nn, activation):  
#                 self.activation = getattr(nn, activation)()
#             else:
#                 raise ValueError(f"Activation function {activation} not found in torch.nn")
#         else:
#             self.activation = activation

#         # Save a dictionary with the model parameters
#         self.model_dict = {
#             "input_dim": input_dim,
#             "latent_dim": latent_dim,
#             "output_dim": output_dim,
#             "hidden_size": hidden_size,
#             "GNN_layers": GNN_layers,
#             "encoder_hidden_layers": encoder_hidden_layers,
#             "decoder_hidden_layers": decoder_hidden_layers,
#             "message_hidden_layers": message_hidden_layers,
#             "update_hidden_layers": update_hidden_layers,
#             "activation": self.activation.__class__.__name__,
#             "drop_p": drop_p,
#             "edge_dim": edge_dim,
#             "graph": graph,
#             "model_parameters": {}
#         }

#         # Encoder: from graph node features to latent space
#         self.encoder = MLP(
#             input_size=self.input_dim,
#             output_size=self.latent_dim,
#             hidden_sizes=[self.hidden_size] * self.encoder_hidden_layers,
#             activation=self.activation,
#             drop_p=self.drop_p
#         )

#         # Decoder: from latent space to output features
#         self.decoder = MLP(
#             input_size=self.latent_dim,
#             output_size=self.output_dim,
#             hidden_sizes=[self.hidden_size] * self.decoder_hidden_layers,
#             activation=self.activation,
#             drop_p=self.drop_p
#         )

#         # Message-passing layers
#         self.conv_layers_list = nn.ModuleList([
#             MessagePassingLayer(
#                 in_channels=2 * self.latent_dim + self.edge_dim,
#                 out_channels=self.latent_dim,
#                 drop_p=self.drop_p,
#                 hiddim=self.hidden_size
#             )
#             for _ in range(self.GNN_layers)
#         ])

#         # Normalization layer
#         self.groupnorm = nn.GroupNorm(2, self.latent_dim)

    
    
#     def forward(self, subgraph):
#         """
#         Forward pass of the model.

#         Args:
#             subgraph (Data): The input subgraph.

#         Returns:
#             torch.Tensor: The predicted target values.
#         """

#         # Get node and edge features
#         x = subgraph.x
#         edge_index = subgraph.edge_index
#         edge_attr = subgraph.edge_attr

#         # 1. Encode node features
#         h = self.encoder(x)
#         h = self.activation(h)

#         # 2. Message-passing layers
#         for conv in self.conv_layers_list:
#             h = conv(h, edge_index, edge_attr)
#             h = self.activation(h)
#             h = self.groupnorm(h)

#         # 3. Decode node features
#         y_hat = self.decoder(h)

#         return y_hat


#     def fit(self, train_dataset: Dataset, eval_set=Optional **kwargs):
#         """
#         Fit the model to the training data.

#         Args:
#             train_dataset: The training dataset.
#             eval_set (Optional): The evaluation dataset.
#             **kwargs: Additional parameters for the fit method.
#         """

#         pass

#     def predict(self, X: Dataset, **kwargs):
#         """
#         Predict the target values for the input data.
        
#         Args:
#             X: The input data. This dataset should have the same type as 
#             the ones used on fit
#             **kwargs: Additional parameters for the predict method.

#         Returns:
#             np.array: The predicted target values.
#         """
#         pass

#     def load_graph_from_mesh(self, mesh: Mesh):
#         """
#         Build the graph needed to train the model from a pyLOM Mesh object.

#         Args:
#             mesh (pyLOM.Mesh): The input mesh.

#         Returns:
#             Data: The graph structure.
#         """
#         pass

#     @classmethod
#     def create_optimized_model(
#         cls,
#         train_dataset: Dataset,
#         eval_dataset: Optional[Dataset],
#         optuna_optimizer: OptunaOptimizer,
#     ) -> Tuple["Model", Dict]:
#         """
#         Create an optimized model using Optuna.

#         Args:
#             train_dataset (BaseDataset): The training dataset.
#             eval_dataset (Optional[BaseDataset]): The evaluation dataset.
#             optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.

#         Returns:
#             Tuple[Model, Dict]: The optimized model and the best parameters
#             found by the optimizer.
#         """

#     def save(self, path: str):
#         """
#         Save the model to a file.

#         Args:
#             path (str): The path to save the model.
#         """
#         pass
    
#     @classmethod
#     def load(self, path: str):
#         """
#         Load a model from a file.

#         Args:
#             path (str): The path to load the model from.

#         Returns:
#             Model: The loaded model.
#         """
#         pass

#     @property
#     def trainable_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad) 


class ScalerProtocol(Protocol):
    '''
    Protocol for scalers. Must include:
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


class Graph(Data):
    '''
    Custom class derived from torch.geometric.Data to handle graph data.
    
    Custom features:
        - from_pyLOM_mesh: Create a torch_geometric Data object from a pyLOM Mesh object.
        - filter: Filter graph by eliminating nodes not in node_mask.
    '''

    @classmethod
    def from_pyLOM_mesh(cls,
                        mesh: Mesh,
                        y: Optional[np.ndarray] = None,
                        scaler: Optional[ScalerProtocol] = None,
                        operational_parameters_size: int = 3):
        """
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
            Data: The graph structure.
        """
        xyzc = mesh.xyzc  # Cell centers coordinates
        print("Computing surface normals")
        surface_normals = mesh.normal
        print("Surface normals computed")

        print("Computing dual edges and wall normals")
        edge_index, wall_normals = cls._dual_edges_and_wall_normals(mesh)

        # Create the node features (node coordinates + surface normals)
        # Add dummy columns to fill with the operational parameters during batch training
        x = np.concatenate((
            xyzc,
            surface_normals,
            np.zeros((xyzc.shape[0], operational_parameters_size))
        ),
        axis=1)

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
            x = scaler.fit_transform(x)
            edge_attr = scaler.fit_transform(edge_attr)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # Return the class instance with the necessary attributes
        return cls(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    

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

        self.x = self.x[node_mask]
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