#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 21/03/2025


import torch
import torch.nn as nn
from torch.nn import ELU
from torch_geometric.nn import MessagePassing

from typing import Optional, Dict, Tuple, Union

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




class GNS(nn.Module):
    """
    Graph Neural Network model for predicting aerodynamic variables on RANS meshes.
    The model uses a message-passing architecture with MLPs for the message and update functions.

    Args:
        input_dim (int): The number of input features.
        latent_dim (int): The number of latent features.
        output_dim (int): The number of output features.
        hidden_size (int): The number of hidden units in the MLPs.
        GNN_layers (int): The number of GNN layers.
        encoder_hidden_layers (int): The number of hidden layers in the encoder.
        decoder_hidden_layers (int): The number of hidden layers in the decoder.
        message_hidden_layers (int): The number of hidden layers in the message MLP.
        update_hidden_layers (int): The number of hidden layers in the update MLP.
        activation (Union[str, nn.Module]): The activation function to use.
        drop_p (float): The dropout probability.
        edge_dim (int): The number of edge features.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 output_dim: int,
                 hidden_size: int,
                 GNN_layers: int,
                 encoder_hidden_layers: int,
                 decoder_hidden_layers: int,
                 message_hidden_layers: int,
                 update_hidden_layers: int,
                 activation: Union[str, nn.Module] = nn.ELU(),
                 drop_p: float = 0.,
                 edge_dim: int = 6):
        super().__init__()
        torch.manual_seed(11235)

        # Guardar en atributos de la clase
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.GNN_layers = GNN_layers
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.drop_p = drop_p
        self.edge_dim = edge_dim

        # Manejo de la activación
        if isinstance(activation, str):
            if hasattr(nn, activation):  
                self.activation = getattr(nn, activation)()
            else:
                raise ValueError(f"Activation function {activation} not found in torch.nn")
        else:
            self.activation = activation

        # Guardar configuración del modelo
        self.model_dict = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "hidden_size": hidden_size,
            "GNN_layers": GNN_layers,
            "encoder_hidden_layers": encoder_hidden_layers,
            "decoder_hidden_layers": decoder_hidden_layers,
            "message_hidden_layers": message_hidden_layers,
            "update_hidden_layers": update_hidden_layers,
            "activation": self.activation.__class__.__name__,
            "drop_p": drop_p,
            "edge_dim": edge_dim,
            "model_parameters": {}
        }

        # Encoder: from node features to latent space
        self.encoder = MLP(
            input_size=self.input_dim,
            output_size=self.latent_dim,
            hidden_sizes=[self.hidden_size] * self.encoder_hidden_layers,
            activation=self.activation,
            drop_p=self.drop_p
        )

        # Decoder: from latent space to output features
        self.decoder = MLP(
            input_size=self.latent_dim,
            output_size=self.output_dim,
            hidden_sizes=[self.hidden_size] * self.decoder_hidden_layers,
            activation=self.activation,
            drop_p=self.drop_p
        )

        # Message-passing layers
        self.conv_layers_list = nn.ModuleList([
            MessagePassingLayer(
                in_channels=2 * self.latent_dim + self.edge_dim,
                out_channels=self.latent_dim,
                drop_p=self.drop_p,
                hiddim=self.hidden_size
            )
            for _ in range(self.GNN_layers)
        ])

        # Normalization layer
        self.groupnorm = nn.GroupNorm(2, self.latent_dim)


    def forward(self, subgraph):
        """
        Forward pass of the model.

        Args:
            subgraph (Data): The input subgraph.

        Returns:
            torch.Tensor: The predicted target values.
        """

        x = subgraph.x
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr

        # 1. Encode node features
        h = self.encoder(x)
        h = self.activation(h)

        # 2. Apply PointNet layer
        h = self.conv1(h, edge_index, edge_attr)
        h = self.activation(h)
        # h = self.conv2(h, edge_index, edge_attr)
        # h = self.activation(h)
        # h = self.groupnorm(h)

        # 3. Decode node features
        y_hat = self.decoder(h)

        return y_hat


    def fit(self, train_dataset: Dataset, eval_set=Optional **kwargs):
        """
        Fit the model to the training data.

        Args:
            train_dataset: The training dataset.
            eval_set (Optional): The evaluation dataset.
            **kwargs: Additional parameters for the fit method.
        """

        pass

    def predict(self, X: Dataset, **kwargs):
        """
        Predict the target values for the input data.
        
        Args:
            X: The input data. This dataset should have the same type as 
            the ones used on fit
            **kwargs: Additional parameters for the predict method.

        Returns:
            np.array: The predicted target values.
        """
        pass

    @classmethod
    def create_optimized_model(
        cls,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        optuna_optimizer: OptunaOptimizer,
    ) -> Tuple["Model", Dict]:
        """
        Create an optimized model using Optuna.

        Args:
            train_dataset (BaseDataset): The training dataset.
            eval_dataset (Optional[BaseDataset]): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.

        Returns:
            Tuple[Model, Dict]: The optimized model and the best parameters
            found by the optimizer.
        """

    def save(self, path: str):
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.
        """
        pass
    
    @classmethod
    def load(self, path: str):
        """
        Load a model from a file.

        Args:
            path (str): The path to load the model from.

        Returns:
            Model: The loaded model.
        """
        pass

    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 

#%%

def f(nombre, edad=6):
    model_dict = {"nombre": nombre, "edad": edad}
    print("Hola", nombre, "tienes", edad, "años")

d = {"nombre": "Juan"}

f(**d)