#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 21/03/2025


import torch
from torch.nn import ELU
from torch_geometric.nn import MessagePassing

class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, drop_p=0.5, hiddim=HIDDEN_SIZE):
        # Message passing with "mean" aggregation.
        super().__init__(aggr='mean')
        self.dropout = torch.nn.Dropout(p=drop_p)

        # MLP for the message function
        self.phi = utils.MLP(in_channels, out_channels, 1*[hiddim], drop_p=0, activation=ELU())
        
        # MLP for the update function
        self.gamma = utils.MLP(2*out_channels, out_channels, 1*[hiddim], drop_p=0, activation=ELU())


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


class Model():
    def __init__(self, **kwargs):
        """
        Initialize the model.

        Args:
            **kwargs: Additional parameters for the model.
        """
        try:
            self.model_dict = kwargs["model_dict"]
            self.from_dict(self.model_dict)
        except KeyError:
            self.model_dict = None
            print("No model dictionary provided. Model is empty!"
            "You can load a model from a file or create a new one.")


    def from_dict(self, model_dict: dict):
        """
        Load the model from a dictionary.

        Args:
            model_dict (Dict): The dictionary containing the model parameters.
        """
        self.model_dict = model_dict

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