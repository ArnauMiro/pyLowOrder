#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN pipeline routines.
#
# Last rev: 02/10/2024

from typing         import List, Dict, Any
from .optimizer     import OptunaOptimizer
from ..utils.errors import raiseWarning
from ..            import pprint

class Pipeline:
    r"""
    Pipeline class to train and evaluate models. 
    To optimize a model, provide an optimizer and model class.
    To train a model with fixed parameters, provide a model and training parameters.

    Args: 
        train_dataset: The training dataset.
        valid_dataset (optional): The validation dataset. Default is ``None``.
        test_dataset (optional): The test dataset. Default is ``None``.
        model (Model, optional): The model to train. Default is ``None``. 
            If optimizer and model_class are provided, this is not used.
        training_params (Dict, optional): The parameters for training the model. Default is ``None``. 
            If optimizer and model_class are provided, this is not used.
        optimizer (OptunaOptimizer, optional): The optimizer to use for optimization. Default is ``None``.
        model_class (Model, optional): The model class to use for optimization. Default is ``None``.
        evaluators (List, optional): The evaluators to use for evaluating the model. Default is ``[]``.

    Raises:
        AssertionError: If neither model and training_params nor optimizer and model_class are provided.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset=None,
        test_dataset=None, 
        model=None, 
        training_params: Dict = None,
        optimizer: OptunaOptimizer = None,
        model_class=None,
        evaluators: List = [],
    ):
        self._model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.training_params = training_params
        self.model_class = model_class
        self.evaluators = evaluators

        assert (self.optimizer is not None and self.model_class is not None) or (
            self._model is not None and self.training_params is not None
        ), "Either model and training_params or optimizer and model_class must be provided"

    @property
    def model(self):
        """
        Get the trained model.
        """
        return self._model

    def run(self) -> Any:
        """
        Run the pipeline, this will train the model and return the output of the model's fit method. If optuna is used, the model will be trained with the optimized parameters.

        Returns:
            model_output (Any): The output of the model's fit method.
        """
        if self.optimizer is not None:
            if self.valid_dataset is None:
                self.valid_dataset = self.train_dataset
                raiseWarning( "Validation dataset not provided, using train dataset for evaluation on optimization")

            self._model, self.training_params = self.model_class.create_optimized_model(
                train_dataset = self.train_dataset,
                eval_dataset = self.valid_dataset,
                optuna_optimizer = self.optimizer,
            )
            pprint(0, "Training now a model with optimized parameters")

        model_output = self._model.fit(
            self.train_dataset, eval_dataset=self.test_dataset, **self.training_params
        )

        return model_output