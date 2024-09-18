from typing import List, Dict
import warnings



class Pipeline:
    r"""
    Pipeline class to train and evaluate models. 
    To optimize a model, provide an optimizer and model class.
    To train a model with fixed parameters, provide a model and training parameters.

    Args: 
        train_dataset (BaseDataset): The training dataset.
        valid_dataset (BaseDataset, optional): The validation dataset. Default is `None`.
        test_dataset (BaseDataset, optional): The test dataset. Default is `None`.
        model (Model, optional): The model to train. Default is `None`. 
            If optimizer and model_class are provided, this is not used.
        training_params (Dict, optional): The parameters for training the model. Default is `None`. 
            If optimizer and model_class are provided, this is not used.
        optimizer (OptunaOptimizer, optional): The optimizer to use for optimization. Default is `None`.
        model_class (Model, optional): The model class to use for optimization. Default is `None`.
        evaluators (List, optional): The evaluators to use for evaluating the model. Default is `[]`.

    Raises:
        AssertionError: If neither model and training_params nor optimizer and model_class are provided.
    """

    def __init__(
        self,
        train_dataset, #: BaseDataset,
        valid_dataset=None, #: BaseDataset = None,
        test_dataset=None, #: BaseDataset = None,
        model=None, #: Model = None,
        training_params: Dict = None,
        optimizer=None, #: OptunaOptimizer = None,
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
    def model(self): # -> Model:
        """
        Get the trained model.
        """
        return self._model

    def run(self):
        """
        Run the pipeline.
        """
        if self.optimizer is not None:
            if self.valid_dataset is None:
                self.valid_dataset = self.train_dataset
                warnings.warn(
                    "Validation dataset not provided, using train dataset for evaluation on optimization"
                )

            self._model, self.training_params = self.model_class.create_optimized_model(
                train_dataset = self.train_dataset,
                eval_dataset = self.valid_dataset,
                optuna_optimizer = self.optimizer,
            )

        self._model.fit(
            self.train_dataset, eval_dataset=self.test_dataset, **self.training_params
        )