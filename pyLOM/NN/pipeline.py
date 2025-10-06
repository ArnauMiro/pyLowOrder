#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN pipeline routines.
#
# Last rev: 02/10/2024

import numpy as np

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
            self.train_dataset, eval_dataset=self.valid_dataset, **self.training_params
        )

        return model_output
    

class ClusteredPipeline:
    r"""
    Train one surrogate model per cluster.
    
    Args:
        train_dataset: the training dataset.
        cluster_col_idx (int): the index of the column containing the cluster labels.
        valid_dataset (optional): the validation dataset. Default is ``None``.
        test_dataset (optional): the test dataset. Default is ``None``.
        models_list (List, optional): the list of models to train for each cluster. Default is ``[]``.
        training_params_list (List[Dict], optional): the list of training parameters for each cluster. Default is ``[]``.
        optimizers_list (List[OptunaOptimizer], optional): the list of optimizers to use for each cluster. Default is ``[]``.
        model_classes_list (List, optional): the list of model classes to use for each cluster. Default is ``[]``.

    Raises:
        AssertionError: If neither models_list and training_params_list nor optimizers_list and model_classes

    Returns:
        model_outputs_list (List): the list of outputs of the model's fit method for each cluster.

    """
    def __init__(
        self,
        train_dataset,
        cluster_col_idx: int,
        valid_dataset=None,
        test_dataset=None, 
        models_list: List = None,
        training_params_list: List[Dict] = None,
        optimizers_list: List[OptunaOptimizer] = None,
        model_classes_list: List = None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset

        self.train_dataset_list = []
        self.valid_dataset_list = []
        self.test_dataset_list = []

        self.cluster_col_idx = cluster_col_idx

        self._models = [] if models_list is None else models_list
        self.training_params_list = [] if training_params_list is None else training_params_list
        self.optimizers_list = [] if optimizers_list is None else optimizers_list
        self.model_classes_list = [] if model_classes_list is None else model_classes_list
        self.model_outputs_list = []

        assert (self.optimizers_list is not None and self.model_classes_list is not None) or (
            self._models is not None and self.training_params_list is not None
        ), "Either models_list and training_params_list or optimizers_list and model_classes_list must be provided"

        self.n_clusters = self._get_n_clusters()
        if len(self._models) > 0:
            assert len(self._models) == self.n_clusters, "Number of models must match number of clusters"
        if len(self.training_params_list) > 0:
            assert len(self.training_params_list) == self.n_clusters, "Number of training_params must match number of clusters"
        if len(self.optimizers_list) > 0:
            assert len(self.optimizers_list) == self.n_clusters, "Number of optimizers must match number of clusters"
        if len(self.model_classes_list) > 0:
            assert len(self.model_classes_list) == self.n_clusters, "Number of model_classes must match number of clusters"

    def _get_n_clusters(self):
        return len(self._get_cluster_ids())
    
    def _get_cluster_ids(self):
        return set(self.train_dataset[:][0][:, self.cluster_col_idx])
    
    def _filter_one_cluster(self, dataset, cluster_id):

        def filter_by_cluster(inputs, outputs, cluster_id, cluster_col):
            inputs = np.asarray(inputs)
            tol = 1e-8
            mask = (np.abs(inputs[:, cluster_col] - cluster_id) < tol)
            return mask.tolist()
    
        ds_filtered = dataset.filter(
            function=filter_by_cluster,
            fn_kwargs={"cluster_id": cluster_id, "cluster_col": self.cluster_col_idx},
            batched=True,
            batch_size=len(dataset),
            return_views=False,
        )
        ds_filtered.remove_column(self.cluster_col_idx, from_variables_out=False)
        return ds_filtered

    @property
    def models(self):
        """
        Get the trained models.
        """
        return self._models

    def run(self):
        cluster_ids = self._get_cluster_ids()
        pprint(0, f"Found {len(cluster_ids)} clusters: {cluster_ids}")

        for idx, cid in enumerate(cluster_ids):
            pprint(0, f"\n--- Cluster {cid} ---")
            train_ds_c = self._filter_one_cluster(self.train_dataset, cid)
            valid_ds_c = self._filter_one_cluster(self.valid_dataset, cid) if self.valid_dataset is not None else None
            test_ds_c = self._filter_one_cluster(self.test_dataset, cid) if self.test_dataset is not None else None

            if self.optimizers_list is not None:
                if valid_ds_c is None:
                    valid_ds_c = train_ds_c
                    raiseWarning( "Validation dataset not provided, using train dataset for evaluation on optimization")

                pprint(0, "Optimizing model hyperparameters for this cluster")
                model_c, training_params_c = self.model_classes_list[idx].create_optimized_model(
                    train_dataset = train_ds_c,
                    eval_dataset = valid_ds_c,
                    optuna_optimizer = self.optimizers_list[idx],
                )

                self._models.append(model_c)
                self.training_params_list.append(training_params_c)

            pprint(0, "Training a model with optimized parameters")
            model_output_c = self._models[idx].fit(train_ds_c, **self.training_params_list[idx])
            self.model_outputs_list.append(model_output_c)

            self.train_dataset_list.append(train_ds_c)
            self.valid_dataset_list.append(valid_ds_c)
            self.test_dataset_list.append(test_ds_c)

        return self.model_outputs_list