#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN pipeline routines.
#
# Last rev: 02/10/2024

import numpy as np, copy, torch

from typing         import List, Dict, Any
from .optimizer     import OptunaOptimizer
from ..utils.errors import raiseWarning, raiseError
from ..             import pprint

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
        models_dict (Dict, optional): the dictionary of models to train for each cluster. Default is ``{}``.
        training_params_dict (Dict, optional): the dictionary of training parameters for each cluster. Default is ``{}``.
        optimizers_dict (Dict[OptunaOptimizer], optional): the dictionary of optimizers to use for each cluster. Default is ``{}``.
        model_classes_dict (Dict, optional): the dictionary of model classes to use for each cluster. Default is ``{}``.

    Raises:
        AssertionError: If neither models_dict and training_params_dict nor optimizers_dict and model_classes_dict are provided.

    Returns:
        model_outputs_dict (Dict): the dictionary of outputs of the model's fit method for each cluster.

    """
    def __init__(
        self,
        train_dataset,
        cluster_col_idx: int,
        valid_dataset=None,
        test_dataset=None, 
        models_dict: Dict = None,
        training_params_dict: Dict = None,
        optimizers_dict: Dict = None,
        model_classes_dict: Dict = None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset

        assert (optimizers_dict is not None and model_classes_dict is not None) or (
            models_dict is not None and training_params_dict is not None
        ), "Either models_dict and training_params_dict or optimizers_dict and model_classes_dict must be provided"

        self.cluster_col_idx = cluster_col_idx
        self.n_clusters = self._get_n_clusters()

        self.dict_keys = ["classifier"]
        self.dict_keys += [f"regressor_{cid}" for cid in self._get_cluster_ids()]
        self.train_dataset_dict, self.valid_dataset_dict, self.test_dataset_dict = self._create_dataset_dict()

        if models_dict is not None:
            assert all(k in models_dict.keys() for k in self.dict_keys), "models_dict must contain keys: " + ", ".join(self.dict_keys)
            assert (len(models_dict)-1) == self.n_clusters, "Number of models must match number of clusters"
            self._models = models_dict
        else:
            self._models = {k: None for k in self.dict_keys}

        if training_params_dict is not None:
            assert all(k in training_params_dict.keys() for k in self.dict_keys), "training_params_dict must contain keys: " + ", ".join(self.dict_keys)
            assert (len(training_params_dict)-1) == self.n_clusters, "Number of training_params must match number of clusters"
            self.training_params_dict = training_params_dict
        else:
            self.training_params_dict = {k: None for k in self.dict_keys}

        if optimizers_dict is not None:
            assert all(k in optimizers_dict.keys() for k in self.dict_keys), "optimizers_dict must contain keys: " + ", ".join(self.dict_keys[1:])
            assert (len(optimizers_dict)-1) == self.n_clusters, "Number of optimizers must match number of clusters"
            self.optimizers_dict = optimizers_dict
        else:
            self.optimizers_dict = {k: None for k in self.dict_keys}

        if model_classes_dict is not None:
            assert all(k in model_classes_dict.keys() for k in self.dict_keys), "model_classes_dict must contain keys: " + ", ".join(self.dict_keys[1:])
            assert (len(model_classes_dict)-1) == self.n_clusters, "Number of model_classes must match number of clusters"
            self.model_classes_dict = model_classes_dict
        else:
            self.model_classes_dict = {k: None for k in self.dict_keys}

        self.model_outputs_dict = {k: None for k in self.dict_keys}

    def _get_n_clusters(self):
        return len(self._get_cluster_ids())
    
    def _get_cluster_ids(self):
        cluster_col = self.train_dataset[:][1][:, self.cluster_col_idx]
        cluster_col_np = cluster_col.detach().cpu().numpy() if hasattr(cluster_col, "detach") else np.asarray(cluster_col)
        unique = np.unique(cluster_col_np)
        return np.rint(unique).astype(np.int64).tolist()
    
    def _create_dataset_dict(self):
        self.train_dataset_dict, self.valid_dataset_dict, self.test_dataset_dict = {}, {}, {}
        self.train_dataset_dict["classifier"] = self._keep_cluster_column(self.train_dataset)
        self.valid_dataset_dict["classifier"] = self._keep_cluster_column(self.valid_dataset) if self.valid_dataset is not None else None
        self.test_dataset_dict["classifier"] = self._keep_cluster_column(self.test_dataset) if self.test_dataset is not None else None

        for cid in self._get_cluster_ids():
            self.train_dataset_dict[f"regressor_{cid}"] = self._filter_one_cluster(self.train_dataset, cid)
            self.valid_dataset_dict[f"regressor_{cid}"] = self._filter_one_cluster(self.valid_dataset, cid) if self.valid_dataset is not None else None
            self.test_dataset_dict[f"regressor_{cid}"] = self._filter_one_cluster(self.test_dataset, cid) if self.test_dataset is not None else None

        return self.train_dataset_dict, self.valid_dataset_dict, self.test_dataset_dict

    def _filter_one_cluster(self, dataset, cluster_id):

        def filter_by_cluster(inputs, outputs, cluster_id, cluster_col):
            arr = outputs.detach().cpu().numpy() if hasattr(outputs, "detach") else np.asarray(outputs)
            cluster_vals = np.rint(arr[:, cluster_col]).astype(np.int64)
            cid = int(np.rint(cluster_id))
            return (cluster_vals == cid).tolist()

        ds_filtered = dataset.filter(
            function=filter_by_cluster,
            fn_kwargs={"cluster_id": cluster_id, "cluster_col": self.cluster_col_idx},
            batched=True,
            batch_size=len(dataset),
            return_views=False,
        )
        ds_filtered.remove_column(self.cluster_col_idx, from_variables_out=True)
        return ds_filtered

    def _keep_cluster_column(self, dataset):
        ds_new = copy.deepcopy(dataset)
        ncols = ds_new[:][1].shape[1]

        i = self.cluster_col_idx
        if i < 0:
            i = ncols + i
        if not (0 <= i < ncols):
            raiseError(f"Invalid output column index {i} for ncols={ncols}")

        for col in range(ncols-1, -1, -1):
            if col != i:
                ds_new.remove_column(col, from_variables_out=True)

        outs = ds_new[:][1]
        if outs.ndim == 1:
            ds_new.variables_out = outs.unsqueeze(1)
        elif outs.ndim == 2 and outs.shape[1] == 1:
            ds_new.variables_out = outs
        else:
            raiseError(f"Expected 1 output column after removal, got shape {tuple(outs.shape)}")
        ds_new.num_channels = 1
        return ds_new

    def _drop_cluster_column(self, dataset):
        ds_new = copy.deepcopy(dataset)
        ds_new.remove_column(self.cluster_col_idx, from_variables_out=True)
        ds_new.num_channels = ds_new.num_channels - 1
        return ds_new

    @property
    def models(self):
        """
        Get the trained models.
        """
        return self._models

    def run(self):
        cluster_ids = self._get_cluster_ids()
        pprint(0, f"Found {len(cluster_ids)} clusters: {cluster_ids}")

        # Train the classifier to predict the cluster from the inputs
        if self.optimizers_dict["classifier"] is not None:
            if self.valid_dataset_dict["classifier"] is None:
                self.valid_dataset_dict["classifier"] = self.train_dataset_dict["classifier"]
                raiseWarning("Validation dataset not provided, using train dataset for evaluation on optimization")

            pprint(0, "Optimizing classifier hyperparameters")
            model_classifier, training_params_classifier = self.model_classes_dict["classifier"].create_optimized_model(
                train_dataset = self.train_dataset_dict["classifier"],
                eval_dataset = self.valid_dataset_dict["classifier"],
                optuna_optimizer = self.optimizers_dict["classifier"],
            )

            self._models["classifier"] = model_classifier
            self.training_params_dict["classifier"] = training_params_classifier
        
        pprint(0, "Training the classifier with optimized parameters")
        model_output_classifier = self._models["classifier"].fit(
            self.train_dataset_dict["classifier"], 
            eval_dataset=self.valid_dataset_dict["classifier"], 
            **self.training_params_dict["classifier"],
        )
        self.model_outputs_dict["classifier"] = model_output_classifier

        # Train one model per cluster
        for idx, cid in enumerate(cluster_ids):
            pprint(0, f"\n--- Cluster {cid} ---")
            if self.optimizers_dict[f"regressor_{cid}"] is not None:
                if self.valid_dataset_dict[f"regressor_{cid}"] is None:
                    self.valid_dataset_dict[f"regressor_{cid}"] = self.train_dataset_dict[f"regressor_{cid}"]
                    raiseWarning("Validation dataset not provided, using train dataset for evaluation on optimization")

                pprint(0, "Optimizing model hyperparameters for this cluster")
                model_c, training_params_c = self.model_classes_dict[f"regressor_{cid}"].create_optimized_model(
                    train_dataset = self.train_dataset_dict[f"regressor_{cid}"],
                    eval_dataset = self.valid_dataset_dict[f"regressor_{cid}"],
                    optuna_optimizer = self.optimizers_dict[f"regressor_{cid}"],
                )

                self._models[f"regressor_{cid}"] = model_c
                self.training_params_dict[f"regressor_{cid}"] = training_params_c

            pprint(0, "Training a model with optimized parameters")
            model_output_c = self._models[f"regressor_{cid}"].fit(
                train_dataset=self.train_dataset_dict[f"regressor_{cid}"], 
                eval_dataset=self.valid_dataset_dict[f"regressor_{cid}"], 
                **self.training_params_dict[f"regressor_{cid}"],
            )

            self.model_outputs_dict[f"regressor_{cid}"] = model_output_c

        return self.model_outputs_dict

    def evaluate(self, evaluators_dict, scalers: List = [None, None], threshold: float = 0.5, set_to_use: str = "test", deep: bool = False) -> Dict:
        r"""
        Evaluate the models on the test datasets.
        """
        if set_to_use == "train":
            if self.train_dataset is None:
                raiseError("Train dataset not provided, cannot evaluate models")
            self.evaluation_dataset = self.train_dataset
            self.evaluation_dataset_dict = self.train_dataset_dict
        elif set_to_use == "valid":
            if self.valid_dataset is None:
                raiseError("Validation dataset not provided, cannot evaluate models")
            self.evaluation_dataset = self.valid_dataset
            self.evaluation_dataset_dict = self.valid_dataset_dict
        else:
            if self.test_dataset is None:
                raiseError("Test dataset not provided, cannot evaluate models")
            self.evaluation_dataset = self.test_dataset
            self.evaluation_dataset_dict = self.test_dataset_dict

        if self.training_params_dict is None:
            raiseError("Training parameters not available, cannot evaluate models")

        if scalers is not None:
            assert len(scalers) == 2, "scalers must be a list of two elements: [input_scaler, output_scaler]"
            input_scaler, output_scaler = scalers
            output_scaler_classifier = copy.deepcopy(output_scaler)
            output_scaler_classifier.keep_columns([self.cluster_col_idx])
            output_scaler_regressors = copy.deepcopy(output_scaler)
            output_scaler_regressors.drop_columns([self.cluster_col_idx])

        cluster_ids = self._get_cluster_ids()

        def _evaluate_model(model, training_params, dataset, evaluators, inputs_scaler, outputs_scaler, kwargs={}):
            y_pred = model.predict(dataset, rescale_output=True, **training_params)
            x_true, y_true = dataset[:]
            if inputs_scaler is not None:
                x_true = inputs_scaler.inverse_transform(x_true)
            if outputs_scaler is not None:
                y_true = outputs_scaler.inverse_transform(y_true)
                y_pred = outputs_scaler.inverse_transform(y_pred)
            metrics = {}
            for evaluator in evaluators:
                metrics.update(evaluator(y_true, y_pred, **kwargs))
                evaluator.print_metrics()
            return metrics, [x_true, y_true, y_pred]

        metrics_dict = {}

        pprint(0, "\nEvaluation of the classifier:")
        metrics_dict["classifier"] = _evaluate_model(
            self._models["classifier"],
            self.training_params_dict["classifier"],
            self.evaluation_dataset_dict["classifier"],
            evaluators_dict["classifier"],
            inputs_scaler = input_scaler,
            outputs_scaler = output_scaler_classifier,
            kwargs={"given_threshold": threshold},
        )

        pprint(0, "\nEvaluation of the regressors:")
        for idx, cid in enumerate(cluster_ids):
            pprint(0, f"\n--- Cluster {cid} ---")
            metrics_dict[f"regressor_{cid}"] = _evaluate_model(
                self._models[f"regressor_{cid}"],
                self.training_params_dict[f"regressor_{cid}"],
                self.evaluation_dataset_dict[f"regressor_{cid}"],
                evaluators_dict[f"regressor_{cid}"],
                inputs_scaler = input_scaler,
                outputs_scaler = output_scaler_regressors,
            )

        pprint(0, "\nEvaluation of the full clustered model:")
        probs = self._models["classifier"].predict(self.evaluation_dataset, rescale_output=True, **self.training_params_dict["classifier"])
        x_true , y_true = self._drop_cluster_column(self.evaluation_dataset)[:]
        y_pred = np.zeros_like(y_true)
        for idx, cid in enumerate(cluster_ids):

            def _mask_filter_fn(inputs, outputs, mask_np):
                return mask_np.tolist()

            mask = (probs.flatten() < threshold) if cid == 0 else (probs.flatten() >= threshold)
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(mask, dtype=bool)

            if mask_np.ndim != 1:
                mask_np = mask_np.reshape(-1)
            if mask_np.size != len(self.evaluation_dataset):
                raiseError(f"Mask length {mask_np.size} != dataset length {len(self.evaluation_dataset)}")

            evaluation_dataset_c = self.evaluation_dataset.filter(
                function=_mask_filter_fn,
                fn_kwargs={"mask_np": mask_np},
                batched=True,
                batch_size=len(self.evaluation_dataset),
                return_views=False,
            )

            evaluation_dataset_c = self._drop_cluster_column(evaluation_dataset_c)
            y_pred[np.flatnonzero(mask_np)] = self._models[f"regressor_{cid}"].predict(evaluation_dataset_c, rescale_output=True, **self.training_params_dict[f"regressor_{cid}"])

        if input_scaler is not None:
            x_true = input_scaler.inverse_transform(x_true)
        if output_scaler_regressors is not None:
            y_true = output_scaler_regressors.inverse_transform(y_true)
            y_pred = output_scaler_regressors.inverse_transform(y_pred)
        
        metrics = {}
        for evaluator in evaluators_dict["full_model"]:
            metrics.update(evaluator(y_true, y_pred))
            evaluator.print_metrics()

        metrics_dict["full_model"] = metrics, [x_true, y_true, y_pred]

        if deep:
            
            # Error of the models conditioned to the classifier output
            pprint(0, "\nDeep evaluation of the full clustered model:")
            y_true_classifier = metrics_dict["classifier"][1][1]
            y_pred_probs_classifier = metrics_dict["classifier"][1][2]
            y_pred_classifier = torch.Tensor([0 if p < threshold else 1 for p in y_pred_probs_classifier.flatten()])

            mask_tp = ( (y_true_classifier.flatten() == 1) & (y_pred_classifier.flatten() == 1) )
            mask_tn = ( (y_true_classifier.flatten() == 0) & (y_pred_classifier.flatten() == 0) )
            mask_fp = ( (y_true_classifier.flatten() == 0) & (y_pred_classifier.flatten() == 1) )
            mask_fn = ( (y_true_classifier.flatten() == 1) & (y_pred_classifier.flatten() == 0) )
            masks = [mask_tp, mask_tn, mask_fp, mask_fn]
            names = ['True Positive', 'True Negative', 'False Positive', 'False Negative']

            x_true_model = metrics_dict["full_model"][1][0]
            y_true_model = metrics_dict["full_model"][1][1]
            y_pred_model = metrics_dict["full_model"][1][2]

            for midx, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.detach().cpu().numpy().astype(bool)
                else:
                    mask_np = np.asarray(mask, dtype=bool)

                # if mask_np.ndim != 1:
                #     mask_np = mask_np.reshape(-1)
                if mask_np.size != len(self.evaluation_dataset):
                    raiseError(f"Mask length {mask_np.size} != dataset length {len(self.evaluation_dataset)}")

                x_true_cond = x_true_model[mask_np]
                y_true_cond = y_true_model[mask_np]
                y_pred_cond = y_pred_model[mask_np]

                metrics_cond = {}
                for evaluator in evaluators_dict["full_model"]:
                    metrics_cond.update(evaluator(y_true_cond, y_pred_cond))
                    pprint(0, f"\nConditioned on {names[midx]}:")
                    evaluator.print_metrics()

                key_suffix = ""
                if np.all(mask_np == (mask_tp.detach().cpu().numpy().astype(bool))):
                    key_suffix = "TP"
                elif np.all(mask_np == (mask_tn.detach().cpu().numpy().astype(bool))):
                    key_suffix = "TN"
                elif np.all(mask_np == (mask_fp.detach().cpu().numpy().astype(bool))):
                    key_suffix = "FP"
                elif np.all(mask_np == (mask_fn.detach().cpu().numpy().astype(bool))):
                    key_suffix = "FN"

                metrics_dict[f"full_model_{key_suffix}"] = metrics_cond, [x_true_cond, y_true_cond, y_pred_cond]

        return metrics_dict
