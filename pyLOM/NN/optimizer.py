#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN optimizer routines using optuna.
#
# Last rev: 25/11/2026

import json, os, math, random, inspect, numbers, numpy as np, torch

from typing             import Callable, Dict, Sequence
from pathlib            import Path
from itertools          import count

from ..utils.errors     import raiseError, raiseWarning
from ..                 import pprint
from .utils.callbacks   import EarlyStopper


def _check_seeds(seeds):
    if seeds is None:
        return None
    if isinstance(seeds, numbers.Integral):
        seeds = [seeds]
    seeds = list(seeds)
    if len(seeds) == 0 or not all(isinstance(seed, numbers.Integral) and not isinstance(seed, bool) for seed in seeds):
        raiseError(f"seeds must be a non-empty sequence of integers or None, got {seeds}")
    if len(set(seeds)) != len(seeds):
        raiseError(f"seeds must be different, got {seeds}")
    return [int(seed) for seed in seeds]


def _get_rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _set_rng_state(state):
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_ISSUED_WARNINGS = set()


# Add optuna as an optional dependency
try:
    import optuna
    from optuna.trial      import TrialState
    from optuna.exceptions import TrialPruned

    class OptunaOptimizer():
        """
        Args:
            optimization_params (Dict): A dictionary containing the parameters to optimize.
            n_trials (int): The number of trials to run. Default is ``100``.
            direction (str): The direction to optimize. Can be 'minimize' or 'maximize'. Default is ``'minimize'``.
            pruner (optuna.pruners.BasePruner): The pruner to use. Default is ``None``.
            save_dir (str): The directory to save the best parameters. Default is ``None``.
            storage (str): The storage URL to use for the study. Default is ``None``.
            study_name (str): The name of the study. Default is ``None``.
            load_if_exists (bool): Whether to load the study if it exists. Default is ``True``.
            seeds (Sequence[int], optional): Seeds of the models trained in each trial. If given, every trial trains one model per seed and its value is the mean over the seeds of the value of each model. The ``seed`` of ``optimization_params``, if any, is then ignored. If ``None``, each trial trains a single model with the ``seed`` of ``optimization_params`` (default: ``None``).
        """
        def __init__(
            self,
            optimization_params: Dict,
            n_trials: int = 100,
            direction: str = 'minimize',
            sampler: optuna.samplers.BaseSampler | None = None,
            pruner: optuna.pruners.BasePruner | None = None,
            save_dir: str | None = None,
            storage: str | None = None,
            study_name: str | None = None,
            load_if_exists: bool = True,
            seeds: Sequence[int] | None = None,
        ):
            self.num_trials = n_trials
            self.direction = direction
            self.sampler = sampler
            self.pruner = pruner
            self._optimization_params = optimization_params
            self.save_dir = save_dir
            self.storage = storage
            self.study_name = study_name
            self.load_if_exists = load_if_exists
            self.seeds = _check_seeds(seeds)

        @property
        def optimization_params(self) -> Dict:
            """
            Get the optimization parameters.
            """
            return self._optimization_params
        
        def _create_study(self):
            """
            Create an Optuna study with the specified parameters.
            """
            if self.storage is not None:
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    direction=self.direction,
                    sampler=self.sampler,
                    pruner=self.pruner,
                    load_if_exists=self.load_if_exists,
                )
            else:
                study = optuna.create_study(
                    direction=self.direction,
                    sampler=self.sampler,
                    pruner=self.pruner,
                )
            return study
        
        @staticmethod
        def _unflatten_dict(flat_dict, sep="."):
            """
            Convert a flat dictionary with keys containing separators into a nested dictionary.

            Args:
                flat_dict (Dict): The flat dictionary to convert.
                sep (str): The separator used in the keys to indicate nesting (default: ".").
            
            Returns:
                Dict: A nested dictionary constructed from the flat dictionary.
            """
            nested = {}
            for key, value in flat_dict.items():
                parts = key.split(sep)
                d = nested
                for part in parts[:-1]:
                    if part not in d or not isinstance(d[part], dict):
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            return nested
        
        def optimize(
            self, 
            objective_function: Callable[[optuna.Trial], float],
        ) -> Dict:
            """
            Optimize a model given an objective function.
            
            Args:
                objective_function (Callable): The objective function to optimize. The function should take a `optuna.Trial` object as input and return a float.
                
            Returns:
                Dict: The best parameters obtained from the optimization.
            """
            study = self._create_study()
            if self.seeds is not None:
                # The values of the trials are only comparable if all of them average the same seeds
                previous_seeds = study.user_attrs.get("seeds")
                if previous_seeds is not None and list(previous_seeds) != self.seeds:
                    raiseWarning(f"The study was run with the seeds {previous_seeds}, and it continues with {self.seeds}")
                study.set_user_attr("seeds", self.seeds)
            study.optimize(objective_function, n_trials=self.num_trials)
            OptunaOptimizer._print_optimization_report(study)

            best_params_flat = study.best_params
            if self.save_dir is not None:
                self._save_best_params(best_params_flat)

            return self._unflatten_dict(best_params_flat)

        @staticmethod
        def _print_optimization_report(study):
            """
            Print a summary report of the optimization results.
            
            Args:
                study (optuna.study.Study): The Optuna study object containing the optimization results.
            """
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            
            # Optimization report
            pprint(0, "\nStudy statistics: ")
            pprint(0, "  Number of finished trials: ", len(study.trials))
            pprint(0, "  Number of pruned trials: ", len(pruned_trials))
            pprint(0, "  Number of completed trials: ", len(completed_trials))

            trial = study.best_trial
            pprint(0, "Best trial:")
            pprint(0, "  Value: ", trial.value)
            pprint(0, "  Params: ")
            for key, value in trial.params.items():
                pprint(0, "    {}: {}".format(key, value))
            pprint(0, "\n")

        def _save_best_params(self, best_params_flat: Dict):
            """
            Save the best parameters to a JSON file in the specified directory. 
            If a study name is provided, the file is saved using that name and overwritten if it already exists. Otherwise, a unique numbered filename is generated.
            
            Args:
                best_params_flat (Dict): The best parameters in flat format to save.
            """
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            if self.study_name is not None:
                file_path = save_dir / f"{self.study_name}.json"
                with open(file_path, "w") as f:
                    json.dump(best_params_flat, f, indent=2, sort_keys=True)
            else:
                for i in count():
                    file_path = save_dir / f"best_params_{i}.json"
                    if not file_path.exists():
                        with open(file_path, "x") as f:
                            json.dump(best_params_flat, f, indent=2, sort_keys=True)
                        break
        
        @staticmethod
        def _deep_update(base, updates):
            """
            Recursively update a nested dictionary with another dictionary.
            
            Args:
                base (Dict): The original dictionary to be updated.
                updates (Dict): The dictionary containing updates, which can be nested.
            """
            for k, v in updates.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    OptunaOptimizer._deep_update(base[k], v)
                else:
                    base[k] = v

        @staticmethod
        def apply_to(
            base_params: Dict,
            optimized_params: Dict,
            verbose: bool = False,
        ) -> Dict:
            """
            Apply optimized parameters to a base parameter dictionary.

            Args:
                base_params (Dict): The base parameter dictionary to update.
                optimized_params (Dict): The optimized parameters to apply, which can be a nested dictionary.
                verbose (bool): Whether to print the updated parameters (default: ``False``).

            Returns:
                Dict: The updated parameter dictionary with the optimized parameters applied.
            """

            OptunaOptimizer._deep_update(base_params, optimized_params)

            if verbose:
                print("\nApplying optimized hyperparameters:")
                for k, v in optimized_params.items():
                    print(f"    {k}: {v}")

            return base_params

        @staticmethod
        def load_best_params(
            base_params: Dict,
            json_path: str,
            verbose: bool = True,
        ) -> Dict:
            """
            Load best hyperparameters from a JSON file (flat Optuna format), unflatten them and apply them to a base parameter dictionary.

            Args:
                base_params (Dict): The base parameter dictionary to update.
                json_path (str): The path to the JSON file containing the best parameters in flat format.
                verbose (bool): Whether to print the updated parameters (default: ``True``).

            Returns:
                Dict: The updated parameter dictionary with the best parameters applied.
            """
            with open(json_path) as f:
                flat_params = json.load(f)

            nested_params = OptunaOptimizer._unflatten_dict(flat_params)
            OptunaOptimizer.apply_to(base_params, optimized_params=nested_params, verbose=verbose)
            return base_params


    def run_trial(
        trial: "optuna.Trial",
        optuna_optimizer: OptunaOptimizer,
        build_model: Callable[[Dict], torch.nn.Module],
        training_params: Dict,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        default_epochs: int = 100,
    ) -> float:
        r"""
        Train the models of one trial of ``create_optimized_model`` and return the value of the trial: the evaluation loss of each
        model, averaged over the seeds of ``optuna_optimizer.seeds`` (one model per seed, all with the same hyperparameters), or the
        evaluation loss of a single model trained with the ``seed`` of ``training_params`` if the optimizer has no seeds.

        The evaluation loss of a model is the one of its last epoch or, if ``save_best`` is ``True`` in ``training_params``, the lowest
        one of its training, which is the one of the model that ``save_best`` keeps. ``early_stopping_patience`` stops each model on
        its own, as in ``fit``.

        - With a pruner, the models are trained one epoch at a time and at the same pace, and the mean over the seeds of the current
          value of each model is reported to the pruner every epoch (a model already stopped keeps its value). Each model keeps its
          own random state, so that it is trained as if it were trained alone.
        - Without a pruner, the models are trained one after the other.

        The trial is pruned if the loss of any of its models is not finite (with ``save_best``, only if none of its evaluation losses
        is finite), or if it runs out of memory. The number of epochs trained is stored in the user attribute ``epochs_trained`` of
        the trial (with seeds, a list with one value per seed, and also the user attributes ``seeds`` and ``seed_values``, the value
        of each model).

        Args:
            trial (optuna.Trial): The trial.
            optuna_optimizer (OptunaOptimizer): The optimizer of the study, which gives the pruner and the seeds.
            build_model (Callable): Function that builds a new model from a dictionary of parameters, which includes its ``seed``.
            training_params (Dict): The parameters of the trial, passed both to ``build_model`` and to the ``fit`` method of the model.
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            default_epochs (int): The number of epochs if ``training_params`` has no ``epochs`` (default: ``100``).

        Returns:
            float: The value of the trial.
        """
        params = dict(training_params)
        params["save_logs_path"] = None
        
        # Only the loss of the best epoch is needed here, not its model
        best = params.pop("save_best", False)
        patience = params.get("early_stopping_patience")
        epochs = params.get("epochs", default_epochs)
        multi_seed = optuna_optimizer.seeds is not None
        if multi_seed:
            if params.get("seed") is not None and "seed_ignored" not in _ISSUED_WARNINGS:
                _ISSUED_WARNINGS.add("seed_ignored")
                raiseWarning(f"The seed {params['seed']} of the optimization parameters is ignored: each trial trains one model per seed of {optuna_optimizer.seeds}")
            seeds = optuna_optimizer.seeds
        else:
            seeds = [params.get("seed")]

        def model_value(run):
            train_losses = np.asarray(run["train"], dtype=float)
            test_losses = np.asarray(run["test"], dtype=float)
            if best:
                # The model that save_best keeps is the one of the best epoch, even if the training diverges later
                finite = test_losses[np.isfinite(test_losses)]
                if finite.size > 0:
                    return float(finite.min())
            elif test_losses.size > 0 and train_losses.size > 0 and np.isfinite(test_losses[-1]) and np.isfinite(train_losses[-1]):
                return float(test_losses[-1])
            return None

        def check_values(runs):
            for run in runs:
                if model_value(run) is None:
                    seed_info = f" with seed {run['seed']}" if multi_seed else ""
                    pprint(0, f"Trial {trial.number} diverged (non-finite loss{seed_info}). Pruning the trial.")
                    raise TrialPruned()
            return [model_value(run) for run in runs]

        runs = []
        _empty_cuda_cache()
        
        try:
            if optuna_optimizer.pruner is not None:
                for seed in seeds:
                    run_params = {**params, "seed": seed}
                    model = build_model(run_params)
                    # Random state right after the model is built (and seeded), the one its training starts from
                    runs.append({
                        "seed": seed, "model": model, "rng": _get_rng_state(), "train": [], "test": [], "active": True,
                        # One epoch per call to fit: the early stopping is applied here, with the same EarlyStopper as fit
                        "params": {**run_params, "epochs": 1, "early_stopping_patience": None},
                        "stopper": EarlyStopper(patience=patience) if patience is not None else None,
                    })
                # The models are evaluated every epoch here: early stopping and save_best only match fit if fit also evaluates every epoch
                fit_signature = inspect.signature(type(runs[0]["model"]).fit).parameters
                if "print_eval_rate" in fit_signature:
                    eval_rate = params.get("print_eval_rate", fit_signature["print_eval_rate"].default)
                    if eval_rate != 1 and (best or patience is not None) and "eval_rate" not in _ISSUED_WARNINGS:
                        _ISSUED_WARNINGS.add("eval_rate")
                        raiseWarning(f"With a pruner, the models of each trial are trained one epoch at a time and evaluated every epoch, while fit evaluates them "
                                     f"every print_eval_rate={eval_rate} epochs: early_stopping_patience and save_best count every epoch in the trials. "
                                     f"Use print_eval_rate=1 so that the trials and the final training stop and keep the same epochs")
                for epoch in range(epochs):
                    for run in runs:
                        if not run["active"]:
                            continue
                        _set_rng_state(run["rng"])
                        results = run["model"].fit(train_dataset, eval_dataset, **run["params"])
                        run["rng"] = _get_rng_state()
                        run["train"].extend(np.asarray(results["train_loss"], dtype=float).tolist())
                        run["test"].extend(np.asarray(results["test_loss"], dtype=float).tolist())
                        if not (np.isfinite(run["train"][-1]) and len(run["test"]) > 0 and np.isfinite(run["test"][-1])):
                            run["active"] = False
                        elif run["stopper"] is not None:
                            prev_train = run["train"][-2] if len(run["train"]) > 1 else math.inf
                            run["active"] = not run["stopper"].early_stop(run["test"][-1], prev_train, run["train"][-1])
                    values = check_values(runs)
                    trial.report(float(np.mean(values)), epoch)
                    if trial.should_prune():
                        raise TrialPruned()
                    if not any(run["active"] for run in runs):
                        break
            else:
                for seed in seeds:
                    run_params = {**params, "seed": seed}
                    model = build_model(run_params)
                    try:
                        results = model.fit(train_dataset, eval_dataset, **run_params)
                    finally:
                        del model
                        _empty_cuda_cache()
                    runs.append({"seed": seed, "train": list(results["train_loss"]), "test": list(results["test_loss"])})
                    # A diverged model prunes the trial without training the remaining seeds
                    check_values(runs[-1:])
            
            values = check_values(runs)

        except RuntimeError as e:
            message = str(e).lower()
            if "out of memory" in message or "allocate memory" in message:
                pprint(0, f"Trial {trial.number} failed due to out of memory error. Pruning the trial.")
                raise TrialPruned()
            raise

        finally:
            for run in runs:
                run.pop("model", None)
            _empty_cuda_cache()

        epochs_trained = [len(run["train"]) for run in runs]
        value = float(np.mean(values))
        if multi_seed:
            trial.set_user_attr("epochs_trained", epochs_trained)
            trial.set_user_attr("seeds", list(seeds))
            trial.set_user_attr("seed_values", values)
            pprint(0, f"Trial {trial.number}: mean value {value:.4e} over the seeds {list(seeds)} "
                      f"({', '.join(f'{v:.4e}' for v in values)}), epochs trained {epochs_trained}")
        else:
            trial.set_user_attr("epochs_trained", epochs_trained[0])
        return value


    class OptunaStudyManager():
        """
        Manager class for Optuna studies stored in a database.
        
        Args:
            storage (str): The storage URL to use for accessing studies. Example: 'sqlite:///path/to/optuna_studies.db'
        """
        def __init__(self, storage: str):
            if not storage.startswith('sqlite:///'):
                storage = f'sqlite:///{storage}'
            self.storage = storage

        def storage_file_exists(self) -> bool:
            """
            Check if the storage file exists for the given storage URL.
            
            Returns:
                bool: True if the storage file exists, False otherwise.
            """
            db_path = self.storage.replace("sqlite:///", "", 1)
            return os.path.isfile(db_path)
        
        def study_exists(self, study_name: str) -> bool:
            """
            Check if a study with the given name exists in the database.
            
            Args:
                study_name (str): The name of the study to check.
                
            Returns:
                bool: True if the study exists, False otherwise.
            """
            if not self.storage_file_exists():
                return False
            
            try:
                optuna.load_study(study_name=study_name, storage=self.storage)
                return True
            except KeyError:
                return False
            except Exception as e:
                raiseError(f"Error checking study existence: {str(e)}")
                return False

        def get_study_names(self) -> list:
            """
            Get a list of all study names available in the database.
            
            Returns:
                list: A list of study names.
            """
            if not self.storage_file_exists():
                return []
        
            try:
                study_summaries = optuna.get_all_study_summaries(storage=self.storage)
                return [summary.study_name for summary in study_summaries]
            except Exception as e:
                raiseError(f"Error retrieving study names: {str(e)}")
                return []

        def list_studies(self) -> None:
            """
            List all available studies in the database with their summary statistics.
            """
            try:
                study_summaries = optuna.get_all_study_summaries(storage=self.storage)
                
                if not study_summaries:
                    pprint(0, "No studies found in the database.")
                    return
                
                pprint(0, "\n" + "="*80)
                pprint(0, "AVAILABLE STUDIES")
                pprint(0, "="*80)
                
                for i, summary in enumerate(study_summaries, 1):
                    pprint(0, f"\n{i}. Study: {summary.study_name}")
                    pprint(0, f"   Direction: {summary.direction.name}")
                    pprint(0, f"   Number of trials: {summary.n_trials}")
                    
                    if summary.best_trial is not None:
                        pprint(0, f"   Best value: {summary.best_trial.value}")
                        pprint(0, f"   Best trial number: {summary.best_trial.number}")
                    else:
                        pprint(0, "   Best value: N/A (no completed trials)")
                    
                    pprint(0, f"   Created at: {summary.datetime_start}")
                
            except Exception as e:
                raiseError(f"Error listing studies: {str(e)}")

        def get_study_trials(self, study_name: str) -> Dict[str, int]:
            """
            Get the trials of a study and their states.
            
            Args:
                study_name (str): The name of the study to check.
                
            Returns:
                Dict[str, int]: A dictionary with trial states as keys and the trial as values.
            """
            try:
                if not self.study_exists(study_name):
                    pprint(0, f"Study '{study_name}' not found in optuna database.")
                    return {}
                
                study = optuna.load_study(study_name=study_name, storage=self.storage)
                pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
                complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
                failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
                
                return {
                    "pruned": pruned_trials,
                    "complete": complete_trials,
                    "failed": failed_trials
                }
                
            except Exception as e:
                raiseError(f"Error retrieving study trials: {str(e)}")
                return {}
        
        def study_summary(self, study_name: str) -> None:
            """
            Print a detailed summary of a specific study.
            
            Args:
                study_name (str): The name of the study to summarize.
            """
            try:
                if not self.study_exists(study_name):
                    pprint(0, f"Study '{study_name}' not found in optuna database.")
                    return
                
                study = optuna.load_study(study_name=study_name, storage=self.storage)
                study_trials = self.get_study_trials(study_name)
                
                pruned_trials = study_trials.get("pruned", [])
                complete_trials = study_trials.get("complete", [])
                failed_trials = study_trials.get("failed", [])
                
                pprint(0, "\n" + "="*80)
                pprint(0, f"STUDY SUMMARY: {study_name}")
                pprint(0, "="*80)
                pprint(0, f"\nDirection: {study.direction.name}")
                pprint(0, f"Total trials: {len(study.trials)}")
                pprint(0, f"  - Completed: {len(complete_trials)}")
                pprint(0, f"  - Pruned: {len(pruned_trials)}")
                pprint(0, f"  - Failed: {len(failed_trials)}")
                
                if study.best_trial is not None:
                    pprint(0, f"\nBest trial:")
                    pprint(0, f"  Trial number: {study.best_trial.number}")
                    pprint(0, f"  Value: {study.best_trial.value}")
                    pprint(0, f"  Parameters:")
                    for key, value in study.best_trial.params.items():
                        pprint(0, f"    {key}: {value}")
                else:
                    pprint(0, "\nNo completed trials available.")
                
            except Exception as e:
                raiseError(f"Error getting study summary: {str(e)}")
        
        def delete_study(self, study_name: str, confirm: bool = True) -> None:
            """
            Delete a study from the database.
            
            Args:
                study_name (str): The name of the study to delete.
                confirm (bool): Whether to ask for confirmation before deleting. Default is True.
            """
            try:
                if not self.study_exists(study_name):
                    pprint(0, f"Study '{study_name}' not found in optuna database.")
                    return
                
                if confirm:
                    pprint(0, f"\nAre you sure you want to delete study '{study_name}'?")
                    response = input("Type 'yes' to confirm: ")
                    if response.lower() != 'yes':
                        pprint(0, "Deletion cancelled.")
                        return
                
                optuna.delete_study(study_name=study_name, storage=self.storage)
                pprint(0, f"Study '{study_name}' successfully deleted.")
                
            except Exception as e:
                raiseError(f"Error deleting study: {str(e)}")
        
        def export_best_params(self, study_name: str, output_file: str = None) -> Dict:
            """
            Export the best parameters from a study to a JSON file.
            
            Args:
                study_name (str): The name of the study.
                output_file (str): Path to save the JSON file. If None, returns dict only.
                
            Returns:
                Dict: The best parameters.
            """
            try:
                if not self.study_exists(study_name):
                    pprint(0, f"Study '{study_name}' not found in optuna database.")
                    return {}

                study = optuna.load_study(study_name=study_name, storage=self.storage)
                
                if study.best_trial is None:
                    pprint(0, f"No completed trials in study '{study_name}'.")
                    return {}
                
                best_params = study.best_params
                
                if output_file is not None:
                    output_path = Path(output_file)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump(best_params, f, indent=2, sort_keys=True)
                    
                    pprint(0, f"Best parameters exported to: {output_file}")
                
                return best_params
                
            except Exception as e:
                raiseError(f"Error exporting best parameters: {str(e)}")
        
        def compare_studies(self, study_names: list) -> None:
            """
            Display a comparison of multiple studies side by side.
            
            Args:
                study_names (list): List of study names to compare.
            """
            try:
                pprint(0, "\n" + "="*80)
                pprint(0, "STUDY COMPARISON")
                pprint(0, "="*80 + "\n")
                
                for study_name in study_names:
                    if not self.study_exists(study_name):
                        pprint(0, f"Study '{study_name}' not found in optuna database.")
                        continue
                    
                    study = optuna.load_study(study_name=study_name, storage=self.storage)
                    
                    pprint(0, f"Study: {study_name}")
                    if study.best_trial is not None:
                        pprint(0, f"  Best value: {study.best_trial.value}")
                        pprint(0, f"  Trials: {len(study.trials)}")
                        pprint(0, f"  Best params: ")
                        for key, value in study.best_trial.params.items():
                            pprint(0, f"    {key}: {value}")
                    else:
                        pprint(0, f"  Best value: N/A")
                        pprint(0, f"  Trials: {len(study.trials)}")
                    pprint(0, "")
                
            except Exception as e:
                raiseError(f"Error comparing studies: {str(e)}")

        def is_study_complete(self, study_name: str, expected_trials: int) -> bool:
            """
            Check if a study has completed all its trials.
            
            Args:
                study_name (str): The name of the study to check.
                expected_trials (int): The total number of trials expected for the study.
                
            Returns:
                bool: True if the study has completed all expected trials, False otherwise.
            """
            if not self.study_exists(study_name):
                    return False
            
            try:         
                study_trials = self.get_study_trials(study_name)
                finished_trials = len(study_trials["complete"]) + len(study_trials["pruned"]) + len(study_trials["failed"])
                return finished_trials >= expected_trials
                
            except Exception as e:
                raiseError(f"Error checking study completion: {str(e)}")
                return False

        def remaining_trials(self, study_name: str, expected_trials: int) -> int:
            """
            Get the number of remaining trials for a study to complete.
            
            Args:
                study_name (str): The name of the study to check.
                expected_trials (int): The total number of trials expected for the study.
            
            Returns:
                int: The number of remaining trials to complete the study.
            """
            try:
                if self.study_exists(study_name):
                    study_trials = self.get_study_trials(study_name)
                    finished_trials = len(study_trials["complete"]) + len(study_trials["pruned"]) + len(study_trials["failed"])
                else:
                    finished_trials = 0
            
                return max(0, expected_trials - finished_trials)
                
            except Exception as e:
                raiseError(f"Error checking remaining trials: {str(e)}")
                return expected_trials

except:
    def TrialState():
        raiseError("Package optuna should be installed")
    
    def TrialPruned():
        raiseError("Package optuna should be installed")

    def run_trial(*args, **kwargs):
        raiseError("Package optuna should be installed")

    class OptunaOptimizer():
        """
        Args:
            optimization_params (Dict): A dictionary containing the parameters to optimize.
            n_trials (int): The number of trials to run. Default is ``100``.
            direction (str): The direction to optimize. Can be 'minimize' or 'maximize'. Default is ``'minimize'``.
            pruner (optuna.pruners.BasePruner): The pruner to use. Default is ``None``.
            save_dir (str): The directory to save the best parameters. Default is ``None``.
        """
        def __init__(self,*args,**kwargs):
            raiseError("Package optuna should be installed")

    class OptunaStudyManager():
        """
        Manager class for Optuna studies stored in a database.
        
        Args:
            storage (str): The storage URL to use for accessing studies. Example: 'sqlite:///path/to/optuna_studies.db'
        """
        def __init__(self,*args,**kwargs):
            raiseError("Package optuna should be installed")