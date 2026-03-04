#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN optimizer routines using optuna.
#
# Last rev: 02/10/2024

import json

from typing         import Callable, Dict
from pathlib        import Path
from itertools      import count
from ..utils.errors import raiseError
from ..             import pprint

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
        """
        def __init__(
            self,
            optimization_params: Dict,
            n_trials: int = 100,
            direction: str = 'minimize',
            pruner: optuna.pruners.BasePruner = None,
            save_dir: str = None,
            storage: str = None,
            study_name: str = None,
            load_if_exists: bool = True,
        ):
            self.num_trials = n_trials
            self.direction = direction
            self.pruner = pruner
            self._optimization_params = optimization_params
            self.save_dir = save_dir
            self.storage = storage
            self.study_name = study_name
            self.load_if_exists = load_if_exists

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
                    pruner=self.pruner,
                    load_if_exists=self.load_if_exists,
                )
            else:
                study = optuna.create_study(
                    direction=self.direction,
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
            The filename is determined by the study name if provided, or a unique name based on the number of existing files in the directory.
            
            Args:
                best_params_flat (Dict): The best parameters in flat format to save.
            """
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            for i in count():
                if self.study_name is None:
                    file_path = save_dir / f"best_params_{i}.json"
                else:
                    file_path = save_dir / f"{self.study_name}.json"
                
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

except:
    def TrialState():
        raiseError("Package optuna should be installed")
    
    def TrialPruned():
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