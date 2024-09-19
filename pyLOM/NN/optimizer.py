from typing import Callable, Dict
import json

import optuna
from optuna.trial import TrialState


class OptunaOptimizer():
    """
    Args:
        optimization_params (Dict): A dictionary containing the parameters to optimize.
        n_trials (int): The number of trials to run. Default is ``100``.
        direction (str): The direction to optimize. Can be 'minimize' or 'maximize'. Default is ``'minimize'``.
        pruner (optuna.pruners.BasePruner): The pruner to use. Default is ``None``.
        save_dir (str): The directory to save the best parameters. Default is ``None``.
    """
    def __init__(
        self,
        optimization_params: Dict,
        n_trials: int = 100,
        direction: str = 'minimize',
        pruner: optuna.pruners.BasePruner = None,
        save_dir: str = None
    ):
        self.num_trials = n_trials
        self.direction = direction
        self.pruner = pruner
        self._optimization_params = optimization_params
        self.save_dir = save_dir

    @property
    def optimization_params(self) -> Dict:
        """
        Get the optimization parameters.
        """
        return self._optimization_params

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
        study = optuna.create_study(direction=self.direction, pruner=self.pruner)
        study.optimize(objective_function, n_trials=self.num_trials)
        if self.save_dir is not None:
            file_path = str(self.save_dir) + '/best_params.json'
            json.dump(study.best_params, open(file_path, 'w'))

        self._print_optimization_report(study)
        
        return study.best_params
    
    def _print_optimization_report(self, study):
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        
        # Optimization report
        print("\nStudy statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of completed trials: ", len(completed_trials))

        trial = study.best_trial
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print("\n")