#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN optimizer routines using optuna.
#
# Last rev: 02/10/2024

import json

from typing         import Callable, Dict, Optional
from ..utils.errors import raiseError
from ..             import pprint

# Add optuna as an optional dependency
try:
    import optuna
    from optuna.trial      import TrialState
    from optuna.exceptions import TrialPruned
    from optuna.samplers import TPESampler

    class OptunaOptimizer:
        def __init__(
            self,
            optimization_params: Dict,
            n_trials: int = 100,
            direction: str = 'minimize',
            pruner: optuna.pruners.BasePruner = None,
            sampler: optuna.samplers.BaseSampler = None,
            save_dir: str = None,
            seed: int = 42,
            graph_path: Optional[str] = None,
        ):
            self.num_trials = n_trials
            self.direction = direction
            self.pruner = pruner
            self.save_dir = save_dir
            self._optimization_params = optimization_params
            self.sampler = sampler or TPESampler(seed=seed)
            self.graph_path = graph_path



        @property
        def optimization_params(self) -> Dict:
            return self._optimization_params

        def optimize(self, objective_function: Callable[[optuna.Trial], float]) -> Dict:
            study = optuna.create_study(
                direction=self.direction,
                pruner=self.pruner,
                sampler=self.sampler,
            )
            study.optimize(objective_function, n_trials=self.num_trials)

            if self.save_dir is not None:
                file_path = str(self.save_dir) + '/best_params.json'
                json.dump(study.best_params, open(file_path, 'w'))

            self._print_optimization_report(study)
            return study.best_params

        def _print_optimization_report(self, study):
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            pprint(0, "\nStudy statistics: ")
            pprint(0, "  Number of finished trials: ", len(study.trials))
            pprint(0, "  Number of pruned trials: ", len(pruned_trials))
            pprint(0, "  Number of completed trials: ", len(completed_trials))

            trial = study.best_trial
            pprint(0, "Best trial:")
            pprint(0, "  Value: ", trial.value)
            pprint(0, "  Params: ")
            for key, value in trial.params.items():
                pprint(0, f"    {key}: {value}")
            pprint(0, "\n")


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