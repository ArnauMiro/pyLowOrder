NON_CONVERGED_REWARD = -100000

from pyLOM.RL.airfoil_solvers import (
    BaseSolver,
    XFoilSolver,
    NeuralFoilSolver,
    DummySolver
)
from pyLOM.RL.wing_solvers import (
    AerosandboxWingSolver,
    AVLSolver
)
from pyLOM.RL.shape_parameterizers import BaseParameterizer, AirfoilCSTParametrizer, WingParameterizer
from . import shape_optimization_env
from pyLOM.RL.env_factory import (
    create_env,
    AirfoilOperatingConditions,
    AirfoilParameterizerConfig,
    SolverFactory,
    WingOperatingConditions,
    WingParameterizerConfig,
)
from pyLOM.RL.evaluations import (
    run_episode,
    evaluate_airfoil_agent,
    evaluate_airfoil_agent_whole_uiuc,
    evaluate_airfoil_agent_whole_uiuc_mpi,
)
from pyLOM.RL.plotting import create_airfoil_optimization_progress_plot, AirfoilEvolutionAnimation, WingEvolutionAnimation

from stable_baselines3 import PPO as SB3_PPO