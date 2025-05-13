NON_CONVERGED_REWARD = -100000

from pyLOM.RL.airfoil_solvers import (
    BaseSolver,
    XFoilSolver,
    NeuralFoilSolver,
    DummySolver
)
from pyLOM.RL.shape_parameterizers import BaseParameterizer, AirfoilCSTParametrizer
from . import shape_optimization_env 
from pyLOM.RL.env_factory import create_env, AirfoilOperatingConditions, AirfoilParameterizerConfig, SolverFactory
