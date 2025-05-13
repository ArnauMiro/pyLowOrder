NON_CONVERGED_REWARD = -100000

from airfoil_solvers import (
    BaseSolver,
    XFoilSolver,
    NeuralFoilSolver,
    DummySolver
)

from shape_parameterizers import BaseParameterizer, AirfoilCSTParametrizer
from env_factory import create_env, AirfoilOperatingConditions, AirfoilParameterizerConfig, SolverFactory
