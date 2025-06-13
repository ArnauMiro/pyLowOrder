from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import aerosandbox as asb

from pyLOM.RL.airfoil_solvers import (
    NeuralFoilSolver,
    XFoilSolver,
    DummySolver,
    BaseSolver
)
from pyLOM.RL.wing_solvers import (
    AerosandboxWingSolver,
    AVLSolver,
)
from pyLOM.RL.shape_parameterizers import AirfoilCSTParametrizer
from pyLOM.RL.shape_parameterizers import WingParameterizer
from pyLOM.utils import raiseError

WING_SOLVER_NAME_TO_CLASS = {
    "aerosandbox": AerosandboxWingSolver,
    "avl": AVLSolver,
    # "dust": DustSolver,
}

@dataclass
class AirfoilOperatingConditions:
    """Configuration for airfoil operating conditions.
    
    This class encapsulates the physical parameters used in airfoil simulations,
    including angle of attack, Reynolds number, and Mach number.
    
    Args:
        alpha (float): Angle of attack in degrees. Default is ``2`` degrees.
        reynolds (float): Reynolds number for the simulation. Default is ``1e6``.
        mach (float): Mach number for the simulation. Default is ``0.5``.
    """
    alpha: float = 2  # degrees
    Reynolds: float = 1e6
    mach: float = 0.5


@dataclass
class WingOperatingConditions:
    """
    Configuration for wing operating conditions
    
    Args:
        velocity (float): Flight velocity in m/s. Default is ``150`` m/s.
        altitude (float): Altitude in meters. Default is ``500`` m.
        alpha (float): Angle of attack in degrees. Default is ``2``.
    """
    velocity: float = 150  # m/s
    altitude: float = 500  # m
    alpha: float = 2  # degrees
    
    def __post_init__(self):
        self.atmosphere = asb.Atmosphere(altitude=self.altitude)


@dataclass
class AirfoilParameterizerConfig:
    """
    Configuration for airfoil parameterization.
    
    This class defines the parameters used to create an airfoil shape using the
    Class-Shape Transformation (CST) method. It controls the number of control
    points and their bounds for both upper and lower surfaces.
    
    Args:
        n_weights_per_side (int): Number of control points per surface side. Default is ``8``.
        leading_edge_weight_bounds (Tuple[float, float]): Min and max values for 
            the leading edge weight parameter. Default is ``(-0.05, 0.75)``.
        te_thickness_bounds (Tuple[float, float]): Min and max values for 
            trailing edge thickness. Default is ``(0.0005, 0.01)``.
        upper_edge_bounds (Tuple[float, float]): Min and max values for 
            upper surface control points. Default is ``(-1.5, 1.25)``.
        lower_edge_bounds (Tuple[float, float]): Min and max values for 
            lower surface control points. Default is ``(-0.75, 1.5)``.
    """
    n_weights_per_side: int = 8
    leading_edge_weight_bounds: Tuple[float, float] = (-0.05, 0.75)
    te_thickness_bounds: Tuple[float, float] = (0.0005, 0.01)
    upper_edge_bounds: Tuple[float, float] = (-1.5, 1.25)
    lower_edge_bounds: Tuple[float, float] = (-0.75, 1.5)
    
    def create_parameterizer(self):
        """
        Creates an AirfoilCSTParametrizer based on this configuration.
        
        This method instantiates a new AirfoilCSTParametrizer using the 
        current configuration settings. The parameterizer can then be used
        to generate airfoil shapes for optimization.
        
        Returns:
            AirfoilCSTParametrizer: A configured airfoil parameterizer instance.
        """
        return AirfoilCSTParametrizer(
            upper_surface_bounds=(
                [self.upper_edge_bounds[0]] * self.n_weights_per_side,
                [self.upper_edge_bounds[1]] * self.n_weights_per_side,
            ),
            lower_surface_bounds=(
                [self.lower_edge_bounds[0]] * self.n_weights_per_side,
                [self.lower_edge_bounds[1]] * self.n_weights_per_side,
            ),
            TE_thickness_bounds=self.te_thickness_bounds,
            leading_edge_weight=self.leading_edge_weight_bounds,
        )


@dataclass
class WingParameterizerConfig:
    """
    Configuration for wing parameterization
    
    This class defines the parameters used to create a wing shape, including
    airfoil type, chord, twist, span, sweep, and dihedral angles. It provides
    methods to create a WingParameterizer instance based on these settings.

    Args:
        airfoil_name (str): Name of an airfoil from the UIUC airfoil dataset to use for the wing. If a custom `asb.Airfoil` wants to me used, pleas create the parameterizere directly with `pyLOM.RL.WingParameterizer`. Default is "naca0012".
        chord_bounds (List[List[float]]): Bounds for the chord length. Default is [[0.75, 0.45], [1.25, 0.75]].
        twist_bounds (List[List[float]]): Bounds for the twist angle in degrees. Default is [[-2, -2], [2, 2]].
        span_bounds (List[List[float]]): Bounds for the span length. Default is [[1.5], [2]].
        sweep_bounds (List[List[float]]): Bounds for the sweep angle in degrees. Default is [[-5], [15]].
        dihedral_bounds (List[List[float]]): Bounds for the dihedral angle in degrees. Default is [[-2], [7]].
    """
    airfoil_name: str = "naca0012"
    chord_bounds: List[List[float]] = field(default_factory=lambda: [[0.75, 0.45], [1.25, 0.75]])
    twist_bounds: List[List[float]] = field(default_factory=lambda: [[-2, -2], [2, 2]])
    span_bounds: List[List[float]] = field(default_factory=lambda: [[1.5], [2]])
    sweep_bounds: List[List[float]] = field(default_factory=lambda: [[-5], [15]])
    dihedral_bounds: List[List[float]] = field(default_factory=lambda: [[-2], [7]])
    
    def create_parameterizer(self):
        """Creates a WingParameterizer based on this configuration"""
        return WingParameterizer(
            airfoil=asb.Airfoil(self.airfoil_name),
            chord_bounds=self.chord_bounds,
            twist_bounds=self.twist_bounds,
            span_bounds=self.span_bounds,
            sweep_bounds=self.sweep_bounds,
            dihedral_bounds=self.dihedral_bounds,
        )


class SolverRegistry:
    """Registry of available solvers.
    
    This class maintains a registry of available solver types for the simulation
    environment. It provides methods to register new solvers and check if a 
    given solver name is valid.
    """
    _airfoil_solvers = {'neuralfoil', 'xfoil', 'dummy'}
    _wing_solvers = {'avl', 'aerosandbox'}
    
    @classmethod
    def register_airfoil_solver(cls, name: str) -> None:
        """Register a new airfoil solver in the registry.
        
        This method adds a new solver name to the list of available airfoil solvers.
        
        Args:
            name (str): The name of the solver to register.
        """
        cls._airfoil_solvers.add(name.lower())
        
    @classmethod
    def is_airfoil_solver(cls, name: str) -> bool:
        """Check if a solver is a registered airfoil solver.
        
        This method verifies if the given solver name exists in the registry
        of airfoil solvers.
        
        Args:
            name (str): The name of the solver to check.
            
        Returns:
            bool: True if the solver is registered, False otherwise.
        """
        return name.lower() in cls._airfoil_solvers
    
    @classmethod
    def is_wing_solver(cls, name: str) -> bool:
        """Check if a solver is a wing solver
        
        This method verifies if the given solver name exists in the registry
        of wing solvers.
        Args:
            name (str): The name of the solver to check.
        
        Returns:
            bool: True if the solver is registered as a wing solver, False otherwise."""
        return name.lower() in cls._wing_solvers

    @classmethod
    def get_all_solvers(cls) -> set:
        """Get all registered solvers.
        
        This method returns the set of all registered solver names.
        
        Returns:
            set: A set containing all registered solver names.
        """
        return cls._airfoil_solvers


class SolverFactory:
    """Factory for creating solver instances.
    
    This class provides methods to create appropriate solver instances
    based on their name and configuration parameters. It acts as a factory
    that abstracts the details of solver instantiation.
    """
    
    @staticmethod
    def create_solver(solver_name: str, conditions: Optional[AirfoilOperatingConditions] = None) -> BaseSolver:
        """Create a solver by name.
        
        This method determines the type of solver (airfoil or wing) based on the name
        and delegates creation to the appropriate specialized factory method.
        
        Args:
            solver_name (str): The name of the solver to create.
            conditions (Optional[AirfoilOperatingConditions]): Configuration parameters 
                for the solver. If None, default values will be used.
                
        Returns:
            BaseSolver: The created solver instance.
            
        Raises:
            ValueError: If the solver name is not recognized.
        """
        solver_name = solver_name.lower()
        
        if SolverRegistry.is_airfoil_solver(solver_name):
            return SolverFactory.create_airfoil_solver(solver_name, conditions)
        elif SolverRegistry.is_wing_solver(solver_name):
            return SolverFactory.create_wing_solver(solver_name, conditions)
        else:
            raise ValueError(
                f"Solver {solver_name} not recognized. Available solvers: {SolverRegistry.get_all_solvers()}"
            )
    
    @staticmethod
    def create_airfoil_solver(solver_name: str, conditions: Optional[AirfoilOperatingConditions] = None) -> BaseSolver:
        """Create an airfoil solver instance.
        
        This method creates and returns an airfoil solver instance based on the
        specified name and configured with the given conditions.
        
        Args:
            solver_name (str): The name of the airfoil solver to create.
            conditions (Optional[AirfoilOperatingConditions]): Configuration parameters 
                for the solver. If None, default values will be used.
                
        Returns:
            BaseSolver: The created airfoil solver instance.
            
        Raises:
            ValueError: If the solver name is not recognized as a valid airfoil solver.
        """
        if conditions is None:
            conditions = AirfoilOperatingConditions()
            
        if solver_name == "neuralfoil":
            return NeuralFoilSolver(
                alpha=conditions.alpha,
                Reynolds=conditions.Reynolds,
                model_size="xxsmall",
            )
        elif solver_name == "xfoil":
            return XFoilSolver(
                alpha=conditions.alpha,
                Reynolds=conditions.Reynolds,
                mach=conditions.mach,
            )
        elif solver_name == "dummy":
            return DummySolver()
        else:
            raise raiseError(f"Solver {solver_name} not recognized")
    
    @staticmethod
    def create_wing_solver(solver_name: str, conditions: Optional[WingOperatingConditions] = None):
        """Create a wing solver"""
        if conditions is None:
            conditions = WingOperatingConditions()
            
        solver_name = solver_name.lower()
        
        if solver_name in WING_SOLVER_NAME_TO_CLASS:
            solver_class = WING_SOLVER_NAME_TO_CLASS[solver_name]
            return solver_class(
                velocity=conditions.velocity,
                alpha=conditions.alpha,
                atmosphere=conditions.atmosphere,
            )
        else:
            raise raiseError(f"Solver {solver_name} not recognized")


def create_env(
    solver_name,
    parameterizer=None,
    operating_conditions=None,
    num_envs=1,
    episode_max_length=64,
    thickness_penalization_factor=0,
    initial_seed=None
):
    """
    Create a reinforcement learning environment for shape optimization. Using these environments, the RL agents have been trained in https://arxiv.org/pdf/2505.02634.

    Args:
        solver_name (str): Name of the solver to use.
        parameterizer: Parameterizer object that defines the shape to optimize (if None, a default will be used)
        num_envs (int): Number of parallel environments to create. If greater than 1, you need to wrap the code in ``if __name__ == "__main__":`` to avoid issues with multiprocessing. Ref: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html. Default is ``1``.
        episode_max_length (int): Maximum episode length. Default is ``64``.
        operating_conditions (Optional[AirfoilOperatingConditions]): Operating conditions for the solver. Default is ``None``.
        thickness_penalization_factor (float): Penalty factor for thickness changes. Default is ``0``.
        initial_seed (Optional[int]): Initial random seed. Default is ``None``.
        
    Returns:
        gym.Env: The created environment
    """
    solver_name = solver_name.lower()
    
    # Handle parallel environments
    if num_envs > 1:
        def make_env(seed):
            def _init():
                return create_env(
                    solver_name=solver_name,
                    parameterizer=parameterizer,
                    operating_conditions=operating_conditions,
                    num_envs=1,
                    episode_max_length=episode_max_length,
                    thickness_penalization_factor=thickness_penalization_factor,
                    initial_seed=seed,
                )
            return _init
            
        envs_fn = [make_env(i) for i in range(num_envs)]
        env = SubprocVecEnv(envs_fn, start_method='spawn')
        return VecMonitor(env)
    
    solver = SolverFactory.create_solver(solver_name, operating_conditions)
    if parameterizer is None:
        if SolverRegistry.is_airfoil_solver(solver_name):
            parameterizer = AirfoilParameterizerConfig().create_parameterizer()
        elif SolverRegistry.is_wing_solver(solver_name):
            parameterizer = WingParameterizerConfig().create_parameterizer()
        else:
            raise ValueError(f"Solver {solver_name} not recognized")
    
    # Create the environment
    env_args = dict(
        solver=solver,
        parameterizer=parameterizer,
        episode_max_length=episode_max_length,
        thickness_penalization_factor=thickness_penalization_factor,
        # disable_env_checker=True
    )
    if initial_seed is not None:
        env_args['seed'] = initial_seed
    env = gym.make("ShapeOptimizationEnv-v0", **env_args)
    return env