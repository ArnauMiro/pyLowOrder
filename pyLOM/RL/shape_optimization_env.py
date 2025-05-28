import gymnasium as gym
import numpy as np

from pyLOM.RL import BaseParameterizer, BaseSolver
from pyLOM.utils import pprint

class ShapeOptimizationEnv(gym.Env):
    """
    Custom Environment that follows gym interface for shape optimization using reinforcement learning.
    
    Args:
        solver (BaseSolver): Solver object that computes the reward.
        parameterizer (BaseParameterizer): Parameterizer object that defines the shape to optimize.
        episode_max_length (int): Maximum length of an episode, i.e, maximum number of modifications that the agent can perform on the shape. Default is ``64``.
        thickness_penalization_factor (float): Penalty factor for thickness changes. Default is ``0``.
        seed (int, optional): Seed for random number generation. Default is ``None``.
    """

    def __init__(
        self,
        solver: BaseSolver,
        parameterizer: BaseParameterizer,
        episode_max_length: int = 64,
        thickness_penalization_factor:float = 0,
        seed=None
    ):
        self.solver = solver
        self.parameterizer = parameterizer
        self.thickness_penalization_factor = thickness_penalization_factor
        self.episode_max_length = episode_max_length

        self.params_bounds = parameterizer.get_optimizable_bounds()
        self.observation_space = gym.spaces.Box(
            low=np.array(self.params_bounds[0]), high=np.array(self.params_bounds[1])
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(self.params_bounds[0]),)
        )
        self.current_iterations = 0
        # maximum change on every cst parameter each iteration
        # Setting the step size like this, enables that after one episode every parameter could have taken all the possible values
        self.max_step_size = [
            (self.params_bounds[1][i] - self.params_bounds[0][i])
            * (1 / episode_max_length)
            for i in range(len(self.params_bounds[0]))
        ]
        self.max_step_size = np.array(self.max_step_size)

        # according to the authors, it should work better with action range [-1, 1]
        self.reward_range = (-np.inf, np.inf)
        self.previous_reward = 0
        if seed is not None:
            self.seed(seed)

    def reset(self, seed=None, options=None):
        self.current_iterations = 0
        if options is not None and "initial_shape" in options:
            initial_shape = options["initial_shape"]
            self.shape_params = self.parameterizer.get_params_from_shape(
                initial_shape
            )
        else:
            self.shape_params = self.parameterizer.generate_random_params(
                seed=seed
            )
            initial_shape = self.parameterizer.get_shape_from_params(self.shape_params)
        self.previous_reward = self.solver(initial_shape)

        if self.thickness_penalization_factor != 0:
            try: 
                self.initial_thickness = initial_shape.max_thickness()
            except Exception as e:
                pprint(0, "Error in computing initial thickness, the shape returned by the parameterizer may no have implemented the max_thickness method: ", e)
                pprint(0, "The agent will not be penalized for changing the thickness")
                self.thickness_penalization_factor = 0
        return self.shape_params, {"shape": initial_shape, "initial_reward": self.previous_reward}

    def step(self, action):
        # modify current airfoil parameters
        truncated = False
        new_params = self.shape_params + (self.max_step_size * action)
        # clip the values to respect the bounds
        self.shape_params = np.clip(
            new_params,
            self.params_bounds[0],
            self.params_bounds[1],
            dtype=np.float32,
        )
        shape = self.parameterizer.get_shape_from_params(self.shape_params)
        if self.thickness_penalization_factor != 0:
            thickness_factor = self._compute_thickness_factor(shape)
        else:
            thickness_factor = 1

        # compute new reward, i.e. lift/drag
        current_reward = self.solver(shape)
        if current_reward == self.solver.NON_CONVERGED_REWARD:
            current_reward = self.previous_reward
            truncated = True

        current_reward *= thickness_factor
        reward = current_reward - self.previous_reward
        self.previous_reward = current_reward
        self.current_iterations += 1
        return (
            self.shape_params,
            reward,
            self.current_iterations >= self.episode_max_length, # a termination condition
            truncated,  # truncated,
            {"shape": shape},
        )

    def render(self, show=False):
        shape = self.parameterizer.get_shape_from_params(self.shape_params)
        plot = (
            shape.draw(backend="matplotlib", show=show)
            if hasattr(shape, "draw")
            else None
        )
        return plot, shape

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def _compute_thickness_factor(self, shape):
        max_thickness = shape.max_thickness()
        thickness_factor = max_thickness / self.initial_thickness
        shrink = self.thickness_penalization_factor
        return np.exp(-shrink * ((thickness_factor - 1) ** 2))  # -1 to center the function in 1 (is symetric)


gym.envs.registration.register(
    id="ShapeOptimizationEnv-v0",
    entry_point="pyLOM.RL.shape_optimization_env:ShapeOptimizationEnv",
)
