from .airfoil_solvers import BaseSolver

import aerosandbox as asb
import numpy as np


class AerosandboxWingSolver(BaseSolver):
    """
    Aerosandbox solver for wing analysis. It uses the AeroBuildup component from the Aerosandbox library to compute the lift and drag coefficients. For more details, see https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/index.html#aerosandbox.AeroBuildup
    Args:
        velocity (float): The flight velocity, expressed as a true airspeed. [m/s]
        alpha (float): Angle of attack in degrees.
        atmosphere (asb.Atmosphere): The atmosphere model to use.
        model_size (str, optional): Size of the model. Default is ``xxxlarge``. Other options are "xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge".
    """
    def __init__(self, velocity, alpha, atmosphere, model_size="xxxlarge"):
        super().__init__()
        self.velocity = velocity # velocity: The flight velocity, expressed as a true airspeed. [m/s]
        self.alpha = alpha # in degrees
        self.atmosphere = atmosphere
        self.model_size = model_size

    def __call__(self, wing):
        airplane = asb.Airplane(
            xyz_ref=[0, 0, 0],  # CG location.
            wings=[wing]
        )
        aero = asb.AeroBuildup(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                velocity=self.velocity,
                alpha=self.alpha,
                atmosphere=self.atmosphere
            ),
            model_size=self.model_size
        ).run_with_stability_derivatives()

        reward = aero['CL'] / aero['CD']
        if np.isnan(reward):
            reward = [self.NON_CONVERGED_REWARD]
        return reward[0] # [0] because is an array


class AVLSolver(BaseSolver):
    """
    AVL solver for wing analysis. It uses the AVL (Athena Vortex Lattice) method to compute the lift and drag coefficients. To use this solver, you need to have AVL executable installed and available in your PATH, see https://web.mit.edu/drela/Public/web/avl/
    Args:
        velocity (float): The flight velocity, expressed as a true airspeed. [m/s]
        alpha (float): Angle of attack in degrees.
        atmosphere (asb.Atmosphere): The atmosphere model to use.
    """
    def __init__(self, velocity, alpha, atmosphere):
        super().__init__()
        self.velocity = velocity # velocity: The flight velocity, expressed as a true airspeed. [m/s]
        self.alpha = alpha # in degrees
        self.atmosphere = atmosphere

    def __call__(self, wing):
        airplane = asb.Airplane(
            xyz_ref=[0, 0, 0],  # CG location.
            wings=[wing]
        )
        aero = asb.AVL(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                velocity=self.velocity,
                alpha=self.alpha,
                atmosphere=self.atmosphere
            ),
            timeout=10
        ).run()

        reward = aero['CL'] / aero['CD'] if aero['CD'] != 0 else self.NON_CONVERGED_REWARD
        if np.isnan(reward):
            reward = self.NON_CONVERGED_REWARD
        return reward