from abc import ABC, abstractmethod

import numpy as np
import aerosandbox as asb
from ._xfoil_wrapper import XFoil
import neuralfoil as nf

from pyLOM.RL import NON_CONVERGED_REWARD
from pyLOM.utils import raiseWarning, pprint


class BaseSolver(ABC):
    """
    Base class for all solvers. It defines the interface for the solvers.
    """
    def __init__(self):
        self.NON_CONVERGED_REWARD = NON_CONVERGED_REWARD

    @abstractmethod
    def __call__(self, shape):
        """
        Compute the reward for the given shape.
        Args:
            shape: Shape to compute the reward for.
        Returns:
            float: Reward for the shape.
        """
        pass


class XFoilSolver(BaseSolver):
    """
    XFoil solver for airfoil analysis. It uses the XFoil library to compute the lift and drag coefficients. Under the hood, it uses the XFoil class from aerosandbox.
    Requires XFoil to be on installed your machine; XFoil is available here: https://github.com/RobotLocomotion/xfoil/
    And xfoil to be in your PATH. You can check this by running `which xfoil` in your terminal.
    
    Args:
        alpha (float): Angle of attack in degrees.
        mach (float): Mach number.
        Reynolds (float, optional): Reynolds number. If not provided, inviscid mode is used. Default is ``None``.
        check_selfintersection (bool, optional): If True, check for self-intersection of the airfoil when computing lift/drag. If the airfoil self-intersects, XFoil will return NON_CONVERGED_REWARD. Default is ``False``.
        xfoil_params (dict, optional): Additional parameters that are passed to the XFoil class. For example, you can set `xfoil_params={"xfoil_repanel": True}` to repanel the airfoil. Default is ``{}``.
    """

    def __init__(self, alpha: float, mach: float, Reynolds: float = None, check_selfintersection: bool = False, xfoil_params: dict = {}):
        super().__init__()
        self.Reynolds = Reynolds
        self.alpha = alpha
        self.mach_number = mach
        self.viscous = Reynolds is not None
        if not self.viscous:
            raiseWarning("CD prediction of inviscid xfoil is poor, note that the results will not be too accurate.")
        self.check_selfintersection = check_selfintersection
        self.xfoil_params = xfoil_params

    def __call__(self, airfoil: asb.Airfoil):
        if self.check_selfintersection and self._check_if_self_intersecting(airfoil):
            pprint(0, "Airfoil self-intersecting")
            return self.NON_CONVERGED_REWARD
        
        base_params = {
            "Re": 0 if not self.viscous else self.Reynolds, # 0 means inviscid mode 
            "mach": self.mach_number,
            "xfoil_repanel": False,
            "max_iter": 200,
            "timeout": 30,
        }
        base_params.update(self.xfoil_params)
        xfoil = XFoil(
            airfoil=airfoil.repanel(n_points_per_side=100 if "xfoil_repanel_n_points" not in base_params else base_params["xfoil_repanel_n_points"]),
            **base_params
        )
        try:
            result = xfoil.alpha(self.alpha)
            lift_drag_ratio = result["CL"] / result["CD"] # cd predition of inviscid xfoil is poor
            if len(lift_drag_ratio) == 0 or np.isnan(lift_drag_ratio) or np.isinf(lift_drag_ratio):
                return self.NON_CONVERGED_REWARD
            # [0] is because xfoil returns an array
            return lift_drag_ratio[0]

        except Exception as e:
            pprint(0, "XFoil failed", e)
            return self.NON_CONVERGED_REWARD
    
    def _check_if_self_intersecting(self, airfoil, threshold=0.0005):
        coordinates_thickness = airfoil.local_thickness()
        # remove leading and trailing zeros from leading edge and trailing edge if they are present
        if coordinates_thickness[0] == 0:
            coordinates_thickness = coordinates_thickness[1:]
        if coordinates_thickness[-1] == 0:
            coordinates_thickness = coordinates_thickness[:-1]

        return np.any(coordinates_thickness < threshold)


class NeuralFoilSolver(BaseSolver):
    """
    NeuralFoil solver for airfoil analysis. It uses the NeuralFoil library to compute the lift and drag coefficients.

    Args:
        alpha (float): Angle of attack in degrees.
        Reynolds (float): Reynolds number.
        model_size (str, optional): Size of the model. Default is "xxxlarge". Other options are "xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge".
    """
    def __init__(self, alpha: float, Reynolds: float, model_size: str = "xxxlarge"):
        super().__init__()
        self.Reynolds = Reynolds
        self.alpha = alpha
        self.model_size = model_size

    def __call__(self, airfoil: asb.Airfoil):
        params = dict(
            airfoil=airfoil,
            alpha=self.alpha,
            Re=self.Reynolds,
            model_size=self.model_size,
        )

        result = nf.get_aero_from_airfoil(**params)
        confidence = result["analysis_confidence"][0]
        reward = (result["CL"][0] / result["CD"][0]) * confidence
        return reward
    

class DummySolver(BaseSolver):
    """
    Dummy solver for airfoil analysis. It returns a constant reward for all airfoils.
    This is useful for testing purposes.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, shape):
        return 0