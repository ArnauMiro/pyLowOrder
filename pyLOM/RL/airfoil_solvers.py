from abc import ABC, abstractmethod

import numpy as np
import aerosandbox as asb
from aerosandbox import XFoil
import neuralfoil as nf

from pyLOM.RL import NON_CONVERGED_REWARD
from utils import raiseWarning, pprint


class BaseSolver(ABC):
    def __init__(self):
        self.NON_CONVERGED_REWARD = NON_CONVERGED_REWARD

    @abstractmethod
    def __call__(self, shape):
        pass


class XFoilSolver(BaseSolver):
    def __init__(self, alpha: float, mach: float, Reynolds: float = None, xfoil_params: dict = {}):
        super().__init__()
        self.Reynolds = Reynolds
        self.alpha = alpha
        self.mach_number = mach
        self.viscous = Reynolds is not None
        if not self.viscous:
            raiseWarning("CD prediction of inviscid xfoil is poor, note that the results will not be too accurate.")
        self.xfoil_params = xfoil_params

    def __call__(self, airfoil: asb.Airfoil):
        base_params = {
            "Re": 0 if not self.viscous else self.Reynolds, # 0 means inviscid mode 
            "mach": self.mach_number,
            "xfoil_repanel": False,
            "max_iter": 200,
            "timeout": 30,
        }
        base_params.update(self.xfoil_params)
        xfoil = XFoil(
            airfoil=airfoil.repanel(n_points_per_side=255 if "xfoil_repanel_n_points" not in base_params else base_params["xfoil_repanel_n_points"]),
            **base_params
        )
        try:
            result = xfoil.alpha(self.alpha)
            lift_drag_ratio = result["CL"] / result["CD"] # cd predition of inviscid xfoil is poor
            if np.isnan(lift_drag_ratio) or np.isinf(lift_drag_ratio):
                return self.NON_CONVERGED_REWARD
            # [0] is because xfoil returns an array
            return lift_drag_ratio[0]

        except Exception as e:
            pprint(0, "XFoil failed", e)
            return self.NON_CONVERGED_REWARD


class NeuralFoilSolver(BaseSolver):
    def __init__(self, alpha: float, Reynolds: float, model_size: str = "xxxlarge", device: str = "cpu"):
        super().__init__()
        self.Reynolds = Reynolds
        self.alpha = alpha
        self.model_size = model_size
        if not hasattr(nf, "get_aero_from_airfoil"):
            raiseWarning("NeuralFoil-torch is not installed, for larger model sizes it will run")
            self.device = None 
        else:
            self.device = device

    def __call__(self, airfoil):
        params = dict(
            airfoil=airfoil,
            alpha=self.alpha,
            Re=self.Reynolds,
            model_size=self.model_size,
        )
        if self.device is not None:
            params["device"] = self.device 

        result = nf.get_aero_from_airfoil(**params)
        confidence = result["analysis_confidence"][0]
        reward = (result["CL"][0] / result["CD"][0]) * confidence
        return reward
    


class DummySolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def __call__(self, shape):
        return self.NON_CONVERGED_REWARD    