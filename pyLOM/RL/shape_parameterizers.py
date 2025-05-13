
import os
import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import aerosandbox as asb
import numpy as np
from aerosandbox import _asb_root

from pyLOM.utils import raiseError

airfoil_database_root = _asb_root / "geometry" / "airfoil" / "airfoil_database"


class BaseParameterizer(ABC):
    @abstractmethod
    def get_shape_from_params(self, params):
        """
        Create a shape from the given parameters.
        """
        pass
    
    @abstractmethod
    def get_optimizable_bounds(self):
        """
        Get the bounds of the optimizable parameters.
        """
        pass
    
    @abstractmethod
    def get_params_from_shape(self, shape):
        """
        Get the parameters from the given shape.
        """
        pass
    
    @abstractmethod
    def generate_random_params(self, seed=None):
        """
        Generate random parameters within the bounds.
        """
        pass


class AirfoilCSTParametrizer(BaseParameterizer):
    naca_4digit_airfoils = [
        "naca0006", "naca0009", "naca0012", "naca0015",
        "naca0018", "naca1408", "naca1410", "naca1412",
        "naca2412", "naca2415", "naca4412", "naca4415",
        "naca4420", "naca6412", "naca6415", "naca7421",
        "naca8409", "naca8412", "naca8415", "naca9421",
    ]

    def __init__(
        self,
        upper_surface_bounds: Tuple[List],
        lower_surface_bounds: Tuple[List],
        TE_thickness_bounds: Tuple[List],
        leading_edge_weight: Tuple[List],
    ):
        self.upper_surface_bounds = upper_surface_bounds
        self.lower_surface_bounds = lower_surface_bounds
        self.leading_edge_weight_bounds = leading_edge_weight
        self.TE_thickness_bounds = TE_thickness_bounds
        self.num_upper_surface_params = len(upper_surface_bounds[0])
        self.num_lower_surface_params = len(lower_surface_bounds[0])


    def get_shape_from_params(self, params):
        """
        Create an airfoil using the CST method.
        The first 16 parameters are for the upper surface, the next 16 are for the lower surface,
        the next 2 are for the TE thickness and leading edge weight.
        """
        # Check if the params are within the bounds
        upper_surface_params, lower_surface_params, leading_edge_weight, TE_thickness = self._deserialize_params(params)
        airfoil = asb.KulfanAirfoil(
            upper_weights=upper_surface_params,
            lower_weights=lower_surface_params,
            leading_edge_weight=leading_edge_weight,
            TE_thickness=TE_thickness,
        )
        
        return airfoil
    
    def get_params_from_shape(self, airfoil):
        """
        Get the parameters from the airfoil.
        The first 16 parameters are for the upper surface, the next 16 are for the lower surface,
        the next 2 are for the TE thickness and leading edge weight.
        """
        if not isinstance(airfoil, asb.KulfanAirfoil):
            airfoil = airfoil.to_kulfan_airfoil(n_weights_per_side=self.num_upper_surface_params)
        return np.hstack(
            (
                airfoil.upper_weights,
                airfoil.lower_weights,
                np.array((airfoil.leading_edge_weight, airfoil.TE_thickness)),
            ),
            dtype=np.float32,
        )

    def generate_random_params(self, source="naca", seed=None):
        """
        Generate random parameters within the bounds.

        If source is "naca", generate a random NACA 4-digit airfoil.
        If source is "uiuc", generate random airfoil from the UIUC dataset.
        """
        if seed is not None:
            random.seed(seed)

        source = source.lower()
        if source == "naca":
            random_idx = random.randint(0, len(self.naca_4digit_airfoils) - 1)
            random_airfoil = self.naca_4digit_airfoils[random_idx]
            airfoil = asb.Airfoil(random_airfoil).to_kulfan_airfoil(n_weights_per_side=self.num_upper_surface_params)
        elif source == "uiuc":
            if not hasattr(self, "uiuc_airfoil_names"):
                # Load the UIUC airfoil dataset
                uiuc_airfoil_names = os.listdir(airfoil_database_root)
                uiuc_airfoil_names.remove("utils")
                self.uiuc_airfoil_names = uiuc_airfoil_names
            random_idx = random.randint(0, len(self.uiuc_airfoil_names) - 1)
            random_airfoil = self.uiuc_airfoil_names[random_idx]
            airfoil = asb.Airfoil(random_airfoil).to_kulfan_airfoil(n_weights_per_side=self.num_upper_surface_params)
        else:  
            raise raiseError("Invalid source. Choose 'naca' or 'uiuc'.")
        
        return self.get_params_from_shape(airfoil)
    
    def _deserialize_params(self, params):
        upper_surface_params = params[:self.num_upper_surface_params]
        lower_surface_params = params[self.num_upper_surface_params:self.num_upper_surface_params + self.num_lower_surface_params]
        leading_edge_weight = params[self.num_upper_surface_params + self.num_lower_surface_params] 
        TE_thickness = params[self.num_upper_surface_params + self.num_lower_surface_params + 1]
    
        return upper_surface_params, lower_surface_params, leading_edge_weight, TE_thickness
    
    def get_optimizable_bounds(self):
        """
        Get the bounds of the parameters.
        """
        return (
            self.upper_surface_bounds[0] + self.lower_surface_bounds[0] + [self.leading_edge_weight_bounds[0], self.TE_thickness_bounds[0]],
            self.upper_surface_bounds[1] + self.lower_surface_bounds[1] + [self.leading_edge_weight_bounds[1], self.TE_thickness_bounds[1]],
        )