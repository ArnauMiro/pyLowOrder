
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
    """
    Base class for all parameterizers. It defines the interface for the parameterizers.
    """
    @abstractmethod
    def get_shape_from_params(self, params: np.ndarray) -> asb.Airfoil:
        """
        Create a shape from the given parameters.

        Args:
            params (np.ndarray): Parameters to create the shape.

        Returns:
            asb.Airfoil: Created shape.
        """
        pass
    
    @abstractmethod
    def get_optimizable_bounds(self) -> Tuple[List, List]:
        """
        Get the bounds of the optimizable parameters.

        Returns:
            Tuple[List, List]: Lower and upper bounds of the parameters.
        """
        pass
    
    @abstractmethod
    def get_params_from_shape(self, shapea: asb.Airfoil) -> np.ndarray:
        """
        Get the parameters from the given shape.
        Args:
            shape (asb.Airfoil): Shape to get the parameters from.

        Returns:
            np.ndarray: Parameters of the shape.
        """
        pass
    
    @abstractmethod
    def generate_random_params(self, seed=None) -> np.ndarray:
        """
        Generate random parameters within the bounds.

        Args:
            seed (int, optional): Seed for the random number generator. Default is ``None``.

        Returns:
            np.ndarray: Random parameters within the bounds.
        """
        pass


class AirfoilCSTParametrizer(BaseParameterizer):
    """
    CST parameterization of airfoils. The CST method is a way to parameterize airfoils using a set of control points.
    Depending on the number of points in upper and lower surfaces, the number of parameters will be different.

    Args:
        upper_surface_bounds (Tuple[List, List]): Bounds for the upper surface parameters. The first list has the lower bounds and the second list has the upper bounds.The length of the lists should be equal to the number of parameters for the upper surface.
        lower_surface_bounds (Tuple[List, List]): Bounds for the lower surface parameters. The first list has the lower bounds and the second list has the upper bounds.The length of the lists should be equal to the number of parameters for the lower surface.
        TE_thickness_bounds (Tuple[float, float]): Bounds for the TE thickness. The first value is the lower bound and the second value is the upper bound.
        leading_edge_weight (Tuple[float, float]): Bounds for the leading edge weight. The first value is the lower bound and the second value is the upper bound.
    """

    naca_4digit_airfoils = [
        "naca0006", "naca0009", "naca0012", "naca0015",
        "naca0018", "naca1408", "naca1410", "naca1412",
        "naca2412", "naca2415", "naca4412", "naca4415",
        "naca4420", "naca6412", "naca6415", "naca7421",
        "naca8409", "naca8412", "naca8415", "naca9421",
    ]

    def __init__(
        self,
        upper_surface_bounds: Tuple[List, List],
        lower_surface_bounds: Tuple[List, List],
        TE_thickness_bounds: Tuple[List, List],
        leading_edge_weight: Tuple[List, List],
    ):
        self.upper_surface_bounds = upper_surface_bounds
        self.lower_surface_bounds = lower_surface_bounds
        self.leading_edge_weight_bounds = leading_edge_weight
        self.TE_thickness_bounds = TE_thickness_bounds
        self.num_upper_surface_params = len(upper_surface_bounds[0])
        self.num_lower_surface_params = len(lower_surface_bounds[0])


    def get_shape_from_params(self, params):
        # Check if the params are within the bounds
        upper_surface_params, lower_surface_params, leading_edge_weight, TE_thickness = self._deserialize_params(params)
        airfoil = asb.KulfanAirfoil(
            upper_weights=upper_surface_params,
            lower_weights=lower_surface_params,
            leading_edge_weight=leading_edge_weight,
            TE_thickness=TE_thickness,
        )
        
        return airfoil
    
    def get_params_from_shape(self, airfoil: asb.Airfoil) -> np.ndarray:
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

    def generate_random_params(self, source="naca", seed=None) -> np.ndarray:
        """
        Generate random parameters within the bounds.

        If source is "naca", generate a random NACA 4-digit airfoil.
        If source is "uiuc", generate random airfoil from the UIUC dataset.

        Args:
            source (str): Source of the airfoil. Can be "naca" or "uiuc". Default is "naca".
            seed (int, optional): Seed for the random number generator. Default is ``None``.
        Returns:
            np.ndarray: Random parameters for an airfoil.
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
    
    def get_optimizable_bounds(self) -> Tuple[List, List]:
        """
        Get the bounds of the parameters.

        Returns:
            Tuple[List, List]: Lower and upper bounds of the parameters. The first list has the lower bounds and the second list has the upper bounds.
        
        """
        return (
            self.upper_surface_bounds[0] + self.lower_surface_bounds[0] + [self.leading_edge_weight_bounds[0], self.TE_thickness_bounds[0]],
            self.upper_surface_bounds[1] + self.lower_surface_bounds[1] + [self.leading_edge_weight_bounds[1], self.TE_thickness_bounds[1]],
        )

class WingParameterizer(BaseParameterizer):
    def __init__(
        self,
        airfoil: asb.Airfoil,
        chord_bounds: Tuple[List],
        twist_bounds: Tuple[List],
        span_bounds: Tuple[List],
        sweep_bounds: Tuple[List],
        dihedral_bounds: Tuple[List],
    ):
        """
        Initialize the WingParametrizerDust class.
        
        Args:
            airfoil (asb.Airfoil): The airfoil to be used for the wing sections.
            The bounds for the chord, twist, span, sweep, and dihedral parameters are tuples with two lists, the first list is the lower bound and the second list is the upper bound. If the upper and lower bounds are the same, the parameter will not be considered for optimization. A section of the wing is created for each chord and twist pair, and a region of the wing is created for each span, sweep, and dihedral triplet, the number of sections should be one more than the number of regions. The angles need to be in degrees.
        """
        self.lower_bounds = np.array(
            chord_bounds[0] + twist_bounds[0] + span_bounds[0] + sweep_bounds[0] + dihedral_bounds[0]
        )
        self.upper_bounds = np.array(
            chord_bounds[1] + twist_bounds[1] + span_bounds[1] + sweep_bounds[1] + dihedral_bounds[1]
        )
        self.non_optimizable_params_mask = (self.lower_bounds == self.upper_bounds)
        self.airfoil = airfoil
        assert len(chord_bounds[0]) == len(twist_bounds[0]), "Chord and twist bounds must have the same length"
        assert len(span_bounds[0]) == len(sweep_bounds[0]) and len(span_bounds[0]) == len(dihedral_bounds[0]), "Span, sweep and dihedral bounds must have the same length"
        self.num_sections = len(chord_bounds[0])
        self.num_regions = len(span_bounds[0])
        assert self.num_sections == self.num_regions + 1, "There should be one more chord section than span section"


    def get_shape_from_params(self, params):
        if len(params) != len(self.lower_bounds):   
            competed_params = self._complete_params(params)
        else:
            competed_params = params
        chords = competed_params[:self.num_sections]
        twists = competed_params[self.num_sections:2*self.num_sections]
        spans = competed_params[2*self.num_sections:2*self.num_sections + self.num_regions]
        sweeps = competed_params[2*self.num_sections + self.num_regions:2*self.num_sections + 2*self.num_regions]
        diheds = competed_params[2*self.num_sections + 2*self.num_regions:]
        coordinates = [np.array([0, 0, 0])]
        for i in range(self.num_regions):
            new_coordinates = np.array([0, 0, 0], dtype=float)
            new_coordinates[0] = coordinates[-1][0] + np.tan(np.radians(sweeps[i])) * spans[i]
            new_coordinates[1] = coordinates[-1][1] + spans[i]
            new_coordinates[2] = coordinates[-1][2] + np.tan(np.radians(diheds[i])) * spans[i]
            coordinates.append(new_coordinates)

        wing = asb.Wing(
            name="Main Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections
                asb.WingXSec(  # Root
                    xyz_le=coords,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=chord,
                    twist=twist,  # degrees
                    airfoil=self.airfoil,  # Airfoils are blended between a given XSec and the next one.
                ) for chord, twist, coords in zip(chords, twists, coordinates) 
            ])
        return wing


    def _complete_params(self, params):
        competed_params = np.zeros_like(self.lower_bounds)
        competed_params[~self.non_optimizable_params_mask] = params
        competed_params[self.non_optimizable_params_mask] = self.lower_bounds[self.non_optimizable_params_mask]
        return competed_params

    def get_params_from_shape(self, wing):
        chords = [section.chord for section in wing.xsecs]
        twists = [section.twist for section in wing.xsecs]
        coordinates = [wing._compute_xyz_le_of_WingXSec(i).tolist() for i in range(len(wing.xsecs))]
        spans, sweeps, diheds = [], [], []

        for i in range(1, len(coordinates)):
            first_coord = coordinates[i-1]
            second_coord = coordinates[i]
            span = second_coord[1] - first_coord[1]
            sweep = np.degrees(np.arctan((second_coord[0] - first_coord[0]) / span))
            dihed = np.degrees(np.arctan((second_coord[2] - first_coord[2]) / span))
            spans.append(span)
            sweeps.append(sweep)
            diheds.append(dihed)

        params = np.array(chords + twists + spans + sweeps + diheds, dtype=np.float32)
        return params[~self.non_optimizable_params_mask]   

    def generate_random_params(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        new_params = np.random.uniform(self.lower_bounds, self.upper_bounds)
        # keep only the optimizable parameters
        new_params = new_params[~self.non_optimizable_params_mask]
        return new_params
    
    def get_optimizable_bounds(self):
        return self.lower_bounds[~self.non_optimizable_params_mask], self.upper_bounds[~self.non_optimizable_params_mask]