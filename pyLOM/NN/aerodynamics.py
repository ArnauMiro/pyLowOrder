#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Global aerodynamic calculation routines.
#
# Last rev: 22/05/2025

import torch

from ..utils.errors     import raiseError

def global_coeff(
    CoefPressure: torch.Tensor,
    SRef: float,
    MomentCenter: torch.Tensor,
    cRef: float,
    CoefSkinFriction: torch.Tensor = None,
    alpha: torch.Tensor = 0,
    coordinates: torch.Tensor = None,
    normals: torch.Tensor = None,
    components: tuple[str, ...] = ("CL", "CD", "CMx", "CMy", "CMz") ,
) -> dict[str, torch.Tensor]:
    r"""
    Calculate selected aerodynamic coefficients (CL, CD, CMx, CMy, CMz) from pressure and skin friction coefficients.

    Args:
        CoefPressure (torch.Tensor): Tensor of shape (n,) or (n,1) with the pressure coefficients of the surface.
        SRef (float): Reference area.
        MomentCenter (torch.Tensor): Tensor of shape (3,) with the moment center coordinates.
        cRef (float): Reference chord length.
        CoefSkinFriction (torch.Tensor, optional): Tensor of shape (n,3) with the skin friction coefficients of the surface (default: `None).
        alpha (torch.Tensor): Angle of attack in deg.
        coordinates (torch.Tensor): Tensor of shape (n,3) with the coordinates of the surface points.
        normals (torch.Tensor): Tensor of shape (n,3) with the normal vectors of the surface.
        components (set[str]): Set containing any combination of "CL", "CD", "CMx", "CMy", "CMz" indicating what to compute.

    Returns:
        set[torch.Tensor]: A set of tensors containing the requested aerodynamic coefficients.
    """
    if normals.ndimension() == 1 or normals.shape[1] == 1:
        normals = normals.view(-1, 3)
    if normals.shape[1] != 3:
        raiseError(f"Normals must have shape (n,3).")

    if coordinates.ndimension() == 1 or coordinates.shape[1] == 1:
        coordinates = coordinates.view(-1, 3)
    if coordinates.shape[1] != 3:
        raiseError(f"Coordinates must have shape (n,3).")

    alpha = alpha*torch.pi/180
    r = coordinates - MomentCenter.view(1, 3)

    results = {}
    
    # Pressure contribution
    ForcePressure = torch.zeros_like(coordinates)
    if CoefPressure is not None:
        if CoefPressure.ndimension() == 1 or CoefPressure.shape[1] == 1:
            ForcePressure = -CoefPressure.view(-1, 1) * normals
        else:
            raiseError(f"CoefPressure must have shape (n,) or (n,1).")

    # Friction contribution
    ForceFriction = torch.zeros_like(coordinates)
    if CoefSkinFriction is not None:
        if CoefSkinFriction.ndimension() == 1:
            CoefSkinFriction = CoefSkinFriction.view(-1, 3)
        elif CoefSkinFriction.shape[1] != 3:
            raiseError(f"CoefSkinFriction must have shape (n,3).")
        Si = torch.norm(normals, dim=1).view(-1, 1)
        ForceFriction = CoefSkinFriction * Si

    if CoefPressure is None and CoefSkinFriction is None:
        raiseError("At least one of CoefPressure or CoefSkinFriction must be provided.")

    # Compute only requested components
    if "CL" in components:
        CLp = torch.sum(ForcePressure[:, 2] * torch.cos(alpha) - ForcePressure[:, 0] * torch.sin(alpha)) if CoefPressure is not None else 0
        CLf = torch.sum(ForceFriction[:, 2] * torch.cos(alpha) - ForceFriction[:, 0] * torch.sin(alpha)) if CoefSkinFriction is not None else 0
        results["CL"] = (CLp + CLf) / SRef

    if "CD" in components:
        CDp = torch.sum(ForcePressure[:, 2] * torch.sin(alpha) + ForcePressure[:, 0] * torch.cos(alpha)) if CoefPressure is not None else 0
        CDf = torch.sum(ForceFriction[:, 2] * torch.sin(alpha) + ForceFriction[:, 0] * torch.cos(alpha)) if CoefSkinFriction is not None else 0
        results["CD"] = (CDp + CDf) / SRef

    if "CMx" in components:
        Mxp = torch.sum(torch.linalg.cross(r, ForcePressure)[:, 0]) if CoefPressure is not None else 0
        Mxf = torch.sum(torch.linalg.cross(r, ForceFriction)[:, 0]) if CoefSkinFriction is not None else 0
        results["CMx"] = (Mxp + Mxf) / (SRef * cRef)

    if "CMy" in components:
        Mp = torch.sum(torch.linalg.cross(r, ForcePressure)[:, 1]) if CoefPressure is not None else 0
        Mf = torch.sum(torch.linalg.cross(r, ForceFriction)[:, 1]) if CoefSkinFriction is not None else 0
        results["CMy"] = (Mp + Mf) / (SRef * cRef)

    if "CMz" in components:
        Mzp = torch.sum(torch.linalg.cross(r, ForcePressure)[:, 2]) if CoefPressure is not None else 0
        Mzf = torch.sum(torch.linalg.cross(r, ForceFriction)[:, 2]) if CoefSkinFriction is not None else 0
        results["CMz"] = (Mzp + Mzf) / (SRef * cRef)

    return tuple(results[key] for key in components)

def jacobians_pressure(
    SRef: float,
    MomentCenter: torch.Tensor,
    cRef: float,
    alpha: torch.Tensor = 0,
    coordinates: torch.Tensor = None,
    normals: torch.Tensor = None,
    components: tuple[str, ...] = ("CL", "CD", "CMx", "CMy", "CMz"),
) -> dict[str, torch.Tensor]:
    r"""
    Calculate the Jacobians of selected aerodynamic coefficients (CL, CD, CMx, CMy, CMz)

    Args:
        SRef (float): Reference area.
        MomentCenter (torch.Tensor): Tensor of shape (3,) with the moment center coordinates.
        cRef (float): Reference chord length.
        alpha (torch.Tensor): Angle of attack in deg.
        coordinates (torch.Tensor): Tensor of shape (n,3) with the coordinates of the surface points.
        normals (torch.Tensor): Tensor of shape (n,3) with the normal vectors of the surface.
        components (set[str]): Set containing any combination of "CL", "CD", "CMx", "CMy", "CMz" indicating what to compute.
    
    Returns:
        set[torch.Tensor]: A set of tensors containing the requested aerodynamic coefficient Jacobians.
    """

    if normals.ndimension() == 1 or normals.shape[1] == 1:
        normals = normals.view(-1, 3)
    if normals.shape[1] != 3:
        raiseError(f"Normals must have shape (n,3).")

    if coordinates.ndimension() == 1 or coordinates.shape[1] == 1:
        coordinates = coordinates.view(-1, 3)
    if coordinates.shape[1] != 3:
        raiseError(f"Coordinates must have shape (n,3).")

    alpha = alpha*torch.pi/180
    r = coordinates - MomentCenter.view(1, 3)

    results = {}

    # Compute only requested components
    if "CL" in components:
        results["CL"] = (1.0 / SRef) * (normals[:, 0] * torch.sin(alpha) - normals[:, 2] * torch.cos(alpha))

    if "CD" in components:
        results["CD"] = (1.0 / SRef) * (-normals[:, 0] * torch.cos(alpha) - normals[:, 2] * torch.sin(alpha))

    if "CMx" in components:
        results["CMx"] = (1.0 / (SRef * cRef)) * (r[:, 2] * normals[:, 1] - r[:, 1] * normals[:, 2])
    
    if "CMy" in components:
        results["CMy"] = (1.0 / (SRef * cRef)) * (r[:, 0] * normals[:, 2] - r[:, 2] * normals[:, 0])
    
    if "CMz" in components:
        results["CMz"] = (1.0 / (SRef * cRef)) * (r[:, 1] * normals[:, 0] - r[:, 0] * normals[:, 1])

    return tuple(results[key] for key in components)