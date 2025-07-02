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
) -> tuple[float, float, float]:
    r"""
    Calculate lift, drag and moment coefficients from pressure coefficient and skin friction coefficient.

    Args:
        CoefPressure (torch.Tensor): Tensor of shape (n,) or (n,1) with the pressure coefficients of the surface.
        SRef (float): Reference area.
        MomentCenter (torch.Tensor): Tensor of shape (3,) with the moment center coordinates.
        cRef (float): Reference chord length.
        CoefSkinFriction (torch.Tensor, optional): Tensor of shape (n,3) with the skin friction coefficients of the surface (default: `None).
        alpha (torch.Tensor): Angle of attack in deg.
        coordinates (torch.Tensor): Tensor of shape (n,3) with the coordinates of the surface points.
        normals (torch.Tensor): Tensor of shape (n,3) with the normal vectors of the surface.
        
    Returns:
        Tuple[float, float]: Lift and drag coefficients.
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

    # Pressure forces
    ForcePressure = torch.zeros_like(coordinates)
    CoefLiftPressure, CoefDragPressure, CoefMomentPressure = 0, 0, 0
    if CoefPressure is not None:
        if CoefPressure.ndimension() == 1 or CoefPressure.shape[1] == 1:
            ForcePressure = -CoefPressure.view(-1, 1) * normals
            CoefLiftPressure = torch.sum(ForcePressure[:, 2]*torch.cos(alpha) - ForcePressure[:, 0]*torch.sin(alpha))
            CoefDragPressure = torch.sum(ForcePressure[:, 2]*torch.sin(alpha) + ForcePressure[:, 0]*torch.cos(alpha))
            CoefMomentPressure = torch.sum(torch.linalg.cross(r, ForcePressure)[:,1])
        else:
            raiseError(f"CoefPressure must have shape (n,) or (n,1).")

    # Skin friction forces
    ForceFriction = torch.zeros_like(coordinates)
    CoefLiftFriction, CoefDragFriction, CoefMomentFriction = 0, 0, 0
    if CoefSkinFriction is not None:
        if CoefSkinFriction.ndimension() == 1:
            CoefSkinFriction = CoefSkinFriction.view(-1, 3)
        elif CoefSkinFriction.shape[1] != 3:
            raiseError(f"CoefSkinFriction must have shape (n,3).")

        Si = torch.norm(normals, dim=1).view(-1, 1)
        ForceFriction = CoefSkinFriction * Si
        CoefLiftFriction = torch.sum(ForceFriction[:, 2]*torch.cos(alpha)) - torch.sum(ForceFriction[:, 0]*torch.sin(alpha))
        CoefDragFriction = torch.sum(ForceFriction[:, 2]*torch.sin(alpha)) + torch.sum(ForceFriction[:, 0]*torch.cos(alpha))
        CoefMomentFriction = torch.sum(torch.linalg.cross(r, ForceFriction)[:,1])

    if CoefPressure is None and CoefSkinFriction is None:
        raiseError(f"At least one of CoefPressure or CoefSkinFriction must be provided.")

    # Coefficients
    CM = (CoefMomentPressure + CoefMomentFriction) / (SRef * cRef)
    CL = (CoefLiftPressure + CoefLiftFriction) / SRef
    CD = (CoefDragPressure + CoefDragFriction) / SRef

    return CL, CD, CM