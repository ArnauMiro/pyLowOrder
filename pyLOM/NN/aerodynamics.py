#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Global aerodynamic calculation routines.
#
# Last rev: 20/05/2025

import torch

def lift_drag_coeff(
    CoefPressure: torch.Tensor = None,
    CoefSkinFriction: torch.Tensor = None,
    alpha: torch.Tensor = 0,
    normals: torch.Tensor = None,
    Sref: float = 80,
) -> tuple[float, float]:
    r"""
    Calculate lift and drag coefficients from pressure and skin friction coefficients.

    Args:
        CoefPressure (torch.Tensor, optional): Tensor of shape (n,) or (n,1) with the pressure coefficients of the surface (default: ``None``).
        CoefSkinFriction (torch.Tensor, optional): Tensor of shape (n,3) with the skin friction coefficients of the surface (default: ``None``).
        alpha (torch.Tensor): Angle of attack in deg.
        normals (torch.Tensor): Tensor of shape (n,3) with the normal vectors of the surface.
        Sref (float, optional): Reference area to calculate the lift and drag coefficients (default: ``80``).
        
    Returns:
        Tuple[float, float]: Lift and drag coefficients.
    """
    CoefLiftPressure, CoefDragPressure, CoefLiftFriction, CoefDragFriction = 0, 0, 0, 0
    alpha = alpha*torch.pi/180

    if normals.flatten().ndimension() == 1:
        normals = normals.view(-1, 3)
    elif normals.shape[1] != 3:
        raise ValueError("Normals must have shape (n,3).")
    
    # Handle CoefPressure
    if CoefPressure is not None:
        if CoefPressure.ndimension() == 1:
            ForcePressure = - torch.matmul(CoefPressure, normals).flatten()
            CoefLiftPressure = ForcePressure[2]*torch.cos(alpha) - ForcePressure[0]*torch.sin(alpha)
            CoefDragPressure = ForcePressure[2]*torch.sin(alpha) + ForcePressure[0]*torch.cos(alpha)
        else:
            raise ValueError("CoefPressure must have shape (n,) or (n,1).")
    
    # Handle CoefSkinFriction
    if CoefSkinFriction is not None:
        if CoefSkinFriction.ndimension() == 1:
            CoefSkinFriction = CoefSkinFriction.view(-1, 3)
        elif CoefSkinFriction.shape[1] != 3:
            raise ValueError("CoefSkinFriction must have shape (n,3).")
        
        Si = torch.sqrt(normals[:, 0]**2 + normals[:, 1]**2 + normals[:, 2]**2).view(-1, 1)
        ForceFriction = CoefSkinFriction * Si
        CoefLiftFriction = torch.sum(ForceFriction[:, 2] * torch.cos(alpha)) - torch.sum(ForceFriction[:, 0] * torch.sin(alpha))
        CoefDragFriction = torch.sum(ForceFriction[:, 2] * torch.sin(alpha)) + torch.sum(ForceFriction[:, 0] * torch.cos(alpha))
    
    if CoefPressure is None and CoefSkinFriction is None:
        raise ValueError("At least one of CoefPressure or CoefSkinFriction must be provided.")
    
    # Compute final lift and drag coefficients
    CL = (CoefLiftPressure + CoefLiftFriction) / Sref 
    CD = (CoefDragPressure + CoefDragFriction) / Sref 
    return CL, CD