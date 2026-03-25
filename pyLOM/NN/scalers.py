"""Backward-compatible scalers exports.

Canonical implementations live in ``pyLOM.NN.utils.scalers``.
This module intentionally re-exports the public scaler API for legacy imports.
"""

from .utils.scalers import (
    ScalerProtocol,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)

__all__ = [
    "ScalerProtocol",
    "MinMaxScaler",
    "StandardScaler",
    "RobustScaler",
]
