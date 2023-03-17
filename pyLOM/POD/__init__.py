#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# POD Module
#
# Last rev: 09/07/2021

__VERSION__ = '1.0.0'

from .wrapper import run, truncate, reconstruct
from .utils   import extract_modes, save, load
from .plots   import plotResidual, plotMode, plotSnapshot


del wrapper, plots