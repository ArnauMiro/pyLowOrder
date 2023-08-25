#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# SPOD Module
#
# Last rev: 06/10/2022

__VERSION__ = '1.0.0'

from .wrapper import run
from .utils   import extract_modes, save, load
from .plots import plotMode, plotSpectra
