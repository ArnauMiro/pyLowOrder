#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD Module
#
# Last rev: 30/09/2021

__VERSION__ = '1.0.0'


# Functions coming from DMD
from .wrapper import run, frequency_damping, reconstruction_jovanovic
from .plots   import ritzSpectrum, amplitudeFrequency, dampingFrequency, plotMode, plotResidual, animateFlow

del wrapper
