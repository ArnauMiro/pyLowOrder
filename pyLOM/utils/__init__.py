#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Utils Module
#
# Last rev: 19/07/2021

__VERSION__ = '1.0.0'

from .errors   import raiseError, raiseWarning
from .cr       import cr_start, cr_stop, cr_info
from .plotting import show_plots, close_plots, plotResidual, plotMode
from .matrix   import transpose, norm

del errors, cr, plotting