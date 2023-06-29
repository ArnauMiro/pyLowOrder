#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# POD Module
#
# Last rev: 27/06/2023

__VERSION__ = '2.0.1'

from .wrapper       import VariationalAutoencoder
from .architectures import EncoderNoPool, DecoderNoPool, EncoderMaxPool, DecoderMaxPool
from .callbacks     import EarlyStopper
from .utils         import Dataset, save

del wrapper