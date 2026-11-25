#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN activation constructors.
#
# Last rev: 23/11/2026

import torch


def tanh() -> torch.nn.Module:
    return torch.nn.Tanh()

def relu() -> torch.nn.Module:
    return torch.nn.ReLU()

def elu() -> torch.nn.Module:
    return torch.nn.ELU()

def sigmoid() -> torch.nn.Module:
    return torch.nn.Sigmoid()

def leakyRelu() -> torch.nn.Module:
    return torch.nn.LeakyReLU()

def silu() -> torch.nn.Module:
    return torch.nn.SiLU()