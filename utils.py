import torch
import numpy as np

def set_parameter_requires_grad(model, is_require):
    if is_require:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False