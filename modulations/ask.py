"""Normalized two-level amplitude modulation (OOK at low=0)."""
import numpy as np


def points(low=0.):
    return np.array([low, 1.], complex) / np.sqrt((1+low**2)/2)
