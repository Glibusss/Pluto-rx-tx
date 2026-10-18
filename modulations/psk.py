"""BPSK and Gray-labelled PSK constellations."""
import numpy as np


def points(order):
    if order == 2:
        return np.array([-1., 1.], complex)
    result = np.empty(order, complex)
    for index in range(order):
        result[index ^ (index >> 1)] = np.exp(1j*(2*np.pi*index/order+np.pi/order))
    return result
