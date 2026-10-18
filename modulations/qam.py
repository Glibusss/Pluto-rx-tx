"""Square QAM, Gray-labelled independently on I and Q."""
import numpy as np


def points(order):
    side = int(np.sqrt(order))
    levels = np.empty(side)
    for index in range(side):
        levels[index ^ (index >> 1)] = 2*index-(side-1)
    result = np.array([i+1j*q for i in levels for q in levels])
    return result / np.sqrt(np.mean(abs(result)**2))
