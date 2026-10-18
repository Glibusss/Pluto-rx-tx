"""Orthogonal binary FSK with configurable half-integer tone cycles per symbol."""
import numpy as np


def tone(sps, cycles=1):
    return np.exp(2j*np.pi*cycles*np.arange(sps)/sps)


def wave(symbols, sps, cycles=1):
    return np.exp(2j*np.pi*cycles*np.asarray(symbols).real[:, None]*np.arange(sps)/sps).ravel()


def observe(rows, cycles=1):
    reference = tone(rows.shape[1], cycles)
    return np.column_stack((rows@reference, rows@reference.conj())) / rows.shape[1]
