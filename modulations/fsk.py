"""Binary FSK tones and matched-filter observations for any positive deviation."""
import numpy as np


def tone(sps, cycles=1):
    return np.exp(2j*np.pi*cycles*np.arange(sps)/sps)


def wave(symbols, sps, cycles=1):
    return np.exp(2j*np.pi*cycles*np.asarray(symbols).real[:, None]*np.arange(sps)/sps).ravel()


def observe(rows, cycles=1):
    reference = tone(rows.shape[1], cycles)
    return np.column_stack((rows@reference, rows@reference.conj())) / rows.shape[1]


def reference(symbols, sps, cycles=1):
    """Ideal correlator outputs, including overlap of nonorthogonal tones."""
    return observe(wave(symbols, sps, cycles).reshape(-1, sps), cycles)
