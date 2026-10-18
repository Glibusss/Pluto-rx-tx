"""Receiver-only quality estimates. See README_QUALITY_RU.md for assumptions.

Known-pilot complex LS residual estimator; one complex gain is fitted per block.
The residual variance and fitted signal power include finite-window corrections.
No transmitted user file or hard payload decisions enter the pilot estimator.
"""
import math
import numpy as np


def pilot_energies(observed, known):
    """Return debiased signal energy, disturbance energy, complex sample count.

    y = h*x + e, E|e|^2 = variance. For N complex samples and one fitted
    complex coefficient, variance = SSE/(N-1); fitted signal energy contains
    one variance of estimation noise, which is subtracted. Powers are unbiased
    under white independent errors; the ratio itself has finite-sample bias.
    For FSK the input contains both correlators, hence both error branches.
    """
    y = np.asarray(observed, complex).ravel()
    x = np.asarray(known, complex).ravel()
    if y.shape != x.shape:
        raise ValueError('Несовпадающие размеры пилотов')
    if len(y) < 3 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        return 0., 0., 0
    energy = float(np.vdot(x,x).real)
    if energy <= 1e-20:
        return 0., 0., 0
    gain = np.vdot(x,y)/energy
    residual = y-gain*x
    variance = float(np.vdot(residual,residual).real)/(len(y)-1)
    signal = max(0.,float(abs(gain)**2*energy)-variance)
    return signal, variance*len(y), len(y)


def vector_energies(observed, known):
    y, x = np.asarray(observed), np.asarray(known)
    if x.shape != y.shape:
        raise ValueError('Несовпадающие размеры векторов EVM')
    return float(np.sum(abs(x)**2)),float(np.sum(abs(y-x)**2))


def ratio_db(signal, disturbance, count):
    if not count or signal <= 0:
        return None
    return 10*math.log10(signal/max(disturbance,signal*1e-12))


def evm_percent(signal, disturbance, count):
    if not count or signal <= 0:
        return None
    return 100*math.sqrt(max(disturbance,0.)/signal)
