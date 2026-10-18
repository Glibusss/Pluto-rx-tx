"""Shared modulation API used by the modem, receiver and HackRF emitter."""
from functools import lru_cache
import numpy as np
from . import ask, fsk, psk, qam
from . import fsk as _fsk

MODS = ('ASK / OOK', '2-FSK', 'BPSK', 'QPSK', '8-PSK', 'QAM-4', 'QAM-16', 'QAM-64')
SPS = 8


def bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def pack(values):
    return np.packbits(np.asarray(values, np.uint8)).tobytes()


@lru_cache(maxsize=128)
def _constellation(mod, phase, low):
    if mod == 'ASK / OOK':
        pts, k = ask.points(low), 1
    elif mod == '2-FSK':
        pts, k = np.array([-1., 1.]), 1
    elif mod in ('BPSK', 'QPSK', '8-PSK'):
        k = {'BPSK': 1, 'QPSK': 2, '8-PSK': 3}[mod]
        pts = psk.points(2**k) * np.exp(1j*np.deg2rad(phase))
    elif mod in ('QAM-4', 'QAM-16', 'QAM-64'):
        k = {'QAM-4': 2, 'QAM-16': 4, 'QAM-64': 6}[mod]
        pts = qam.points(2**k) * np.exp(1j*np.deg2rad(phase))
    else:
        raise ValueError(mod)
    pts.setflags(write=False)
    return pts, k


def constellation(mod, cfg=None):
    return _constellation(mod, getattr(cfg, 'phase_deg', 0.), getattr(cfg, 'ook_low', 0.))


def bit_symbols(values, mod, cfg=None):
    pts, k = constellation(mod, cfg)
    values = np.asarray(values, np.uint8)
    values = np.pad(values, (0, (-len(values)) % k)).reshape(-1, k)
    return pts[values @ (1 << np.arange(k-1, -1, -1))]


def symbols(data, mod, cfg=None):
    return bit_symbols(bits(data), mod, cfg)


def decisions(observations, mod, cfg=None):
    if mod == '2-FSK':
        return (abs(observations[:, 1]) > abs(observations[:, 0])).astype(np.uint8)
    pts, k = constellation(mod, cfg)
    labels = np.argmin(abs(observations[:, None]-pts[None, :])**2, axis=1)
    return ((labels[:, None] >> np.arange(k-1, -1, -1)) & 1).astype(np.uint8).ravel()


def to_wave(values, fsk=False, cycles=1):
    result = _fsk.wave(values, SPS, cycles) if fsk else np.repeat(values, SPS)
    return result.astype(np.complex64)


def decision_error_energies(observations, mod, cfg=None):
    z = np.asarray(observations)
    if not len(z):
        return 0., 0.
    if mod == '2-FSK':
        choose_one = abs(z[:, 1]) > abs(z[:, 0])
        decided = np.column_stack((~choose_one, choose_one)).astype(complex)
    else:
        pts, _ = constellation(mod, cfg)
        decided = pts[np.argmin(abs(z[:, None]-pts[None, :])**2, axis=1)]
    return float(np.sum(abs(decided)**2)), float(np.sum(abs(z-decided)**2))
