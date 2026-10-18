"""Spectrum calculation shared by the packet receiver and spectrum debugger."""
import numpy as np


def calculate_spectrum(iq, sample_rate, center_frequency, full_scale=2048.0):
    """Return absolute frequency bins and linear power relative to ADC full scale."""
    samples = np.asarray(iq, np.complex64).ravel()
    if len(samples) < 16:
        raise ValueError('Для FFT нужно не менее 16 комплексных отсчётов')
    if sample_rate <= 0 or full_scale <= 0:
        raise ValueError('Sample rate и full scale должны быть положительными')
    window = np.hanning(len(samples)).astype(np.float32)
    scale = full_scale*float(window.sum())
    transformed = np.fft.fftshift(np.fft.fft(samples*window))/scale
    power = np.maximum(abs(transformed)**2,1e-20)
    frequency = center_frequency+np.fft.fftshift(np.fft.fftfreq(len(samples),1/sample_rate))
    return frequency,power
