"""Waveform generation and ``hackrf_transfer`` control for TX-emitter.

The hardware exposes gain controls, not a calibrated output-power setting.  The
module therefore keeps digital amplitude, TX VGA gain, and the RF amplifier as
separate parameters and never labels any of them as dBm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from modem import Config, MODS, make_frame
from session import Source


HACKRF_MIN_FREQUENCY = 1_000_000
HACKRF_MAX_FREQUENCY = 6_000_000_000
HACKRF_SAMPLE_RATES = (2_000_000, 4_000_000, 8_000_000, 10_000_000,
                       12_000_000, 16_000_000, 20_000_000)
BASEBAND_FILTERS = (1_750_000, 2_500_000, 3_500_000, 5_000_000, 5_500_000,
                    6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
                    12_000_000, 14_000_000, 15_000_000, 20_000_000,
                    24_000_000, 28_000_000)
SIGNAL_MODES = ('packet', 'noise', 'file')
IQ_FORMATS = ('cs8', 'cf32')
TUNING_MODES = ('fixed', 'sweep')
OPERATION_MODES = ('continuous', 'pulse')
MAX_CYCLE_BYTES = 8 * 1024 * 1024
PACKET_RATE = 2_000_000


@dataclass(frozen=True)
class EmitterConfig:
    frequency_hz: int = 2_400_000_000
    sample_rate: int = 8_000_000
    bandwidth_hz: int = 1_000_000
    txvga_gain: int = 0
    rf_amp: bool = False
    amplitude: float = 0.35
    signal_mode: str = 'packet'
    modulation: str = 'BPSK'
    tuning_mode: str = 'fixed'
    operation_mode: str = 'continuous'
    sweep_period_ms: float = 100.0
    pulse_on_ms: float = 50.0
    pulse_off_ms: float = 50.0
    iq_path: str = ''
    iq_format: str = 'cs8'
    serial: str = ''
    executable: str = 'hackrf_transfer'

    def validate(self):
        if not HACKRF_MIN_FREQUENCY <= self.frequency_hz <= HACKRF_MAX_FREQUENCY:
            raise ValueError('Частота центра HackRF: 1…6000 МГц')
        if self.sample_rate not in HACKRF_SAMPLE_RATES:
            raise ValueError('Sample rate должен быть одним из 2, 4, 8, 10, 12, 16, 20 MS/s')
        if not 1_000 <= self.bandwidth_hz <= int(self.sample_rate * 0.70):
            raise ValueError('Полоса: 1 кГц…70% sample rate (оставлен защитный край)')
        half = self.bandwidth_hz / 2
        if (self.frequency_hz - half < HACKRF_MIN_FREQUENCY or
                self.frequency_hz + half > HACKRF_MAX_FREQUENCY):
            raise ValueError('Вся заданная полоса должна находиться внутри 1…6000 МГц')
        if not 0 <= self.txvga_gain <= 47 or int(self.txvga_gain) != self.txvga_gain:
            raise ValueError('TX VGA gain HackRF: целое число 0…47 dB')
        if not math.isfinite(self.amplitude) or not 0.01 <= self.amplitude <= 1.0:
            raise ValueError('Цифровая амплитуда: 1…100%')
        if self.signal_mode not in SIGNAL_MODES:
            raise ValueError('Неизвестный тип сигнала')
        if self.iq_format not in IQ_FORMATS:
            raise ValueError('Формат IQ-файла: CS8 или CF32')
        if self.signal_mode == 'file':
            iq_file_sample_count(self)
        if self.modulation not in MODS:
            raise ValueError('Неизвестная модуляция')
        if self.tuning_mode not in TUNING_MODES:
            raise ValueError('Неизвестный режим частоты')
        if self.operation_mode not in OPERATION_MODES:
            raise ValueError('Неизвестный режим работы')
        if not 5 <= self.sweep_period_ms <= 10_000:
            raise ValueError('Период сканирования: 5…10000 мс')
        if not 1 <= self.pulse_on_ms <= 10_000 or not 1 <= self.pulse_off_ms <= 10_000:
            raise ValueError('Длительность импульса и паузы: 1…10000 мс')
        if not self.executable.strip():
            raise ValueError('Укажите путь к hackrf_transfer')
        samples = cycle_sample_count(self)
        if samples * 2 > MAX_CYCLE_BYTES:
            raise ValueError('Цикл IQ превышает 8 MiB; уменьшите времена или sample rate')


def _active_sample_count(cfg: EmitterConfig) -> int:
    if (cfg.signal_mode == 'file' and cfg.operation_mode == 'continuous' and
            cfg.tuning_mode == 'fixed'):
        return iq_file_sample_count(cfg)
    if cfg.operation_mode == 'pulse':
        duration_ms = cfg.pulse_on_ms
    elif cfg.tuning_mode == 'sweep':
        duration_ms = cfg.sweep_period_ms
    else:
        duration_ms = 100.0
    return max(1, int(round(cfg.sample_rate * duration_ms / 1000)))


def cycle_sample_count(cfg: EmitterConfig) -> int:
    active = _active_sample_count(cfg)
    if cfg.operation_mode == 'pulse':
        active += max(1, int(round(cfg.sample_rate * cfg.pulse_off_ms / 1000)))
    return active


def iq_file_sample_count(cfg: EmitterConfig) -> int:
    path = Path(cfg.iq_path).expanduser()
    if not cfg.iq_path.strip() or not path.is_file():
        raise ValueError('Выберите существующий IQ-файл')
    bytes_per_sample = {'cs8': 2, 'cf32': 8}.get(cfg.iq_format)
    if bytes_per_sample is None:
        raise ValueError('Формат IQ-файла: CS8 или CF32')
    size = path.stat().st_size
    if not size or size % bytes_per_sample:
        unit = '2 байтам' if cfg.iq_format == 'cs8' else '8 байтам'
        raise ValueError(f'Размер IQ-файла должен быть ненулевым и кратным {unit}')
    return size // bytes_per_sample


def load_iq_file(path: str | Path, iq_format: str) -> np.ndarray:
    """Load signed interleaved CS8 or little-endian complex float32 samples."""
    path = Path(path).expanduser()
    raw = path.read_bytes()
    if iq_format == 'cs8':
        if not raw or len(raw) % 2:
            raise ValueError('CS8 должен содержать пары signed int8: I, Q')
        values = np.frombuffer(raw, np.int8).astype(np.float32)
        iq = (np.clip(values[0::2] / 127.0, -1, 1) +
              1j * np.clip(values[1::2] / 127.0, -1, 1))
    elif iq_format == 'cf32':
        if not raw or len(raw) % 8:
            raise ValueError('CF32 должен содержать little-endian complex64')
        iq = np.frombuffer(raw, dtype='<c8').astype(np.complex64, copy=True)
    else:
        raise ValueError('Формат IQ-файла: CS8 или CF32')
    if not np.all(np.isfinite(iq)):
        raise ValueError('IQ-файл содержит NaN или бесконечность')
    return np.asarray(iq, np.complex64)


def _packet_template(cfg: EmitterConfig) -> np.ndarray:
    """Return one real frame made by the existing project modem."""
    source = Source.text(('HackRF TX-emitter · ' + cfg.modulation + ' · ') * 8)
    modem_cfg = Config(cfg.modulation, True, False, PACKET_RATE)
    frame = make_frame(source.meta(0x4841434B52465458, 0), source.payload(0), modem_cfg)
    if cfg.sample_rate != PACKET_RATE:
        divisor = math.gcd(cfg.sample_rate, PACKET_RATE)
        frame = resample_poly(frame, cfg.sample_rate // divisor,
                              PACKET_RATE // divisor).astype(np.complex64)
    peak = float(np.max(np.abs(frame)))
    return (frame / max(peak, 1e-9)).astype(np.complex64)


def _band_limited_noise(sample_count: int, sample_rate: int,
                        width_hz: float) -> np.ndarray:
    # Complex white noise through a linear-phase LPF occupies approximately
    # -width/2…+width/2 around the configured RF center.
    taps = 257
    cutoff = float(np.clip(width_hz / sample_rate, 1e-5, 0.98))
    kernel = firwin(taps, cutoff)
    rng = np.random.default_rng(0x4841434B)
    raw = (rng.standard_normal(sample_count + taps) +
           1j * rng.standard_normal(sample_count + taps)).astype(np.complex64)
    return lfilter(kernel, [1.0], raw)[taps:].astype(np.complex64)


def frequency_track(cfg: EmitterConfig, sample_count: int) -> np.ndarray:
    """Instantaneous digital frequency offset for one active interval."""
    if cfg.tuning_mode == 'fixed':
        return np.zeros(sample_count, np.float64)
    if cfg.signal_mode == 'noise':
        # The instantaneous noise slice occupies 15% of the requested band;
        # its center scans through the rest, filling the band over time.
        span = cfg.bandwidth_hz * 0.85
    else:
        span = cfg.bandwidth_hz * 0.80
    # In pulse mode every RF burst makes one complete pass.  This keeps the
    # scan repeatable even when the requested ON/OFF timings are not a rational
    # multiple of the continuous-sweep period.
    period = (sample_count if cfg.operation_mode == 'pulse' else
              max(1.0, cfg.sample_rate * cfg.sweep_period_ms / 1000))
    position = (np.arange(sample_count, dtype=np.float64) / period) % 1.0
    triangle = 4.0 * np.abs(position - 0.5) - 1.0
    track = triangle * span / 2.0
    # A complete repeated period has zero accumulated phase at the boundary.
    if sample_count:
        track -= float(np.mean(track))
    return track


def build_cycle(cfg: EmitterConfig) -> np.ndarray:
    """Build one repeatable complex IQ cycle in the normalized [-1, 1] range."""
    cfg.validate()
    active_count = _active_sample_count(cfg)
    if cfg.signal_mode == 'packet':
        template = _packet_template(cfg)
        repeats = math.ceil(active_count / len(template))
        active = np.tile(template, repeats)[:active_count]
    elif cfg.signal_mode == 'noise':
        width = (cfg.bandwidth_hz * 0.15 if cfg.tuning_mode == 'sweep'
                 else cfg.bandwidth_hz)
        active = _band_limited_noise(active_count, cfg.sample_rate, width)
    else:
        template = load_iq_file(cfg.iq_path, cfg.iq_format)
        repeats = math.ceil(active_count / len(template))
        active = np.tile(template, repeats)[:active_count]

    if cfg.tuning_mode == 'sweep':
        track = frequency_track(cfg, active_count)
        phase = 2 * np.pi * np.cumsum(track, dtype=np.float64) / cfg.sample_rate
        active = active * np.exp(1j * phase)

    reference = float(np.quantile(np.abs(active), 0.999))
    active = active * (cfg.amplitude / max(reference, 1e-9))

    if cfg.operation_mode == 'pulse':
        # A short cosine ramp reduces switching splatter; the OFF part is exact zero.
        ramp_count = min(active_count // 10, max(1, int(cfg.sample_rate * 0.001)))
        ramp = np.sin(np.linspace(0, np.pi / 2, ramp_count, endpoint=True)) ** 2
        active[:ramp_count] *= ramp
        active[-ramp_count:] *= ramp[::-1]
        idle_count = max(1, int(round(cfg.sample_rate * cfg.pulse_off_ms / 1000)))
        active = np.r_[active, np.zeros(idle_count, np.complex64)]

    return np.asarray(active, np.complex64)


def complex_to_cs8(iq: np.ndarray) -> bytes:
    """Convert normalized complex samples to HackRF signed interleaved I/Q."""
    iq = np.asarray(iq)
    result = np.empty(iq.size * 2, np.int8)
    result[0::2] = np.rint(np.clip(iq.real, -1, 1) * 127).astype(np.int8)
    result[1::2] = np.rint(np.clip(iq.imag, -1, 1) * 127).astype(np.int8)
    return result.tobytes()


def select_filter_bandwidth(cfg: EmitterConfig) -> int:
    wanted = max(1_750_000, min(int(cfg.sample_rate * 0.75),
                               int(cfg.bandwidth_hz * 1.20)))
    return min(BASEBAND_FILTERS, key=lambda value: (value < wanted, abs(value - wanted)))


def build_command(cfg: EmitterConfig, iq_path: str | Path,
                  executable: str | None = None) -> list[str]:
    cfg.validate()
    command = [executable or cfg.executable]
    if cfg.serial.strip():
        command += ['-d', cfg.serial.strip()]
    command += ['-t', str(iq_path), '-f', str(cfg.frequency_hz),
                '-s', str(cfg.sample_rate), '-b', str(select_filter_bandwidth(cfg)),
                '-x', str(cfg.txvga_gain), '-a', '1' if cfg.rf_amp else '0', '-R']
    return command


def resolve_executable(value: str) -> str:
    value = value.strip().strip('"')
    path = Path(value)
    if path.is_absolute() or path.parent != Path('.'):
        if path.is_file():
            return str(path)
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise FileNotFoundError(
        'hackrf_transfer не найден. Установите HackRF Tools или укажите полный путь к exe.')


def run_transmitter(cfg: EmitterConfig, stop: threading.Event, emit):
    """Generate a cycle, start HackRF Tools, and remain active until Stop."""
    cfg.validate()
    executable = resolve_executable(cfg.executable)
    process = None
    with tempfile.TemporaryDirectory(prefix='tx_emitter_') as folder:
        iq_path = Path(folder) / 'waveform.cs8'
        emit('status', 'Формирование IQ…')
        iq = build_cycle(cfg)
        iq_path.write_bytes(complex_to_cs8(iq))
        duration_ms = len(iq) / cfg.sample_rate * 1000
        emit('log', f'IQ-цикл: {len(iq):,} отсчётов, {duration_ms:.1f} мс, '
                    f'{iq_path.stat().st_size / 1024 / 1024:.2f} MiB.')
        command = build_command(cfg, iq_path, executable)
        emit('log', 'Запуск: ' + subprocess.list2cmdline(command))
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, errors='replace', bufsize=1)
        if os.name == 'nt':
            kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        try:
            process = subprocess.Popen(command, **kwargs)

            def read_output():
                assert process is not None
                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            emit('log', line)

            reader = threading.Thread(target=read_output, name='hackrf_transfer output',
                                      daemon=True)
            reader.start()
            emit('status', 'TX активен')
            emit('log', 'Передача запущена. Stop завершит hackrf_transfer.')
            while process.poll() is None and not stop.wait(0.1):
                pass
            if stop.is_set() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            reader.join(timeout=1)
            code = process.returncode
            if not stop.is_set() and code:
                raise RuntimeError(f'hackrf_transfer завершился с кодом {code}')
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            emit('status', 'Остановлено')
