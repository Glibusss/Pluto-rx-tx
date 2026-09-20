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
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

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


def _tool_filename(name: str) -> str:
    return name + '.exe' if os.name == 'nt' and not name.lower().endswith('.exe') else name


def _common_tool_directories(preferred: str = '') -> list[Path]:
    """Return bounded, conventional HackRF Tools locations without scanning a drive."""
    directories: list[Path] = []
    if preferred:
        preferred_path = Path(preferred.strip().strip('"')).expanduser()
        if preferred_path.parent != Path('.'):
            directories.append(preferred_path.parent)
    directories += [Path(__file__).resolve().parent, Path(sys.executable).resolve().parent]

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        directories += [Path(conda_prefix) / 'Library' / 'bin', Path(conda_prefix) / 'bin']
    mamba_root = os.environ.get('MAMBA_ROOT_PREFIX')
    if mamba_root:
        directories += [Path(mamba_root) / 'Library' / 'bin', Path(mamba_root) / 'bin']
    user_profile = os.environ.get('USERPROFILE')
    if user_profile:
        directories.append(Path(user_profile) / 'scoop' / 'apps' / 'hackrf' / 'current' / 'bin')

    roots = [os.environ.get('ProgramFiles'), os.environ.get('ProgramFiles(x86)'),
             os.environ.get('LOCALAPPDATA')]
    for value in filter(None, roots):
        root = Path(value)
        directories += [root / 'HackRF' / 'bin', root / 'HackRF Tools' / 'bin',
                        root / 'Great Scott Gadgets' / 'HackRF' / 'bin',
                        root / 'PothosSDR' / 'bin', root / 'Programs' / 'HackRF' / 'bin']
        try:
            directories.extend(path / 'bin' for path in root.glob('PothosSDR*'))
        except OSError:
            pass

    unique: list[Path] = []
    seen = set()
    for directory in directories:
        key = os.path.normcase(os.path.abspath(str(directory)))
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def find_hackrf_tool(name: str, preferred: str = '') -> str | None:
    """Locate one HackRF command in an explicit path, PATH, or common installs."""
    filename = _tool_filename(name)
    if preferred:
        explicit = Path(preferred.strip().strip('"')).expanduser()
        if explicit.is_file():
            return str(explicit.resolve())
        resolved = shutil.which(preferred.strip().strip('"'))
        if resolved:
            return str(Path(resolved).resolve())
    resolved = shutil.which(name) or shutil.which(filename)
    if resolved:
        return str(Path(resolved).resolve())
    for directory in _common_tool_directories(preferred):
        candidate = directory / filename
        if candidate.is_file():
            return str(candidate.resolve())
        if os.name != 'nt':
            candidate = directory / name
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def find_hackrf_transfer(preferred: str = '') -> str | None:
    return find_hackrf_tool('hackrf_transfer', preferred)


def find_hackrf_info(transfer_path: str = '') -> str | None:
    preferred = ''
    if transfer_path:
        path = Path(transfer_path.strip().strip('"')).expanduser()
        preferred = str(path.with_name(_tool_filename('hackrf_info')))
    return find_hackrf_tool('hackrf_info', preferred)


def parse_hackrf_info(output: str) -> list[str]:
    """Extract unique device serial numbers from standard hackrf_info output."""
    serials = re.findall(r'^\s*Serial number:\s*(\S+)\s*$', output,
                         flags=re.IGNORECASE | re.MULTILINE)
    return list(dict.fromkeys(serials))


def discover_hackrf_devices(transfer_path: str = '', runner=subprocess.run) -> tuple[str, list[str], str]:
    """Run hackrf_info and return its path, detected serials, and diagnostic output."""
    info_path = find_hackrf_info(transfer_path)
    if not info_path:
        raise FileNotFoundError(
            'hackrf_info не найден рядом с hackrf_transfer или в каталогах HackRF Tools.')
    kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                  errors='replace', timeout=10)
    if os.name == 'nt':
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    result = runner([info_path], **kwargs)
    output = result.stdout or ''
    serials = parse_hackrf_info(output)
    no_boards = 'No HackRF boards found.' in output
    if result.returncode and not serials and not no_boards:
        details = '\n'.join(output.strip().splitlines()[-4:])
        raise RuntimeError('hackrf_info завершился с кодом '
                           f'{result.returncode}' + (f':\n{details}' if details else ''))
    return info_path, serials, output


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
    resolved = find_hackrf_transfer(value)
    if resolved:
        return resolved
    raise FileNotFoundError(
        'hackrf_transfer не найден. Установите HackRF Tools или укажите полный путь к exe.')


def stop_hackrf_process(process, windows: bool | None = None) -> str:
    """Stop hackrf_transfer gracefully when possible, then fall back to termination."""
    if process.poll() is not None:
        return 'already-stopped'
    windows = os.name == 'nt' if windows is None else windows
    if windows and hasattr(signal, 'CTRL_BREAK_EVENT'):
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=5)
            return 'ctrl-break'
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    if process.poll() is not None:
        return 'ctrl-break-late'
    try:
        process.terminate()
        process.wait(timeout=3)
        return 'terminate'
    except OSError:
        if process.poll() is not None:
            return 'terminate-race'
        raise
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=3)
            return 'kill'
        except (OSError, subprocess.TimeoutExpired):
            return 'still-running'


def cleanup_temp_directory(folder: str | Path, emit=lambda *_: None,
                           remover=shutil.rmtree, sleeper=time.sleep) -> bool:
    """Retry Windows cleanup because hackrf_transfer can release its file handle late."""
    folder = Path(folder)
    last_error = None
    for delay in (0, 0.05, 0.15, 0.35, 0.75, 1.5):
        if delay:
            sleeper(delay)
        try:
            remover(folder)
            return True
        except FileNotFoundError:
            return True
        except OSError as error:
            last_error = error
    emit('log', f'Временный IQ пока занят и оставлен для последующей очистки: '
                f'{folder} ({last_error})')
    return False


def run_transmitter(cfg: EmitterConfig, stop: threading.Event, emit):
    """Generate a cycle, start HackRF Tools, and remain active until Stop."""
    cfg.validate()
    executable = resolve_executable(cfg.executable)
    process = None
    reader = None
    stop_attempted = False
    folder = Path(tempfile.mkdtemp(prefix='tx_emitter_'))
    try:
        iq_path = folder / 'waveform.cs8'
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
            # hackrf_transfer handles CTRL_BREAK and shuts libhackrf/file handles
            # down cleanly. A separate process group lets us target only it.
            kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
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
            method = stop_hackrf_process(process)
            stop_attempted = True
            emit('log', f'hackrf_transfer остановлен ({method}).')
            if method == 'still-running':
                raise RuntimeError(
                    'hackrf_transfer не завершился после Stop. Отключите и снова '
                    'подключите HackRF; при необходимости перезагрузите Windows.')
        reader.join(timeout=2)
        code = process.returncode
        if not stop.is_set() and code:
            raise RuntimeError(f'hackrf_transfer завершился с кодом {code}')
    finally:
        if process is not None and process.poll() is None and not stop_attempted:
            stop_hackrf_process(process)
        if reader is not None and reader.is_alive():
            reader.join(timeout=2)
        cleanup_temp_directory(folder, emit)
        emit('status', 'Остановлено')
