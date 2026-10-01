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
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from modem import Config, MODS, PAYLOAD, PREAMBLE, frame_len, make_frame
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
PACKET_KINDS = ('text', 'image')
IQ_CHUNK_SAMPLES = 262144
REFERENCE_SAMPLES = 65536
PACKET_RATE = 2_000_000
STARTUP_STABILITY_SECONDS = 0.5
START_RETRY_DELAYS = (0.4, 0.8)


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


def wait_for_hackrf_ready(transfer_path: str = '', serial: str = '', *,
                          attempts: int = 20, interval: float = 0.25,
                          runner=subprocess.run, sleeper=time.sleep,
                          cancel: threading.Event | None = None) -> tuple[bool, str]:
    """Wait until ``hackrf_info`` can open the requested device.

    On Windows the ``hackrf_transfer`` process can exit slightly before the
    USB backend makes the board available to a new process. Probing with
    ``hackrf_info`` closes that race without guessing a fixed delay.
    """
    if attempts < 1:
        raise ValueError('Число попыток проверки HackRF должно быть положительным')
    if interval < 0:
        raise ValueError('Интервал проверки HackRF не может быть отрицательным')
    if not find_hackrf_info(transfer_path):
        return False, 'hackrf_info не найден рядом с hackrf_transfer'

    wanted = serial.strip().casefold()
    last_diagnostic = 'HackRF не обнаружен'
    for attempt in range(attempts):
        if cancel is not None and cancel.is_set():
            return False, 'Остановка запрошена'
        try:
            _info_path, serials, output = discover_hackrf_devices(
                transfer_path, runner=runner)
            available = {value.casefold() for value in serials}
            if (wanted in available) if wanted else bool(available):
                return True, output
            if wanted and serials:
                last_diagnostic = (f'HackRF {serial.strip()} не найден; доступны: '
                                   + ', '.join(serials))
            else:
                last_diagnostic = output.strip() or 'HackRF не обнаружен'
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            last_diagnostic = str(error)

        if attempt + 1 < attempts:
            if cancel is not None:
                if cancel.wait(interval):
                    return False, 'Остановка запрошена'
            else:
                sleeper(interval)
    return False, last_diagnostic


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
    packet_kind: str = 'text'

    def validate(self):
        if not HACKRF_MIN_FREQUENCY <= self.frequency_hz <= HACKRF_MAX_FREQUENCY:
            raise ValueError('Частота центра HackRF: 1…6000 МГц')
        if self.sample_rate not in HACKRF_SAMPLE_RATES:
            raise ValueError('Sample rate должен быть одним из 2, 4, 8, 10, 12, 16, 20 MS/s')
        if not 0 < self.bandwidth_hz <= self.sample_rate:
            raise ValueError('Полоса должна быть положительной и не превышать sample rate')
        half = self.bandwidth_hz / 2
        if (self.frequency_hz - half < HACKRF_MIN_FREQUENCY or
                self.frequency_hz + half > HACKRF_MAX_FREQUENCY):
            raise ValueError('Вся заданная полоса должна находиться внутри 1…6000 МГц')
        if not 0 <= self.txvga_gain <= 47 or int(self.txvga_gain) != self.txvga_gain:
            raise ValueError('TX VGA gain HackRF: целое число 0…47 dB')
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError('Цифровая амплитуда должна быть конечным числом ≥0%')
        if self.signal_mode not in SIGNAL_MODES:
            raise ValueError('Неизвестный тип сигнала')
        if self.iq_format not in IQ_FORMATS:
            raise ValueError('Формат IQ-файла: CS8 или CF32')
        if self.signal_mode == 'file':
            iq_file_sample_count(self)
        if self.modulation not in MODS:
            raise ValueError('Неизвестная модуляция')
        if self.packet_kind not in PACKET_KINDS:
            raise ValueError('Содержимое кадра: текст или изображение')
        if self.tuning_mode not in TUNING_MODES:
            raise ValueError('Неизвестный режим частоты')
        if self.operation_mode not in OPERATION_MODES:
            raise ValueError('Неизвестный режим работы')
        if not math.isfinite(self.sweep_period_ms) or self.sweep_period_ms<=0:
            raise ValueError('Период сканирования должен быть конечным числом >0 мс')
        if (not math.isfinite(self.pulse_on_ms) or self.pulse_on_ms<=0 or
                not math.isfinite(self.pulse_off_ms) or self.pulse_off_ms<0):
            raise ValueError('Длительность импульса должна быть >0 мс, паузы ≥0 мс')
        if not self.executable.strip():
            raise ValueError('Укажите путь к hackrf_transfer')
        cycle_sample_count(self)


def _sample_count(sample_rate: int, duration_ms: float, minimum: int = 0) -> int:
    samples = sample_rate * (duration_ms / 1000)
    if not math.isfinite(samples) or samples > sys.maxsize:
        raise ValueError('Длительность IQ-цикла не представима размером файла')
    return max(minimum, int(round(samples)))


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
    samples = _sample_count(cfg.sample_rate, duration_ms, 1)
    if cfg.signal_mode=='packet' and cfg.operation_mode=='continuous' and cfg.tuning_mode=='fixed':
        packet_samples=(len(PREAMBLE)+frame_len(Config(cfg.modulation,True,False,PACKET_RATE)))
        packet_samples=packet_samples*cfg.sample_rate//PACKET_RATE
        samples=((samples+packet_samples-1)//packet_samples)*packet_samples
    return samples


def cycle_sample_count(cfg: EmitterConfig) -> int:
    active = _active_sample_count(cfg)
    if cfg.operation_mode == 'pulse':
        active += _sample_count(cfg.sample_rate, cfg.pulse_off_ms)
    if active > sys.maxsize:
        raise ValueError('Длительность IQ-цикла не представима размером файла')
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


def packet_source(cfg: EmitterConfig) -> Source:
    if cfg.packet_kind=='image':
        return Source(1,bytes((255,0,0))*(16*16),16,16,'Красное изображение HackRF 16×16')
    return Source.text(secrets.token_hex(PAYLOAD//2),'Случайный текст HackRF')


def _packet_template(cfg: EmitterConfig, source=None, transfer=None) -> np.ndarray:
    """Return one real frame made by the existing project modem."""
    source = source if source is not None else packet_source(cfg)
    transfer = secrets.randbits(64) if transfer is None else transfer
    modem_cfg = Config(cfg.modulation, True, False, PACKET_RATE)
    frame = make_frame(source.meta(transfer, 0), source.payload(0), modem_cfg)
    if cfg.sample_rate != PACKET_RATE:
        divisor = math.gcd(cfg.sample_rate, PACKET_RATE)
        frame = resample_poly(frame, cfg.sample_rate // divisor,
                              PACKET_RATE // divisor).astype(np.complex64)
    peak = float(np.max(np.abs(frame)))
    return (frame / max(peak, 1e-9)).astype(np.complex64)


def _frequency_slice(cfg, start, count, total):
    if cfg.tuning_mode == 'fixed':
        return np.zeros(count, np.float64)
    if cfg.signal_mode == 'noise':
        # The instantaneous noise slice occupies 15% of the requested band;
        # its center scans through the rest, filling the band over time.
        span = cfg.bandwidth_hz * 0.85
    else:
        span = cfg.bandwidth_hz * 0.80
    # In pulse mode every RF burst makes one complete pass.  This keeps the
    # scan repeatable even when the requested ON/OFF timings are not a rational
    # multiple of the continuous-sweep period.
    period = (total if cfg.operation_mode == 'pulse' else
              max(1.0, cfg.sample_rate * cfg.sweep_period_ms / 1000))
    position = (np.arange(start,start+count,dtype=np.float64) / period) % 1.0
    triangle = 4.0 * np.abs(position - 0.5) - 1.0
    return triangle * span / 2.0


def frequency_track(cfg: EmitterConfig, sample_count: int) -> np.ndarray:
    """Instantaneous digital frequency offset for one active interval."""
    track = _frequency_slice(cfg,0,sample_count,sample_count)
    # A complete repeated period has zero accumulated phase at the boundary.
    if sample_count:
        track -= float(np.mean(track))
    return track


class WaveformCancelled(Exception):
    pass


def _cancel_if_requested(stop):
    if stop is not None and stop.is_set():
        raise WaveformCancelled()


def _baseband_chunks(cfg, template, active_count, chunk_samples, stop):
    rng=np.random.default_rng(0x4841434B)
    width=cfg.bandwidth_hz*(.15 if cfg.tuning_mode=='sweep' else 1)
    kernel=None
    if cfg.signal_mode=='noise' and width<cfg.sample_rate:
        kernel=firwin(257,width/cfg.sample_rate)
        state=np.zeros(len(kernel)-1,complex)
        # Discard the initial filter transient, retaining its state.
        warm=rng.standard_normal((len(kernel),2))
        _,state=lfilter(kernel,[1.],warm[:,0]+1j*warm[:,1],zi=state)
    for start in range(0,active_count,chunk_samples):
        _cancel_if_requested(stop)
        count=min(chunk_samples,active_count-start)
        if cfg.signal_mode=='noise':
            noise=rng.standard_normal((count,2))
            chunk=noise[:,0]+1j*noise[:,1]
            if kernel is not None:
                chunk,state=lfilter(kernel,[1.],chunk,zi=state)
        else:
            length=len(template)//2 if cfg.signal_mode=='file' and cfg.iq_format=='cs8' else len(template)
            indices=np.arange(start,start+count,dtype=np.int64)%length
            if cfg.signal_mode=='file' and cfg.iq_format=='cs8':
                chunk=(np.clip(template[indices*2].astype(float)/127,-1,1)+
                       1j*np.clip(template[indices*2+1].astype(float)/127,-1,1))
            else:
                chunk=template[indices]
        if not np.all(np.isfinite(chunk)):
            raise ValueError('IQ-файл содержит NaN или бесконечность')
        yield start,chunk


def iter_cycle_chunks(cfg: EmitterConfig, chunk_samples=IQ_CHUNK_SAMPLES, stop=None,
                      source=None, transfer=None):
    """Generate a repeatable cycle with bounded RAM, independent of cycle length."""
    cfg.validate()
    if chunk_samples<1:
        raise ValueError('Размер порции IQ должен быть положительным')
    active_count = _active_sample_count(cfg)
    if cfg.signal_mode == 'packet':
        template = _packet_template(cfg,source,transfer)
    elif cfg.signal_mode=='file':
        template=np.memmap(Path(cfg.iq_path).expanduser(),mode='r',
                           dtype=np.int8 if cfg.iq_format=='cs8' else '<c8')
    else:
        template=None
    # A bounded, uniformly spaced reference sample replaces a full-array quantile.
    reference_indices=np.linspace(0,active_count-1,min(active_count,REFERENCE_SAMPLES),dtype=np.int64)
    levels=np.empty(len(reference_indices),float)
    track_sum=0.
    for start,chunk in _baseband_chunks(cfg,template,active_count,chunk_samples,stop):
        left=np.searchsorted(reference_indices,start)
        right=np.searchsorted(reference_indices,start+len(chunk))
        levels[left:right]=abs(chunk[reference_indices[left:right]-start])
        if cfg.tuning_mode=='sweep':
            track_sum+=float(np.sum(_frequency_slice(cfg,start,len(chunk),active_count),dtype=np.float64))
    reference=max(float(np.quantile(levels,.999)),1e-9)
    track_mean=track_sum/active_count
    phase_cycles=0.
    ramp_count=min(active_count//10,max(1,int(cfg.sample_rate*.001))) if cfg.operation_mode=='pulse' else 0
    for start,chunk in _baseband_chunks(cfg,template,active_count,chunk_samples,stop):
        if cfg.tuning_mode=='sweep':
            track=_frequency_slice(cfg,start,len(chunk),active_count)-track_mean
            phase=phase_cycles+np.cumsum(track,dtype=np.float64)/cfg.sample_rate
            chunk=chunk*np.exp(2j*np.pi*phase)
            phase_cycles=float(phase[-1])%1.
        if ramp_count:
            positions=np.arange(start,start+len(chunk))
            distance=np.minimum(positions,active_count-1-positions)
            ramp=np.ones(len(chunk),float)
            edges=distance<ramp_count
            ramp[edges]=(np.sin(np.pi/2*distance[edges]/(ramp_count-1))**2 if ramp_count>1 else 0.)
            chunk=chunk*ramp
        # CS8 physically saturates at its rails. Permit overdrive without int8 wrap
        # or NaNs when finite but very large user amplitudes overflow a product.
        with np.errstate(over='ignore'):
            real=np.clip(np.asarray(chunk.real,np.float64)/reference*cfg.amplitude,-1,1)
            imag=np.clip(np.asarray(chunk.imag,np.float64)/reference*cfg.amplitude,-1,1)
        yield (real+1j*imag).astype(np.complex64)
    idle_count=cycle_sample_count(cfg)-active_count
    for start in range(0,idle_count,chunk_samples):
        _cancel_if_requested(stop)
        yield np.zeros(min(chunk_samples,idle_count-start),np.complex64)


def build_cycle(cfg: EmitterConfig, source=None, transfer=None) -> np.ndarray:
    """In-memory convenience API; the transmitter writes large cycles in chunks."""
    return np.concatenate(list(iter_cycle_chunks(cfg,source=source,transfer=transfer)))


def write_cycle(cfg, path, stop=None, source=None, transfer=None, chunk_samples=IQ_CHUNK_SAMPLES):
    _cancel_if_requested(stop)
    cfg.validate()
    path=Path(path)
    required=cycle_sample_count(cfg)*2
    if required>shutil.disk_usage(path.parent).free:
        raise ValueError('Недостаточно свободного места для IQ-цикла: требуется '
                         f'{required/1024**3:.2f} GiB')
    samples=0
    with path.open('wb') as stream:
        for chunk in iter_cycle_chunks(cfg,chunk_samples,stop,source,transfer):
            stream.write(complex_to_cs8(chunk))
            samples+=len(chunk)
    return samples


def complex_to_cs8(iq: np.ndarray) -> bytes:
    """Convert normalized complex samples to HackRF signed interleaved I/Q."""
    iq = np.asarray(iq)
    result = np.empty(iq.size * 2, np.int8)
    result[0::2] = np.rint(np.clip(iq.real, -1, 1) * 127).astype(np.int8)
    result[1::2] = np.rint(np.clip(iq.imag, -1, 1) * 127).astype(np.int8)
    return result.tobytes()


def select_filter_bandwidth(cfg: EmitterConfig) -> int:
    wanted = max(1_750_000,int(cfg.bandwidth_hz * 1.20))
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


def interrupt_windows_console(pid: int):
    # An isolated helper targets only hackrf_transfer's private console.
    # hackrf_transfer handles CTRL_C_EVENT; CTRL_BREAK terminates it abruptly.
    code = (
        "import ctypes,sys,time\n"
        "k=ctypes.WinDLL('kernel32',use_last_error=True)\n"
        "k.FreeConsole()\n"
        "if not k.AttachConsole(int(sys.argv[1])): raise ctypes.WinError(ctypes.get_last_error())\n"
        "try:\n"
        " if not k.SetConsoleCtrlHandler(None,True): raise ctypes.WinError(ctypes.get_last_error())\n"
        " if not k.GenerateConsoleCtrlEvent(0,0): raise ctypes.WinError(ctypes.get_last_error())\n"
        " time.sleep(.1)\n"
        "finally: k.FreeConsole()\n"
    )
    subprocess.run([sys.executable, '-c', code, str(pid)], check=True, timeout=3,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def stop_hackrf_process(process, windows: bool | None = None) -> str:
    """Stop hackrf_transfer gracefully when possible, then fall back to termination."""
    if process.poll() is not None:
        return 'already-stopped'
    windows = os.name == 'nt' if windows is None else windows
    if windows:
        try:
            interrupt_windows_console(process.pid)
            process.wait(timeout=5)
            return 'ctrl-c'
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if process.poll() is not None:
        return 'interrupt-late'
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


def _hackrf_failure_message(code: int, output_lines: list[str]) -> str:
    details = '; '.join(output_lines[-3:])
    return (f'hackrf_transfer завершился с кодом {code}'
            + (f': {details}' if details else ''))


def run_transmitter(cfg: EmitterConfig, stop: threading.Event, emit):
    """Generate a cycle, start HackRF Tools, and remain active until Stop."""
    cfg.validate()
    executable = resolve_executable(cfg.executable)
    emit('status', 'Ожидание готовности HackRF…')
    ready, diagnostic = wait_for_hackrf_ready(
        executable, cfg.serial, cancel=stop)
    if stop.is_set():
        emit('status', 'Остановлено')
        return
    if not ready:
        details = '; '.join(diagnostic.strip().splitlines()[-3:])
        raise RuntimeError(
            'HackRF не готов к запуску. Проверьте подключение и драйвер USB'
            + (f': {details}' if details else '.'))
    process = None
    reader = None
    stop_attempted = False
    folder = Path(tempfile.mkdtemp(prefix='tx_emitter_'))
    try:
        iq_path = folder / 'waveform.cs8'
        emit('status', 'Формирование IQ…')
        source=packet_source(cfg) if cfg.signal_mode=='packet' else None
        transfer=secrets.randbits(64) if source is not None else None
        if source is not None:
            emit('log',f'Кадр {transfer:016x}: '+source.label+'. DATA №0 повторяется до Stop; END не отправляется.')
            if not source.kind:
                emit('log','Текст кадра: '+source.raw.decode('utf-8'))
        if cfg.amplitude>1:
            emit('log',f'Цифровая амплитуда {cfg.amplitude:g}×: значения за пределами CS8 насыщаются.')
        try:
            samples=write_cycle(cfg,iq_path,stop,source,transfer)
        except WaveformCancelled:
            return
        if stop.is_set():
            return
        duration_ms = samples / cfg.sample_rate * 1000
        emit('log', f'IQ-цикл: {samples:,} отсчётов, {duration_ms:.1f} мс, '
                    f'{iq_path.stat().st_size / 1024 / 1024:.2f} MiB.')
        command = build_command(cfg, iq_path, executable)
        emit('log', 'Запуск: ' + subprocess.list2cmdline(command))
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, errors='replace', bufsize=1)
        if os.name == 'nt':
            # A private hidden console allows Ctrl+C without signaling the GUI.
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startup
        output_lines: list[str] = []
        for launch_attempt in range(len(START_RETRY_DELAYS) + 1):
            output_lines = []
            process = subprocess.Popen(command, **kwargs)

            def read_output(current_process=process, current_output=output_lines):
                if current_process.stdout is not None:
                    for line in current_process.stdout:
                        line = line.strip()
                        if line:
                            current_output.append(line)
                            emit('log', line)

            reader = threading.Thread(target=read_output, name='hackrf_transfer output',
                                      daemon=True)
            reader.start()
            deadline = time.monotonic() + STARTUP_STABILITY_SECONDS
            while process.poll() is None and time.monotonic() < deadline:
                if stop.wait(0.05):
                    break
            code = process.poll()
            if code is None or stop.is_set():
                break
            reader.join(timeout=2)
            if code != 1 or launch_attempt >= len(START_RETRY_DELAYS):
                raise RuntimeError(_hackrf_failure_message(code, output_lines))
            delay = START_RETRY_DELAYS[launch_attempt]
            emit('log', f'hackrf_transfer вернул код 1 при запуске; '
                 f'повтор через {delay:g} с.')
            emit('status', 'Повторный захват HackRF…')
            if stop.wait(delay):
                break

        if stop.is_set() and process.poll() is not None:
            return
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
            emit('status', 'Ожидание освобождения USB…')
            ready, diagnostic = wait_for_hackrf_ready(executable, cfg.serial)
            if ready:
                emit('log', 'HackRF освобождён и готов к повторному запуску.')
            else:
                details = '; '.join(diagnostic.strip().splitlines()[-3:])
                emit('log', 'HackRF пока не подтвердил готовность после Stop'
                     + (f': {details}' if details else '.'))
        reader.join(timeout=2)
        code = process.returncode
        if not stop.is_set() and code:
            raise RuntimeError(_hackrf_failure_message(code, output_lines))
    finally:
        if process is not None and process.poll() is None and not stop_attempted:
            stop_hackrf_process(process)
        if reader is not None and reader.is_alive():
            reader.join(timeout=2)
        cleanup_temp_directory(folder, emit)
        emit('status', 'Остановлено')
