"""Packet PHY. No SDR or GUI imports; deterministic and independently testable."""
from __future__ import annotations
import math
import json
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from quality import pilot_energies, vector_energies

from modulations import (MODS, PHASE_MODS, SPS, bits, pack, constellation, symbols, bit_symbols,
                         decisions, decision_error_energies, to_wave, fsk as fsk_mod)

DATA_SYMBOLS = 128
PILOT_SYMBOLS = 32
CP_SYMBOLS = 16
PAYLOAD = 768
MAX_BYTES = 32 * 1024 * 1024
PHY_VERSION = 2
MAGIC = bytes.fromhex('d391c5a76e2b48f0')
HEADER = struct.Struct('!BBBBQIIIHHH16sI')
HEADER_BYTES = HEADER.size + 4
H_SYMBOLS = HEADER_BYTES * 8 * 3
H_BLOCKS = math.ceil(H_SYMBOLS / DATA_SYMBOLS)


def whitening(n, seed):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8)


def whiten(data, seed):
    return (np.frombuffer(data, np.uint8) ^ whitening(len(data), seed)).tobytes()


def tile_shape(payload_bytes):
    pixels = payload_bytes // 3
    width = math.isqrt(pixels)
    while pixels % width:
        width -= 1
    return width, pixels // width


@lru_cache(maxsize=64)
def pilot_symbols(cfg):
    pts, _ = constellation(cfg.mod, cfg)
    labels = np.random.default_rng(cfg.pilot_seed).integers(0, len(pts), cfg.pilot_symbols)
    labels[0] = labels[-1] = len(pts)-1  # At least two active OOK pilots for phase fitting.
    return pts[labels]


@lru_cache(maxsize=64)
def pilot_wave(cfg):
    return to_wave(pilot_symbols(cfg), cfg.mod == '2-FSK', cfg.fsk_cycles)


@lru_cache(maxsize=64)
def preamble_wave(cfg):
    if not cfg.preamble:
        return np.empty(0, np.complex64)
    pts, _ = constellation(cfg.mod, cfg)
    z = pts[np.random.default_rng(cfg.preamble_seed).integers(0, len(pts), cfg.preamble_symbols)]
    return to_wave(z, cfg.mod == '2-FSK', cfg.fsk_cycles)


@lru_cache(maxsize=64)
def sync_wave(cfg):
    # Keep 64 symbols for robust acquisition for every constellation order.
    k = constellation(cfg.mod, cfg)[1]
    return to_wave(symbols(MAGIC*k, cfg.mod, cfg), cfg.mod == '2-FSK', cfg.fsk_cycles)


@lru_cache(maxsize=32)
def channel_estimator(cfg):
    known = pilot_wave(cfg)
    taps = 9
    indices = np.arange(taps-1, len(known))[:, None] - np.arange(taps)
    x = known[indices]
    operator = np.linalg.solve(x.conj().T@x + 0.05*np.eye(taps), x.conj().T)
    return x, operator, taps


def block_len(cfg):
    return (cfg.pilot_symbols + cfg.data_symbols + (cfg.cp_symbols if cfg.cp else 0)) * SPS


def header_blocks(cfg):
    return math.ceil(math.ceil(H_SYMBOLS/constellation(cfg.mod, cfg)[1])/cfg.data_symbols)


def encode_blocks(z, cfg):
    z = np.pad(z, (0, (-len(z)) % cfg.data_symbols), constant_values=constellation(cfg.mod,cfg)[0][0])
    out = []
    for row in z.reshape(-1, cfg.data_symbols):
        wave = to_wave(np.r_[pilot_symbols(cfg), row], cfg.mod == '2-FSK', cfg.fsk_cycles)
        if cfg.cp:
            wave = np.r_[wave[-cfg.cp_symbols*SPS:], wave]
        out.append(wave)
    return np.concatenate(out).astype(np.complex64)


@dataclass(frozen=True)
class Config:
    mod: str = 'BPSK'
    preamble: bool = True
    cp: bool = False
    sample_rate: int = 1_000_000
    cfo_range: float = 40_000
    preamble_symbols: int = 256
    preamble_seed: int = 331
    cp_symbols: int = 16
    pilot_symbols: int = 32
    pilot_seed: int = 7729
    data_symbols: int = 128
    payload_bytes: int = 768
    phase_deg: float = 0.
    ook_low: float = 0.
    fsk_deviation_hz: float = 125_000.

    @property
    def fsk_cycles(self):
        return self.fsk_deviation_hz * SPS / self.sample_rate

    def profile_tag(self):
        # Receiver search range and settings of disabled sections are local.
        values = {name: getattr(self, name) for name in (
            'mod', 'preamble', 'cp', 'sample_rate',
            'pilot_symbols', 'pilot_seed', 'data_symbols', 'payload_bytes')}
        if self.preamble:
            values.update(preamble_symbols=self.preamble_symbols, preamble_seed=self.preamble_seed)
        if self.cp:
            values['cp_symbols'] = self.cp_symbols
        if self.mod == 'ASK / OOK':
            values['ook_low'] = float(self.ook_low)
        elif self.mod == '2-FSK':
            values['fsk_cycles'] = float(self.fsk_cycles)
        else:
            # Preserve default wire profiles for fixed BPSK/QAM constellations.
            values['phase_deg'] = float(self.phase_deg) if self.mod in PHASE_MODS else 0.
        return zlib.crc32(json.dumps(values, sort_keys=True).encode())

    def validate(self):
        if self.mod not in MODS:
            raise ValueError('Неизвестная модуляция')
        if self.sample_rate not in (1_000_000, 2_000_000):
            raise ValueError('Sample rate: 1 или 2 MS/s')
        if not 0 <= self.cfo_range <= 100_000:
            raise ValueError('Поиск CFO: 0…100000 Гц')
        for name, low, high in (('preamble_symbols',64,4096), ('cp_symbols',1,256),
                               ('pilot_symbols',8,256), ('data_symbols',32,1024),
                               ('payload_bytes',48,12288), ('preamble_seed',0,2**32-1),
                               ('pilot_seed',0,2**32-1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name}: целое число {low}…{high}')
        # Both tones must fit after the receiver's Fs/4 digital frequency shift.
        if self.mod == '2-FSK' and (not math.isfinite(self.fsk_deviation_hz) or
                not 0 < self.fsk_deviation_hz < self.sample_rate/4):
            raise ValueError(f'Девиация FSK: больше 0 и меньше {self.sample_rate/4_000_000:g} МГц')
        if self.cp_symbols > self.pilot_symbols+self.data_symbols:
            raise ValueError('CP не должен быть длиннее блока пилотов и данных')
        if self.mod in PHASE_MODS and (
                not math.isfinite(self.phase_deg) or not -180 <= self.phase_deg <= 180):
            raise ValueError('Поворот созвездия: −180…180°')
        if self.mod == 'ASK / OOK' and (not math.isfinite(self.ook_low) or not 0 <= self.ook_low <= 0.9):
            raise ValueError('Нижний уровень ASK: 0…0.9 от верхнего')


# Compatibility names describe only the default BPSK profile.
PREAMBLE = preamble_wave(Config())
SYNC = sync_wave(Config())



@dataclass(frozen=True)
class Meta:
    kind: int
    transfer: int
    seq: int
    total: int
    size: int
    width: int
    height: int
    length: int
    digest: bytes
    end: bool = False
    payload_bytes: int = PAYLOAD

    def encode(self, cfg):
        flags = int(cfg.preamble) | (int(cfg.cp)<<1) | (int(self.end)<<2)
        h = HEADER.pack(PHY_VERSION, self.kind, MODS.index(cfg.mod), flags, self.transfer,
                        self.seq, self.total, self.size, self.width, self.height,
                        self.length, self.digest, cfg.profile_tag())
        return h + struct.pack('!I', zlib.crc32(h))

    @classmethod
    def decode(cls, raw, cfg):
        h, check = raw[:-4], raw[-4:]
        if len(raw) != HEADER_BYTES or zlib.crc32(h) != int.from_bytes(check, 'big'):
            raise ValueError('header CRC')
        ver, kind, mod, flags, tid, seq, total, size, w, h, length, digest, profile = HEADER.unpack(h)
        if ver != PHY_VERSION or profile != cfg.profile_tag() or kind not in (0, 1) or mod != MODS.index(cfg.mod):
            raise ValueError('header version/type/modulation')
        if flags & 3 != int(cfg.preamble) | (int(cfg.cp)<<1) or flags & ~7:
            raise ValueError('header flags')
        tw, th = tile_shape(cfg.payload_bytes)
        expected = math.ceil(w/tw)*math.ceil(h/th) if kind else math.ceil(size/cfg.payload_bytes)
        if not 0 < size <= MAX_BYTES or not 0 < total <= 65536 or total != expected:
            raise ValueError('header size/count')
        if kind and (not w or not h or size != w*h*3):
            raise ValueError('header RGB dimensions')
        if not kind and (w or h):
            raise ValueError('header text dimensions')
        end = bool(flags & 4)
        want_length = 0 if end else (tw*th*3 if kind else min(cfg.payload_bytes, size-seq*cfg.payload_bytes))
        if (end and seq != total) or (not end and not 0 <= seq < total):
            raise ValueError('header sequence')
        if length != want_length:
            raise ValueError('header payload length')
        return cls(kind, tid, seq, total, size, w, h, length, digest, end, cfg.payload_bytes)


def payload_seed(meta):
    return (meta.transfer ^ (meta.seq*0x9E3779B1)) & 0xFFFFFFFF


def make_frame(meta, data, cfg):
    cfg.validate()
    if len(data) != meta.length or meta.payload_bytes != cfg.payload_bytes:
        raise ValueError('payload length/profile mismatch')
    hb = np.repeat(bits(whiten(meta.encode(cfg), 9121)), 3)
    header = encode_blocks(bit_symbols(hb, cfg.mod, cfg), cfg)
    # Fixed frame length within a profile, including END.
    padded = data.ljust(cfg.payload_bytes, b'\0')
    body = padded + struct.pack('!I', zlib.crc32(data))
    encoded = encode_blocks(symbols(whiten(body, payload_seed(meta)), cfg.mod, cfg), cfg)
    return np.r_[preamble_wave(cfg), sync_wave(cfg), header, encoded].astype(np.complex64)


def frame_len(cfg):
    k = constellation(cfg.mod, cfg)[1]
    n = math.ceil(math.ceil((cfg.payload_bytes+4)*8/k)/cfg.data_symbols)
    # From marker; the optional preamble is consumed by the stream decoder.
    return len(sync_wave(cfg)) + (header_blocks(cfg)+n)*block_len(cfg)


def align_block_timing(wave, count, cfg, known):
    """Follow sample-clock drift using each block's existing pilot sequence."""
    length = block_len(cfg)
    prefix = cfg.cp_symbols*SPS if cfg.cp else 0
    radius = 2*SPS
    aligned = []
    for index in range(count):
        nominal = index*length + prefix
        left = max(0, nominal-radius)
        right = min(len(wave)-len(known), nominal+radius)
        windows = np.lib.stride_tricks.sliding_window_view(
            wave[left:right+len(known)], len(known))
        energy = np.sum(abs(windows)**2, axis=1)
        scores = abs(windows@known.conj())**2 / np.maximum(energy, 1e-15)
        pilot = left + int(np.argmax(scores))
        start = pilot-prefix
        end = start+length
        block = wave[max(0, start):min(len(wave), end)]
        # Only CP or trailing PHY padding can extend past the available slice.
        block = np.pad(block, (max(0, -start), max(0, end-len(wave))))
        aligned.append(block)
    return np.concatenate(aligned)


def decode_blocks(wave, count, cfg, fsk=False, equalize=False, quality=None):
    """Per-block pilots track timing/phase/gain. Optional SC-FDE for linear payloads.

    CP length is configurable. LS channel length 9 samples; pilot rows avoid unknown
    preceding data. Regularized frequency-domain inverse operates on each block.
    FSK uses two noncoherent correlators (no channel inverse).
    """
    length = block_len(cfg)
    known = pilot_wave(cfg)
    known_energy = np.vdot(known, known)
    wave = align_block_timing(wave, count, cfg, known)
    if equalize and cfg.cp and not fsk:
        x, operator, taps = channel_estimator(cfg)
    out = []
    for block in wave[:count*length].reshape(count, length):
        if cfg.cp:
            block = block[cfg.cp_symbols*SPS:]
        pilot_rx = block[:len(known)]
        if quality is not None:
            # Measure before estimating phase/gain/channel from these same pilots.
            rows = pilot_rx.reshape(cfg.pilot_symbols, SPS)
            if fsk:
                observed = fsk_mod.observe(rows, cfg.fsk_cycles)
                reference = fsk_mod.reference(pilot_symbols(cfg), SPS, cfg.fsk_cycles)
            else:
                observed = rows[:,2:-2].mean(axis=1)
                reference = pilot_symbols(cfg)
            quality.append(pilot_energies(observed, reference))
        # Estimate residual frequency on known pilot waveforms, in sample domain.
        despread = pilot_rx * known.conj()
        grouped = despread.reshape(cfg.pilot_symbols, SPS).mean(axis=1)
        active = np.mean(abs(known.reshape(-1, SPS))**2, axis=1) > 1e-12
        phase = np.unwrap(np.angle(grouped[active]))
        times = (np.arange(cfg.pilot_symbols)*SPS+(SPS-1)/2)[active]
        slope, intercept = np.polyfit(times, phase, 1)
        # Fixed CFO was acquired on header; suppress noisy estimates for each short block.
        slope = float(np.clip(slope, -0.01, 0.01))
        block = block * np.exp(-1j*(slope*np.arange(len(block))+intercept))
        gain = np.vdot(known, block[:len(known)]) / known_energy
        if abs(gain) < 1e-9:
            gain = 1e-9
        if equalize and cfg.cp and not fsk:
            y = block[taps-1:len(known)]
            hh = operator@y
            err = np.mean(abs(y-x@hh)**2)
            hf = np.fft.fft(hh, len(block))
            floor = max(float(err), float(abs(gain)**2)*1e-4)
            block = np.fft.ifft(np.fft.fft(block)*hf.conj()/(abs(hf)**2+floor))
        else:
            block = block / gain
        rows = block[len(known):].reshape(cfg.data_symbols, SPS)
        if fsk:
            out.append(fsk_mod.observe(rows, cfg.fsk_cycles))
        else:
            # All samples after SC-FDE; skip symbol edges otherwise to reduce ISI.
            out.append(rows.mean(axis=1) if equalize and cfg.cp else rows[:, 2:-2].mean(axis=1))
    return np.concatenate(out)


@dataclass
class Decoded:
    meta: Meta
    payload: bytes
    crc_ok: bool
    observations: np.ndarray
    cfo: float
    score: float
    sir_signal_energy: float = 0.
    sir_interference_energy: float = 0.
    sir_vector_count: int = 0
    pilot_signal_energy: float = 0.
    pilot_disturbance_energy: float = 0.
    pilot_sample_count: int = 0
    crc_signal_energy: float = 0.
    crc_error_energy: float = 0.
    crc_symbol_count: int = 0


def decode_frame(wave, cfg, cfo, score=1.):
    wave = wave * np.exp(-2j*np.pi*cfo*np.arange(len(wave))/cfg.sample_rate)
    sync = sync_wave(cfg)
    d = (wave[:len(sync)]*sync.conj()).reshape(-1, SPS).mean(axis=1)
    active = np.mean(abs(sync.reshape(-1, SPS))**2, axis=1) > 1e-12
    phase = np.unwrap(np.angle(d[active]))
    times = (np.arange(len(d))*SPS+(SPS-1)/2)[active]
    slope, intercept = np.polyfit(times, phase, 1)
    wave *= np.exp(-1j*(slope*np.arange(len(wave))+intercept))
    cfo += slope*cfg.sample_rate/(2*np.pi)
    cursor = len(sync)
    nh = header_blocks(cfg)
    hsize = nh*block_len(cfg)
    hz = decode_blocks(wave[cursor:cursor+hsize], nh, cfg, cfg.mod == '2-FSK')
    repeated = decisions(hz, cfg.mod, cfg)[:H_SYMBOLS]
    hb = repeated.reshape(-1, 3).sum(axis=1) >= 2
    meta = Meta.decode(whiten(pack(hb), 9121), cfg)
    cursor += hsize
    k = constellation(cfg.mod, cfg)[1]
    nsyms = math.ceil((cfg.payload_bytes+4)*8/k)
    nblocks = math.ceil(nsyms/cfg.data_symbols)
    pilot_quality = []
    obs = decode_blocks(wave[cursor:], nblocks, cfg, cfg.mod == '2-FSK',
                        equalize=True, quality=pilot_quality)[:nsyms]
    b = decisions(obs, cfg.mod, cfg)
    data = whiten(pack(b[:(cfg.payload_bytes+4)*8]), payload_seed(meta))
    payload = data[:meta.length]
    valid = zlib.crc32(payload) == int.from_bytes(data[cfg.payload_bytes:cfg.payload_bytes+4], 'big')
    measured = meta.length*8//k
    sir_signal, sir_interference = decision_error_energies(obs[:measured], cfg.mod, cfg)
    pilot_signal = sum(row[0] for row in pilot_quality)
    pilot_error = sum(row[1] for row in pilot_quality)
    pilot_count = sum(row[2] for row in pilot_quality)
    crc_signal = crc_error = 0.
    if valid and measured:
        # CRC-validated data can be reconstructed locally; no external reference.
        encoded = whiten(payload.ljust(cfg.payload_bytes,b'\0')+
                         struct.pack('!I',zlib.crc32(payload)),payload_seed(meta))
        expected = symbols(encoded,cfg.mod,cfg)[:measured]
        if cfg.mod == '2-FSK':
            expected = fsk_mod.reference(expected, SPS, cfg.fsk_cycles)
        crc_signal,crc_error = vector_energies(obs[:measured],expected)
    return Decoded(meta, payload, valid, obs, cfo, score,
                   sir_signal, sir_interference, measured,
                   pilot_signal,pilot_error,pilot_count,crc_signal,crc_error,
                   measured if valid else 0)


class StreamDecoder:
    """Bounded overlap buffer; accepts arbitrary chunks, never assumes packet alignment."""
    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.buffer = np.empty(0, np.complex64)
        self.prefix_len = len(preamble_wave(cfg))
        self.sync = preamble_wave(cfg)[:64*SPS] if cfg.preamble else sync_wave(cfg)
        self.length = frame_len(cfg) + self.prefix_len
        self.header_failures = 0
        self.candidates = 0
        self.lock_cfo = None
        self.scanned = 0
        self.short = self.sync[:32*SPS]
        self.step = cfg.sample_rate / len(self.short) / 2
        self.grid = np.arange(-math.ceil(cfg.cfo_range/self.step),
                              math.ceil(cfg.cfo_range/self.step)+1)*self.step
        self._grid_freqs = frozenset(self.grid)
        self._template_ffts = {}
        self._fft_size = None
        self._short_time = np.arange(len(self.short))/cfg.sample_rate
        self._sync_time = np.arange(len(self.sync))/cfg.sample_rate

    def _template_fft(self, freq, size):
        # Cache only the fixed search grid at one FFT size. Locked CFOs may
        # change every packet and must not grow the cache without a bound.
        if size != self._fft_size:
            self._template_ffts.clear()
            self._fft_size = size
        cached = self._template_ffts.get(freq)
        if cached is None:
            template = self.short*np.exp(2j*np.pi*freq*self._short_time)
            cached = fft(template.conj()[::-1], size)
            if freq in self._grid_freqs:
                self._template_ffts[freq] = cached
        return cached

    def reset(self):
        self.buffer = np.empty(0, np.complex64)
        self.lock_cfo = None

    def feed(self, iq, stop=None):
        self.buffer = np.r_[self.buffer, np.asarray(iq, np.complex64)]
        result = []
        short, step = self.short, self.step
        while len(self.buffer) >= self.length:
            if stop is not None and stop.is_set():
                break
            # Search only starts for which a whole frame is already present.
            positions = min(len(self.buffer)-self.length+1, 32768)
            segment = self.buffer[:positions+len(short)-1]
            energy = np.convolve(abs(segment)**2, np.ones(len(short)), 'valid')
            denom = np.maximum(energy*float(np.vdot(short, short).real), 1e-15)
            # FFT roundoff over exact silence must not win normalized correlation.
            energetic = energy > max(float(energy.max())*1e-6, 1e-15)
            if not np.any(energetic):
                # Every score would be zero under the existing energy mask.
                self.buffer = self.buffer[positions:]
                continue
            grid = self.grid
            if self.lock_cfo is not None:
                grid = np.r_[self.lock_cfo, self.lock_cfo-step, self.lock_cfo+step, grid]
            best = np.zeros(positions)
            freqs = np.zeros(positions)
            fft_size = next_fast_len(len(segment)+len(short)-1)
            segment_fft = fft(segment, fft_size)
            for freq in grid:
                if stop is not None and stop.is_set():
                    break
                corr = ifft(segment_fft*self._template_fft(freq, fft_size))[
                    len(short)-1:len(short)-1+positions]
                score = np.where(energetic, abs(corr)**2/denom, 0)
                improve = score > best
                best[improve], freqs[improve] = score[improve], freq
                # Once locked, good prefix detection normally avoids full CFO sweep.
                if self.lock_cfo is not None and freq == self.lock_cfo and best.max() > 0.75:
                    break
            peaks, _ = find_peaks(np.r_[0, best, 0], height=0.48, distance=len(short)//2)
            peaks = peaks-1
            accepted = False
            # Highest 24 plausible peaks, attempted in time order.
            if len(peaks) > 24:
                peaks = peaks[np.argsort(best[peaks])[-24:]]
            for pos in sorted(peaks):
                self.candidates += 1
                fragment = self.buffer[pos:pos+self.length]
                base = freqs[pos]
                t = self._sync_time
                objective = lambda f: -abs(np.vdot(self.sync*np.exp(2j*np.pi*f*t), fragment[:len(self.sync)]))**2
                fine = minimize_scalar(objective, bounds=(base-step, base+step), method='bounded').x
                # First maximum can be displaced by one sample after analog filtering.
                for delta in (0, -1, 1, -2, 2):
                    start = pos+delta
                    if start < 0 or start+self.length > len(self.buffer):
                        continue
                    try:
                        packet = decode_frame(self.buffer[start+self.prefix_len:start+self.length], self.cfg, fine, best[pos])
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    result.append(packet)
                    self.lock_cfo = packet.cfo
                    self.buffer = self.buffer[start+self.length:]
                    accepted = True
                    break
                if accepted:
                    break
                self.header_failures += 1
            if not accepted:
                self.buffer = self.buffer[positions:]
        return result
