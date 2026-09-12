"""Packet PHY. No SDR or GUI imports; deterministic and independently testable."""
from __future__ import annotations
import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate, find_peaks
from scipy.optimize import minimize_scalar

MODS = ('ASK / OOK', '2-FSK', 'BPSK', 'QPSK', '8-PSK', 'QAM-4', 'QAM-8', 'QAM-16')
SPS = 8
DATA_SYMBOLS = 128
PILOT_SYMBOLS = 32
CP_SYMBOLS = 16
PAYLOAD = 768
MAX_BYTES = 32 * 1024 * 1024
MAGIC = bytes.fromhex('d391c5a76e2b48f0')
HEADER = struct.Struct('!BBBBQIIIHHH16s')
HEADER_BYTES = HEADER.size + 4
H_SYMBOLS = HEADER_BYTES * 8 * 3
H_BLOCKS = math.ceil(H_SYMBOLS / DATA_SYMBOLS)
SYNC_BITS = np.unpackbits(np.frombuffer(MAGIC, np.uint8))
SYNC = np.repeat(2.0 * SYNC_BITS - 1.0, SPS).astype(np.complex64)
PILOTS = (2 * np.random.default_rng(7729).integers(0, 2, PILOT_SYMBOLS) - 1).astype(np.complex64)
PREAMBLE = np.repeat(2 * np.random.default_rng(331).integers(0, 2, 256) - 1, SPS).astype(np.complex64)


def bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def pack(b):
    return np.packbits(np.asarray(b, np.uint8)).tobytes()


def whitening(n, seed):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8)


def whiten(data, seed):
    return (np.frombuffer(data, np.uint8) ^ whitening(len(data), seed)).tobytes()


def constellation(mod):
    if mod == 'ASK / OOK':
        return np.array([0, np.sqrt(2)], complex), 1
    if mod == 'BPSK':
        return np.array([-1, 1], complex), 1
    if mod in ('QPSK', '8-PSK'):
        k = 2 if mod == 'QPSK' else 3
        # Integer index is the bit label; adjacent phase points have Gray labels.
        pts = np.empty(2**k, complex)
        for p in range(2**k):
            pts[p ^ (p >> 1)] = np.exp(1j * (2*np.pi*p/(2**k) + np.pi/(2**k)))
        return pts, k
    if mod.startswith('QAM-'):
        k = int(math.log2(int(mod.split('-')[1])))
        ki, kq = (k+1)//2, k//2
        def levels(k):
            result = np.empty(2**k)
            for p in range(2**k):
                result[p ^ (p >> 1)] = 2*p - (2**k-1)
            return result
        a, b = levels(ki), levels(kq)
        pts = np.array([i+1j*q for i in a for q in b])
        return pts / np.sqrt(np.mean(abs(pts)**2)), k
    if mod == '2-FSK':
        return np.array([-1, 1]), 1
    raise ValueError(mod)


def symbols(data, mod):
    pts, k = constellation(mod)
    b = bits(data)
    b = np.pad(b, (0, (-len(b)) % k)).reshape(-1, k)
    labels = b @ (1 << np.arange(k-1, -1, -1))
    return pts[labels]


def decisions(z, mod):
    pts, k = constellation(mod)
    labels = np.argmin(abs(z[:, None] - pts[None, :])**2, axis=1)
    return ((labels[:, None] >> np.arange(k-1, -1, -1)) & 1).astype(np.uint8).ravel()


def to_wave(z, fsk=False):
    if fsk:
        return np.exp(2j*np.pi*np.asarray(z)[:, None]*np.arange(SPS)/SPS).ravel().astype(np.complex64)
    return np.repeat(z, SPS).astype(np.complex64)


def block_len(cp):
    return (PILOT_SYMBOLS + DATA_SYMBOLS + (CP_SYMBOLS if cp else 0)) * SPS


def encode_blocks(z, cp, fsk=False):
    z = np.pad(z, (0, (-len(z)) % DATA_SYMBOLS))
    out = []
    for row in z.reshape(-1, DATA_SYMBOLS):
        wave = to_wave(np.r_[PILOTS.real if fsk else PILOTS, row], fsk)
        if cp:
            wave = np.r_[wave[-CP_SYMBOLS*SPS:], wave]
        out.append(wave)
    return np.concatenate(out).astype(np.complex64)


@dataclass(frozen=True)
class Config:
    mod: str = 'BPSK'
    preamble: bool = True
    cp: bool = False
    sample_rate: int = 1_000_000
    cfo_range: float = 40_000

    def validate(self):
        if self.mod not in MODS:
            raise ValueError('Неизвестная модуляция')
        if self.sample_rate not in (1_000_000, 2_000_000):
            raise ValueError('Sample rate: 1 или 2 MS/s')
        if not 0 <= self.cfo_range <= 100_000:
            raise ValueError('Поиск CFO: 0…100000 Гц')


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

    def encode(self, cfg):
        flags = int(cfg.preamble) | (int(cfg.cp)<<1) | (int(self.end)<<2)
        h = HEADER.pack(1, self.kind, MODS.index(cfg.mod), flags, self.transfer,
                        self.seq, self.total, self.size, self.width, self.height,
                        self.length, self.digest)
        return h + struct.pack('!I', zlib.crc32(h))

    @classmethod
    def decode(cls, raw, cfg):
        h, check = raw[:-4], raw[-4:]
        if len(raw) != HEADER_BYTES or zlib.crc32(h) != int.from_bytes(check, 'big'):
            raise ValueError('header CRC')
        ver, kind, mod, flags, tid, seq, total, size, w, h, length, digest = HEADER.unpack(h)
        if ver != 1 or kind not in (0, 1) or mod != MODS.index(cfg.mod):
            raise ValueError('header version/type/modulation')
        if flags & 3 != int(cfg.preamble) | (int(cfg.cp)<<1) or flags & ~7:
            raise ValueError('header flags')
        expected = math.ceil(w/16)*math.ceil(h/16) if kind else math.ceil(size/PAYLOAD)
        if not 0 < size <= MAX_BYTES or not 0 < total <= 65536 or total != expected:
            raise ValueError('header size/count')
        if kind and (not w or not h or size != w*h*3):
            raise ValueError('header RGB dimensions')
        if not kind and (w or h):
            raise ValueError('header text dimensions')
        end = bool(flags & 4)
        want_length = 0 if end else (PAYLOAD if kind else min(PAYLOAD, size-seq*PAYLOAD))
        if (end and seq != total) or (not end and not 0 <= seq < total):
            raise ValueError('header sequence')
        if length != want_length:
            raise ValueError('header payload length')
        return cls(kind, tid, seq, total, size, w, h, length, digest, end)


def payload_seed(meta):
    return (meta.transfer ^ (meta.seq*0x9E3779B1)) & 0xFFFFFFFF


def make_frame(meta, data, cfg):
    if len(data) != meta.length:
        raise ValueError('payload length mismatch')
    hb = np.repeat(bits(whiten(meta.encode(cfg), 9121)), 3)
    header = encode_blocks(2.0*hb-1, cfg.cp)
    # Fixed PHY length, including END, so streaming decoder never trusts a corrupt length.
    padded = data.ljust(PAYLOAD, b'\0')
    body = padded + struct.pack('!I', zlib.crc32(data))
    encoded = encode_blocks(symbols(whiten(body, payload_seed(meta)), cfg.mod), cfg.cp, cfg.mod == '2-FSK')
    return np.r_[PREAMBLE if cfg.preamble else np.empty(0), SYNC, header, encoded].astype(np.complex64)


def frame_len(cfg):
    k = constellation(cfg.mod)[1]
    n = math.ceil(math.ceil((PAYLOAD+4)*8/k)/DATA_SYMBOLS)
    # From header marker; optional preamble is consumed as unframed leading samples.
    return len(SYNC) + (H_BLOCKS+n)*block_len(cfg.cp)


def decode_blocks(wave, count, cfg, fsk=False, equalize=False):
    """Per-block pilots track phase/gain. Optional SC-FDE for linear payloads.

    CP spans 128 samples. LS channel length 9 samples; pilot rows avoid unknown
    preceding data. Regularized frequency-domain inverse operates on each block.
    FSK uses two noncoherent correlators (no channel inverse).
    """
    length = block_len(cfg.cp)
    out = []
    for block in wave[:count*length].reshape(count, length):
        if cfg.cp:
            block = block[CP_SYMBOLS*SPS:]
        known = to_wave(PILOTS.real if fsk else PILOTS, fsk)
        pilot_rx = block[:len(known)]
        # Estimate residual frequency on known pilot waveforms, in sample domain.
        despread = pilot_rx * known.conj()
        grouped = despread.reshape(PILOT_SYMBOLS, SPS).mean(axis=1)
        phase = np.unwrap(np.angle(grouped))
        slope, intercept = np.polyfit(np.arange(PILOT_SYMBOLS)*SPS+(SPS-1)/2, phase, 1)
        # Fixed CFO was acquired on header; suppress noisy estimates for each short block.
        slope = float(np.clip(slope, -0.01, 0.01))
        block = block * np.exp(-1j*(slope*np.arange(len(block))+intercept))
        gain = np.vdot(known, block[:len(known)]) / np.vdot(known, known)
        if abs(gain) < 1e-9:
            gain = 1e-9
        if equalize and cfg.cp and not fsk:
            taps = 9
            # Rows contain x[n], x[n-1], ...; all are known pilots.
            x = np.array([known[n-np.arange(taps)] for n in range(taps-1, len(known))])
            y = block[taps-1:len(known)]
            hh = np.linalg.solve(x.conj().T@x + 0.05*np.eye(taps), x.conj().T@y)
            err = np.mean(abs(y-x@hh)**2)
            hf = np.fft.fft(hh, len(block))
            floor = max(float(err), float(abs(gain)**2)*1e-4)
            block = np.fft.ifft(np.fft.fft(block)*hf.conj()/(abs(hf)**2+floor))
        else:
            block = block / gain
        rows = block[len(known):].reshape(DATA_SYMBOLS, SPS)
        if fsk:
            tone = np.exp(2j*np.pi*np.arange(SPS)/SPS)
            r0, r1 = rows@tone / SPS, rows@tone.conj() / SPS
            out.append(np.column_stack((r0, r1)))
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


def decode_frame(wave, cfg, cfo, score=1.):
    wave = wave * np.exp(-2j*np.pi*cfo*np.arange(len(wave))/cfg.sample_rate)
    # Full marker refines coarse frequency estimate before decoding all blocks.
    d = (wave[:len(SYNC)]*SYNC.conj()).reshape(-1, SPS).mean(axis=1)
    phase = np.unwrap(np.angle(d))
    slope, intercept = np.polyfit(np.arange(len(d))*SPS+(SPS-1)/2, phase, 1)
    wave *= np.exp(-1j*(slope*np.arange(len(wave))+intercept))
    cfo += slope*cfg.sample_rate/(2*np.pi)
    cursor = len(SYNC)
    hsize = H_BLOCKS*block_len(cfg.cp)
    hz = decode_blocks(wave[cursor:cursor+hsize], H_BLOCKS, cfg)
    hb = (hz.real[:H_SYMBOLS] >= 0).reshape(-1, 3).sum(axis=1) >= 2
    meta = Meta.decode(whiten(pack(hb), 9121), cfg)
    cursor += hsize
    k = constellation(cfg.mod)[1]
    nsyms = math.ceil((PAYLOAD+4)*8/k)
    nblocks = math.ceil(nsyms/DATA_SYMBOLS)
    obs = decode_blocks(wave[cursor:], nblocks, cfg, cfg.mod == '2-FSK', equalize=True)[:nsyms]
    b = (abs(obs[:, 1]) > abs(obs[:, 0])).astype(np.uint8) if cfg.mod == '2-FSK' else decisions(obs, cfg.mod)
    data = whiten(pack(b[:(PAYLOAD+4)*8]), payload_seed(meta))
    payload = data[:meta.length]
    valid = zlib.crc32(payload) == int.from_bytes(data[PAYLOAD:PAYLOAD+4], 'big')
    return Decoded(meta, payload, valid, obs, cfo, score)


class StreamDecoder:
    """Bounded overlap buffer; accepts arbitrary chunks, never assumes packet alignment."""
    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.buffer = np.empty(0, np.complex64)
        self.prefix_len = len(PREAMBLE) if cfg.preamble else 0
        self.sync = PREAMBLE[:len(SYNC)] if cfg.preamble else SYNC
        self.length = frame_len(cfg) + self.prefix_len
        self.header_failures = 0
        self.candidates = 0
        self.lock_cfo = None
        self.scanned = 0

    def reset(self):
        self.buffer = np.empty(0, np.complex64)
        self.lock_cfo = None

    def feed(self, iq, stop=None):
        self.buffer = np.r_[self.buffer, np.asarray(iq, np.complex64)]
        result = []
        short = self.sync[:32*SPS]
        # A half-symbol differential product removes the unknown carrier phase.
        # Its correlation finds the packet start with one FFT and its phase gives
        # CFO over the whole supported ±100 kHz range, avoiding a costly CFO grid.
        lag = SPS//2
        differential_sync = short[lag:]*short[:-lag].conj()
        differential_energy = float(np.vdot(differential_sync,differential_sync).real)
        step = self.cfg.sample_rate / len(short) / 2
        while len(self.buffer) >= self.length:
            if stop is not None and stop.is_set():
                break
            # Search only starts for which a whole frame is already present.
            positions = min(len(self.buffer)-self.length+1, 32768)
            segment = self.buffer[:positions+len(short)-1]
            differential = segment[lag:]*segment[:-lag].conj()
            corr = correlate(differential,differential_sync,mode='valid',method='fft')
            energy = np.convolve(abs(differential)**2,np.ones(len(differential_sync)),'valid')
            denom = np.maximum(energy*differential_energy,1e-15)
            # FFT roundoff over exact silence must not win normalized correlation.
            energetic = energy > max(float(energy.max())*1e-6, 1e-15)
            best = np.where(energetic,abs(corr)**2/denom,0)
            freqs = np.angle(corr)*self.cfg.sample_rate/(2*np.pi*lag)
            margin = step
            best[np.abs(freqs)>self.cfg.cfo_range+margin] = 0
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
                t = np.arange(len(self.sync))/self.cfg.sample_rate
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
