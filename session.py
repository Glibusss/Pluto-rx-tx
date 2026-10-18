from __future__ import annotations
import csv
import hashlib
import json
import math
import secrets
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from quality import ratio_db, evm_percent
from modulations import SPS, fsk
from modem import (PAYLOAD, MAX_BYTES, Meta, Decoded, bits, symbols, whiten, constellation,
                   payload_seed, MODS, Config, tile_shape)


class Source:
    def __init__(self, kind, raw, width=0, height=0, label='', payload_bytes=PAYLOAD):
        if not 0 < len(raw) <= MAX_BYTES:
            raise ValueError('Размер исходных данных должен быть от 1 байта до 32 MiB')
        if kind and (not 0 < width <= 65535 or not 0 < height <= 65535 or len(raw) != width*height*3):
            raise ValueError('Размеры RGB: 1…65535 по каждой оси; буфер должен содержать width×height×3 байта')
        self.kind, self.raw = kind, raw
        self.width, self.height, self.label = width, height, label
        if not isinstance(payload_bytes, int) or not 48 <= payload_bytes <= 12288:
            raise ValueError('Данных в пакете: 48…12288 байт')
        self.payload_bytes = payload_bytes
        self.tile_width, self.tile_height = tile_shape(payload_bytes)
        self.total = (math.ceil(width/self.tile_width)*math.ceil(height/self.tile_height)
                      if kind else math.ceil(len(raw)/payload_bytes))
        if self.total > 65536:
            raise ValueError('Больше 65536 пакетов; уменьшите изображение/текст')
        self.digest = hashlib.sha256(raw).digest()[:16]
        self.rgb = np.frombuffer(raw,np.uint8).reshape(height,width,3) if kind else None

    @classmethod
    def image(cls, path, payload_bytes=PAYLOAD):
        with Image.open(path) as im:
            # Preserve simulator orientation semantics; discard alpha, no resizing.
            if im.width*im.height*3 > MAX_BYTES:
                raise ValueError('Декодированное RGB превышает 32 MiB')
            im = ImageOps.exif_transpose(im).convert('RGB')
            return cls(1, im.tobytes(), im.width, im.height, Path(path).name, payload_bytes)

    @classmethod
    def text(cls, text, label='Текст UTF-8', payload_bytes=PAYLOAD):
        return cls(0, text.encode('utf-8'), label=label, payload_bytes=payload_bytes)

    def with_payload(self, payload_bytes):
        if self.payload_bytes == payload_bytes:
            return self
        return Source(self.kind, self.raw, self.width, self.height, self.label, payload_bytes)

    def payload(self, seq):
        if not 0 <= seq < self.total:
            raise IndexError(seq)
        if not self.kind:
            return self.raw[seq*self.payload_bytes:(seq+1)*self.payload_bytes]
        tw, th = self.tile_width, self.tile_height
        cols = math.ceil(self.width/tw)
        x, y = (seq % cols)*tw, (seq//cols)*th
        block = np.zeros((th,tw,3),np.uint8)
        part = self.rgb[y:y+th,x:x+tw]
        block[:part.shape[0],:part.shape[1]] = part
        return block.tobytes()

    def meta(self, transfer, seq, end=False):
        length = 0 if end else len(self.payload(seq))
        return Meta(self.kind, transfer, seq, self.total, len(self.raw), self.width,
                    self.height, length, self.digest, end, self.payload_bytes)


class Reception:
    def __init__(self, meta, reference, mod, settings=None):
        self.meta, self.mod = meta, mod
        self.settings = dict(settings or {})
        cfg_values = self.settings.get('cfg', {})
        self.cfg = Config(**cfg_values) if cfg_values else Config(mod=mod, payload_bytes=meta.payload_bytes)
        self.reference = reference
        if reference is None:
            self.reference_status='эталон не выбран'
        elif reference.kind != meta.kind:
            self.reference_status='тип эталона не совпадает (текст/RGB)'
        elif len(reference.raw) != meta.size:
            self.reference_status=f'размер эталона не совпадает ({len(reference.raw)} вместо {meta.size} байт)'
        elif reference.width != meta.width or reference.height != meta.height:
            self.reference_status=(f'размеры изображения не совпадают '
                                   f'({reference.width}×{reference.height} вместо {meta.width}×{meta.height})')
        elif reference.digest != meta.digest:
            self.reference_status='содержимое эталона не совпадает (другой SHA-256)'
        else:
            self.reference_status='эталон совпадает'
        self.reference_ok=self.reference_status=='эталон совпадает'
        if self.reference_ok:
            self.reference = reference.with_payload(meta.payload_bytes)
        self.data = bytearray(b'\x80')*meta.size if meta.kind else bytearray(meta.size)
        self.rows = {}
        self.good_packets = 0
        self.compared_bits = 0
        self.bit_errors = 0
        self.received_pixels = 0
        self.corrupted_pixels = 0
        self.ended = False
        self.duplicates = 0
        self.signal_sum = self.error_sum = 0.
        self.symbol_count = 0
        self.sir_signal_sum = self.sir_interference_sum = 0.
        self.sir_vector_count = 0
        self.pilot_signal_sum = self.pilot_error_sum = 0.
        self.pilot_sample_count = 0
        self.crc_signal_sum = self.crc_error_sum = 0.
        self.crc_symbol_count = 0
        self.decision_samples = np.empty((0,2))
        self.cfo = 0.
        self.transport_drops = 0
        self.transport_read_errors = 0
        self.transport_buffer_restarts = 0
        self.preview_size = 0
        self.first_sequence = meta.seq
        self.first_header_was_end = meta.end

    def accept(self, packet):
        m = packet.meta
        base = self.meta
        if (m.transfer,m.kind,m.total,m.size,m.width,m.height,m.digest,m.payload_bytes) != (
                base.transfer,base.kind,base.total,base.size,base.width,base.height,base.digest,base.payload_bytes):
            raise ValueError('Конфликт метаданных одной передачи')
        if m.end:
            self.ended = True
            return
        if m.seq in self.rows:
            self.duplicates += 1
            return  # First observation wins: no concealed BER improvement by retransmissions.
        self.pilot_signal_sum += packet.pilot_signal_energy
        self.pilot_error_sum += packet.pilot_disturbance_energy
        self.pilot_sample_count += packet.pilot_sample_count
        if packet.crc_ok:
            self.crc_signal_sum += packet.crc_signal_energy
            self.crc_error_sum += packet.crc_error_energy
            self.crc_symbol_count += packet.crc_symbol_count
        self.sir_signal_sum += packet.sir_signal_energy
        self.sir_interference_sum += packet.sir_interference_energy
        self.sir_vector_count += packet.sir_vector_count
        errors = None
        nbits = m.length*8
        if self.reference_ok:
            ref = self.reference.payload(m.seq)
            errors = int(np.count_nonzero(bits(ref) != bits(packet.payload)))
            # Compare pre-decision received observations against independently loaded reference.
            import struct, zlib
            encoded = whiten(ref.ljust(m.payload_bytes,b'\0')+struct.pack('!I',zlib.crc32(ref)),payload_seed(m))
            expected = symbols(encoded,self.mod,self.cfg)
            k = constellation(self.mod,self.cfg)[1]
            n = nbits//k  # only full payload symbols; excludes padding and CRC
            if self.mod == '2-FSK':
                target = fsk.reference(expected[:n], SPS, self.cfg.fsk_cycles)
                observed = packet.observations[:n]
            else:
                target, observed = expected[:n], packet.observations[:n]
            self.signal_sum += float(np.sum(abs(target)**2))
            self.error_sum += float(np.sum(abs(observed-target)**2))
            self.symbol_count += n
        self.rows[m.seq] = dict(seq=m.seq,status='GOOD' if packet.crc_ok else 'CRC',
                               bit_errors=errors,payload_bits=nbits,cfo_hz=packet.cfo,
                               pilot_sinr_db=ratio_db(packet.pilot_signal_energy,
                                   packet.pilot_disturbance_energy,packet.pilot_sample_count),
                               crc_payload_evm_percent=evm_percent(packet.crc_signal_energy,
                                   packet.crc_error_energy,packet.crc_symbol_count))
        self.good_packets += int(packet.crc_ok)
        if errors is not None:
            self.compared_bits += nbits
            self.bit_errors += errors
        self.cfo = packet.cfo
        if self.mod == '2-FSK':
            self.decision_samples = abs(packet.observations[:600])
        else:
            self.decision_samples = np.column_stack((packet.observations[:600].real,packet.observations[:600].imag))
        if m.kind:
            tw, th = tile_shape(m.payload_bytes)
            cols = math.ceil(m.width/tw)
            x,y = (m.seq%cols)*tw,(m.seq//cols)*th
            out = np.frombuffer(self.data,np.uint8).reshape(m.height,m.width,3)
            block = np.frombuffer(packet.payload,np.uint8).reshape(th,tw,3)
            hh,ww = min(th,m.height-y),min(tw,m.width-x)
            self.received_pixels += hh*ww
            if self.reference_ok:
                target = self.reference.rgb[y:y+hh,x:x+ww]
                self.corrupted_pixels += int(np.count_nonzero(np.any(block[:hh,:ww] != target, axis=2)))
            out[y:y+hh,x:x+ww] = block[:hh,:ww]
        else:
            start=m.seq*m.payload_bytes
            self.data[start:start+m.length] = packet.payload
            self.preview_size = max(self.preview_size,start+m.length)

    def stats(self, final=False):
        good = self.good_packets
        received = len(self.rows)
        compared = self.compared_bits
        errors = self.bit_errors if self.reference_ok else None
        expected_bits = self.meta.total*(self.meta.payload_bytes//3*3)*8 if self.meta.kind else self.meta.size*8
        snr = 10*math.log10(self.signal_sum/max(self.error_sum,1e-30)) if self.symbol_count else None
        sir = (10*math.log10(self.sir_signal_sum/max(self.sir_interference_sum,1e-30))
               if self.sir_vector_count and self.sir_signal_sum > 0 else None)
        return dict(transfer=f'{self.meta.transfer:016x}',modulation=self.mod,
            kind='RGB' if self.meta.kind else 'text',width=self.meta.width,height=self.meta.height,
            expected_packets=self.meta.total,received_packets=received,good_packets=good,
            crc_packets=received-good,missing_packets=self.meta.total-received,
            lost_packets=self.meta.total-received,
            lost_pixels=self.meta.width*self.meta.height-self.received_pixels if self.meta.kind else None,
            received_pixels=self.received_pixels if self.meta.kind else None,
            corrupted_pixels=self.corrupted_pixels if self.meta.kind and self.reference_ok else None,
            corrupted_bits=errors, payload_bytes=self.meta.payload_bytes,
            per=(self.meta.total-good)/self.meta.total,per_final=bool(final or self.ended),
            ber=errors/compared if compared else None,bit_errors=errors,compared_bits=compared,
            expected_bits=expected_bits,ber_coverage=compared/expected_bits,
            reference_match=self.reference_ok,reference_status=self.reference_status,snr_estimate_db=snr,
            snr_method='reference error at symbol decisions; includes residual channel distortion',
            sinr_estimate_db=ratio_db(self.pilot_signal_sum,self.pilot_error_sum,self.pilot_sample_count),
            sinr_method='known-pilot complex LS residual; before equalization; includes noise, interference and residual distortion',
            pilot_evm_percent=evm_percent(self.pilot_signal_sum,self.pilot_error_sum,self.pilot_sample_count),
            pilot_sample_count=self.pilot_sample_count,
            crc_payload_sinr_db=ratio_db(self.crc_signal_sum,self.crc_error_sum,self.crc_symbol_count),
            crc_payload_evm_percent=evm_percent(self.crc_signal_sum,self.crc_error_sum,self.crc_symbol_count),
            crc_payload_symbol_count=self.crc_symbol_count,
            crc_quality_packet_coverage=good/received if received else 0.,
            quality_warning=('CRC-ошибки: EVM данных учитывает только верные пакеты; '
                             'помехи между пилотами могут не попасть в SINR' if received>good else ''),
            quality_requires_reference=False,
            quality_basis='two-correlator vector' if self.mod=='2-FSK' else 'complex symbol',
            sir_estimate_db=sir,
            sir_method='decision-directed error vectors at payload symbol decisions; dB',
            end_seen=self.ended,duplicates=self.duplicates,last_cfo_hz=self.cfo,
            transport_dropped_buffers=self.transport_drops,
            transport_read_errors=self.transport_read_errors,
            transport_buffer_restarts=self.transport_buffer_restarts,
            first_received_sequence=self.first_sequence,
            first_header_was_end=self.first_header_was_end,
            receiver_settings=self.settings)

    def snapshot(self, final=False):
        return dict(stats=self.stats(final),data=bytes(self.data),
                    statuses={k:v['status'] for k,v in self.rows.items()},
                    samples=self.decision_samples.copy(),preview_size=self.preview_size)

    def save(self, folder, final=False):
        folder=Path(folder)/f'transfer_{self.meta.transfer:016x}'
        folder.mkdir(parents=True,exist_ok=True)
        if self.meta.kind:
            Image.frombytes('RGB',(self.meta.width,self.meta.height),bytes(self.data)).save(folder/'received.png')
        else:
            (folder/'received.bin').write_bytes(self.data)
            text = bytes(self.data).decode('utf-8',errors='replace')
            (folder/'received.txt').write_text(text,encoding='utf-8')
        (folder/'metrics.json').write_text(json.dumps(self.stats(final),ensure_ascii=False,indent=2),encoding='utf-8')
        with (folder/'packets.csv').open('w',newline='',encoding='utf-8-sig') as f:
            fields=['seq','status','bit_errors','payload_bits','cfo_hz','pilot_sinr_db','crc_payload_evm_percent']
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for seq in range(self.meta.total):
                writer.writerow(self.rows.get(seq,dict(seq=seq,status='LOST' if final or self.ended else 'PENDING')))
        return str(folder)
