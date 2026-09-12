from __future__ import annotations
import csv
import hashlib
import json
import math
import secrets
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from modem import (PAYLOAD, MAX_BYTES, Meta, Decoded, bits, symbols, whiten, constellation,
                   payload_seed, MODS)


class Source:
    def __init__(self, kind, raw, width=0, height=0, label=''):
        if not 0 < len(raw) <= MAX_BYTES:
            raise ValueError('Размер исходных данных должен быть от 1 байта до 32 MiB')
        if kind and (not 0 < width <= 65535 or not 0 < height <= 65535 or len(raw) != width*height*3):
            raise ValueError('Размеры RGB: 1…65535 по каждой оси; буфер должен содержать width×height×3 байта')
        self.kind, self.raw = kind, raw
        self.width, self.height, self.label = width, height, label
        self.total = math.ceil(width/16)*math.ceil(height/16) if kind else math.ceil(len(raw)/PAYLOAD)
        if self.total > 65536:
            raise ValueError('Больше 65536 пакетов; уменьшите изображение/текст')
        self.digest = hashlib.sha256(raw).digest()[:16]
        self.rgb = np.frombuffer(raw,np.uint8).reshape(height,width,3) if kind else None

    @classmethod
    def image(cls, path):
        with Image.open(path) as im:
            # Preserve simulator orientation semantics; discard alpha, no resizing.
            if im.width*im.height*3 > MAX_BYTES:
                raise ValueError('Декодированное RGB превышает 32 MiB')
            im = ImageOps.exif_transpose(im).convert('RGB')
            return cls(1, im.tobytes(), im.width, im.height, Path(path).name)

    @classmethod
    def text(cls, text, label='Текст UTF-8'):
        return cls(0, text.encode('utf-8'), label=label)

    def payload(self, seq):
        if not 0 <= seq < self.total:
            raise IndexError(seq)
        if not self.kind:
            return self.raw[seq*PAYLOAD:(seq+1)*PAYLOAD]
        cols = math.ceil(self.width/16)
        x, y = (seq % cols)*16, (seq//cols)*16
        block = np.zeros((16,16,3),np.uint8)
        part = self.rgb[y:y+16,x:x+16]
        block[:part.shape[0],:part.shape[1]] = part
        return block.tobytes()

    def meta(self, transfer, seq, end=False):
        length = 0 if end else len(self.payload(seq))
        return Meta(self.kind, transfer, seq, self.total, len(self.raw), self.width,
                    self.height, length, self.digest, end)


class Reception:
    def __init__(self, meta, reference, mod):
        self.meta, self.mod = meta, mod
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
        self.data = bytearray([128]*meta.size) if meta.kind else bytearray(meta.size)
        self.rows = {}
        self.ended = False
        self.duplicates = 0
        self.signal_sum = self.error_sum = 0.
        self.symbol_count = 0
        self.decision_samples = np.empty((0,2))
        self.cfo = 0.
        self.transport_drops = 0
        self.preview_size = 0

    def accept(self, packet):
        m = packet.meta
        base = self.meta
        if (m.transfer,m.kind,m.total,m.size,m.width,m.height,m.digest) != (
                base.transfer,base.kind,base.total,base.size,base.width,base.height,base.digest):
            raise ValueError('Конфликт метаданных одной передачи')
        if m.end:
            self.ended = True
            return
        if m.seq in self.rows:
            self.duplicates += 1
            return  # First observation wins: no concealed BER improvement by retransmissions.
        errors = None
        nbits = m.length*8
        if self.reference_ok:
            ref = self.reference.payload(m.seq)
            errors = int(np.count_nonzero(bits(ref) != bits(packet.payload)))
            # Compare pre-decision received observations against independently loaded reference.
            import struct, zlib
            encoded = whiten(ref.ljust(PAYLOAD,b'\0')+struct.pack('!I',zlib.crc32(ref)),payload_seed(m))
            expected = symbols(encoded,self.mod)
            k = constellation(self.mod)[1]
            n = nbits//k  # only full payload symbols; excludes padding and CRC
            if self.mod == '2-FSK':
                target = np.column_stack((expected[:n]<0, expected[:n]>0)).astype(complex)
                observed = packet.observations[:n]
            else:
                target, observed = expected[:n], packet.observations[:n]
            self.signal_sum += float(np.sum(abs(target)**2))
            self.error_sum += float(np.sum(abs(observed-target)**2))
            self.symbol_count += n
        self.rows[m.seq] = dict(seq=m.seq,status='GOOD' if packet.crc_ok else 'CRC',
                               bit_errors=errors,payload_bits=nbits,cfo_hz=packet.cfo)
        self.cfo = packet.cfo
        if self.mod == '2-FSK':
            self.decision_samples = abs(packet.observations[:600])
        else:
            self.decision_samples = np.column_stack((packet.observations[:600].real,packet.observations[:600].imag))
        if m.kind:
            cols = math.ceil(m.width/16)
            x,y = (m.seq%cols)*16,(m.seq//cols)*16
            out = np.frombuffer(self.data,np.uint8).reshape(m.height,m.width,3)
            block = np.frombuffer(packet.payload,np.uint8).reshape(16,16,3)
            hh,ww = min(16,m.height-y),min(16,m.width-x)
            out[y:y+hh,x:x+ww] = block[:hh,:ww]
        else:
            start=m.seq*PAYLOAD
            self.data[start:start+m.length] = packet.payload
            self.preview_size = max(self.preview_size,start+m.length)

    def stats(self, final=False):
        good = sum(r['status']=='GOOD' for r in self.rows.values())
        received = len(self.rows)
        compared = sum(r['payload_bits'] for r in self.rows.values()) if self.reference_ok else 0
        errors = sum(r['bit_errors'] for r in self.rows.values()) if self.reference_ok else None
        expected_bits = self.meta.total*PAYLOAD*8 if self.meta.kind else self.meta.size*8
        snr = 10*math.log10(self.signal_sum/max(self.error_sum,1e-30)) if self.symbol_count else None
        return dict(transfer=f'{self.meta.transfer:016x}',modulation=self.mod,
            kind='RGB' if self.meta.kind else 'text',width=self.meta.width,height=self.meta.height,
            expected_packets=self.meta.total,received_packets=received,good_packets=good,
            crc_packets=received-good,missing_packets=self.meta.total-received,
            per=(self.meta.total-good)/self.meta.total,per_final=bool(final or self.ended),
            ber=errors/compared if compared else None,bit_errors=errors,compared_bits=compared,
            expected_bits=expected_bits,ber_coverage=compared/expected_bits,
            reference_match=self.reference_ok,reference_status=self.reference_status,snr_estimate_db=snr,
            snr_method='reference error at symbol decisions; includes residual channel distortion',
            end_seen=self.ended,duplicates=self.duplicates,last_cfo_hz=self.cfo,
            transport_dropped_buffers=self.transport_drops)

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
            fields=['seq','status','bit_errors','payload_bits','cfo_hz']
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for seq in range(self.meta.total):
                writer.writerow(self.rows.get(seq,dict(seq=seq,status='LOST' if final or self.ended else 'PENDING')))
        return str(folder)
