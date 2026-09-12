from __future__ import annotations
import queue
import secrets
import threading
import time
from pathlib import Path
import numpy as np
from modem import Config, make_frame, StreamDecoder
from session import Reception


def connect(settings, tx=False):
    try:
        import adi
    except (ImportError, OSError) as e:
        raise RuntimeError('Не удалось загрузить pyadi-iio/libiio. См. README_RU.md. '+str(e)) from e
    cfg = settings['cfg']
    cfg.validate()
    frequency = settings['frequency']
    if not 326_000_000 <= frequency <= 3_800_000_000:
        raise ValueError('Частота центра: 326…3800 МГц для штатного Pluto')
    sdr = adi.Pluto(uri=settings['uri'])
    sdr._ctx.set_timeout(2000)
    sdr.sample_rate = cfg.sample_rate
    if int(sdr.sample_rate) != cfg.sample_rate:
        raise RuntimeError(f'Pluto установил другую sample rate: {sdr.sample_rate}')
    # Digital IF avoids AD936x DC cancellation eating the OOK carrier.
    lo = frequency-cfg.sample_rate//4
    if tx:
        sdr.tx_enabled_channels=[0]
        sdr.tx_lo=lo
        sdr.tx_rf_bandwidth=cfg.sample_rate
        sdr.tx_hardwaregain_chan0=settings['gain']
        sdr.tx_cyclic_buffer=False
    else:
        sdr.rx_enabled_channels=[0]
        sdr.rx_lo=lo
        sdr.rx_rf_bandwidth=cfg.sample_rate
        sdr.gain_control_mode_chan0='manual'
        sdr.rx_hardwaregain_chan0=settings['gain']
        sdr.rx_buffer_size=32768
        sdr._rxadc.set_kernel_buffers_count(4)
    return sdr


def transmit(settings, source, stop, emit):
    sdr=None
    try:
        cfg=settings['cfg']
        sdr=connect(settings,True)
        emit('log',f'Pluto подключён. TX LO={int(sdr.tx_lo)} Гц, Fs={int(sdr.sample_rate)}')
        tid=secrets.randbits(64)
        emit('log',f'Передача {tid:016x}: {source.total} пакетов, {len(source.raw)} байт')
        sent=0
        size=None
        # END is control, repeated 3 times. Data packets are sent exactly once.
        for index in range(source.total+3):
            if stop.is_set():
                break
            end=index>=source.total
            seq=source.total if end else index
            frame=make_frame(source.meta(tid,seq,end),b'' if end else source.payload(seq),cfg)
            gap=np.zeros(int(cfg.sample_rate*0.012),np.complex64)
            frame=np.r_[gap,frame,gap]
            frame=np.pad(frame,(0,(-len(frame))%4))
            iq=(frame*4096*np.exp(2j*np.pi*.25*np.arange(len(frame)))).astype(np.complex64)
            size=len(iq)
            sdr.tx(iq)
            # push completion need not mean last DAC sample; conservatively wait a full buffer.
            stop.wait(size/cfg.sample_rate + settings['gap_ms']/1000)
            if not end:
                sent+=1
                emit('progress',(sent,source.total))
        emit('log',f'В SDR отправлено {sent}/{source.total} пакетов. '+('Отмена.' if stop.is_set() else 'Отправлен END.'))
    finally:
        if sdr is not None:
            try:
                if size:
                    sdr.tx(np.zeros(size,np.complex64))
                    time.sleep(size/settings['cfg'].sample_rate + .02)
            finally:
                sdr.tx_destroy_buffer()
                # Explicitly suppress RF level after the final flush.
                sdr.tx_hardwaregain_chan0=-89.75


def receive(settings, reference, folder, stop, emit):
    sdr=None
    capture=None
    frames=queue.Queue(maxsize=96)
    errors=queue.Queue()
    capture_stop=threading.Event()
    dropped=[0]
    session=None
    decoder=StreamDecoder(settings['cfg'])
    try:
        sdr=connect(settings,False)
        emit('log',f'Pluto подключён. RX LO={int(sdr.rx_lo)} Гц. Приём до Stop.')
        def collect():
            offset=0
            gap=False
            try:
                while not capture_stop.is_set():
                    raw=np.asarray(sdr.rx(),np.complex64)
                    # fs/4 oscillator has exactly four states, no growing float phase.
                    mixer=np.array([1,-1j,-1,1j],np.complex64)[(np.arange(len(raw))+offset)%4]
                    iq=raw*mixer
                    offset=(offset+len(raw))%4
                    clipping=float(np.mean((abs(raw.real)>=32760)|(abs(raw.imag)>=32760)))
                    try:
                        frames.put_nowait((iq,gap,clipping))
                        gap=False
                    except queue.Full:
                        dropped[0]+=1
                        gap=True
            except Exception as e:
                if not capture_stop.is_set():
                    errors.put(e)
        capture=threading.Thread(target=collect,name='Pluto RX capture',daemon=True)
        capture.start()
        last=0
        last_clip=0
        while not stop.is_set():
            if not errors.empty():
                raise RuntimeError(f'Ошибка чтения SDR: {errors.get()}')
            try:
                iq,gap,clipping=frames.get(timeout=.15)
            except queue.Empty:
                continue
            if gap:
                decoder.reset()
                emit('log',f'Перегрузка обработки: потеряно IQ-буферов {dropped[0]}. Поиск пакета заново.')
            if clipping>.001 and time.monotonic()-last_clip>3:
                last_clip=time.monotonic()
                emit('log','Обнаружено ограничение I/Q. Уменьшите RX gain / уровень TX.')
            packets=decoder.feed(iq,stop)
            for packet in packets:
                if session is None or session.meta.transfer != packet.meta.transfer:
                    if session is not None:
                        emit('log','Предыдущий результат: '+session.save(folder,True))
                    session=Reception(packet.meta,reference,settings['cfg'].mod)
                    emit('log',f'Передача {packet.meta.transfer:016x}; пакетов: {packet.meta.total}; эталон: '+
                         ('совпадает' if session.reference_ok else 'не выбран / не совпадает — BER и SNR недоступны'))
                was_end=session.ended
                session.accept(packet)
                session.transport_drops=dropped[0]
                emit('snapshot',session.snapshot())
                if session.ended and not was_end:
                    emit('log','Получен END. Приём остаётся включённым.')
                    emit('log','Результат: '+session.save(folder,True))
            now=time.monotonic()
            if now-last>.5:
                if session:
                    session.transport_drops=dropped[0]
                    emit('snapshot',session.snapshot())
                emit('health',dict(candidates=decoder.candidates,header_failures=decoder.header_failures,
                                   queue=frames.qsize(),drops=dropped[0]))
                last=now
    finally:
        capture_stop.set()
        if capture:
            capture.join(timeout=3)
        if sdr is not None and (capture is None or not capture.is_alive()):
            sdr.rx_destroy_buffer()
        if session:
            session.transport_drops=dropped[0]
            emit('snapshot',session.snapshot(True))
            emit('log','Результат: '+session.save(folder,True))
        if capture is not None and capture.is_alive():
            raise RuntimeError('Драйвер RX не завершился за 3 с. Закройте приложение перед повторным запуском.')
