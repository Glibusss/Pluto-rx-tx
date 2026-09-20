from __future__ import annotations
from collections import deque
import math
import queue
import secrets
import threading
import time
from pathlib import Path
import numpy as np
from scipy.stats import f as fisher_f, norm
from modem import Config, make_frame, StreamDecoder
from session import Reception


RX_QUEUE_MIN = 16
RX_QUEUE_DEFAULT = 512
RX_QUEUE_MAX = 8192
ADC_FULL_SCALE = 2048.0
NOISE_FALSE_ALARM = 1e-6
NOISE_OUTLIER_SIGMAS = 5.0


def discover_pluto_usb(scan=None):
    """Return USB IIO contexts, with contexts identified as Pluto first."""
    if scan is None:
        try:
            import iio
        except (ImportError,OSError) as error:
            raise RuntimeError('Не удалось загрузить libiio для поиска Pluto: '+str(error)) from error
        scan=iio.scan_contexts
    try:
        contexts=scan()
    except Exception as error:
        raise RuntimeError('Ошибка сканирования USB-контекстов libiio: '+str(error)) from error
    if not hasattr(contexts,'items'):
        raise RuntimeError('libiio вернул некорректный результат сканирования')
    result=[]
    for uri,description in contexts.items():
        uri=str(uri)
        if not uri.lower().startswith('usb:'):
            continue
        description='' if description is None else str(description)
        text=description.lower()
        identified='pluto' in text or 'adalm' in text
        result.append(dict(uri=uri,description=description,identified=identified))
    result.sort(key=lambda item:(not item['identified'],item['uri']))
    return result


def rx_queue_capacity(settings):
    try:
        value=int(settings.get('queue_buffers',RX_QUEUE_DEFAULT))
    except (TypeError,ValueError) as error:
        raise ValueError(f'Размер IQ-очереди: {RX_QUEUE_MIN}…{RX_QUEUE_MAX} буферов') from error
    if not RX_QUEUE_MIN<=value<=RX_QUEUE_MAX:
        raise ValueError(f'Размер IQ-очереди: {RX_QUEUE_MIN}…{RX_QUEUE_MAX} буферов')
    return value


def iq_variance(iq):
    """Unbiased complex variance after removing the buffer's DC component."""
    samples=np.asarray(iq).ravel()
    if samples.size<2:
        raise ValueError('Пустой IQ-буфер')
    mean=np.mean(samples,dtype=np.complex128)
    second=float(np.mean(abs(samples)**2,dtype=np.float64))
    variance=max(0.,second-float(abs(mean)**2))*samples.size/(samples.size-1)
    return variance


def dbfs_from_power(power):
    return 10*math.log10(max(float(power)/(ADC_FULL_SCALE**2),1e-20))


def power_dbfs(iq):
    return dbfs_from_power(iq_variance(iq))


def noise_threshold(buffer_powers, samples_per_buffer, false_alarm=NOISE_FALSE_ALARM):
    """Robust calibrated energy-detector threshold.

    For circular complex Gaussian noise, a ratio of the current unbiased power
    estimate to the calibrated estimate follows Fisher's F distribution.  MAD
    clipping prevents occasional transmissions/impulses from poisoning the
    reference estimate; it is not used as a substitute for the CFAR model.
    """
    powers=np.asarray(buffer_powers,float)
    samples=int(samples_per_buffer)
    if (powers.size<4 or samples<2 or not np.all(np.isfinite(powers)) or
            np.any(powers<0) or not 0<float(false_alarm)<.5):
        raise ValueError('Нет корректных измерений шума')
    floor=ADC_FULL_SCALE**2*1e-20
    powers=np.maximum(powers,floor)
    levels=10*np.log10(powers/(ADC_FULL_SCALE**2))

    center=float(np.median(levels))
    sigma_db=1.4826*float(np.median(abs(levels-center)))
    clip_db=max(.25,NOISE_OUTLIER_SIGMAS*sigma_db)
    inliers=abs(levels-center)<=clip_db
    if int(np.count_nonzero(inliers))<max(3,math.ceil(len(levels)/2)):
        inliers=np.ones(len(levels),bool)

    selected=powers[inliers]
    selected_levels=levels[inliers]
    noise_power=float(np.mean(selected))
    robust_center=float(np.median(selected_levels))
    spread_db=1.4826*float(np.median(abs(selected_levels-robust_center)))
    used=len(selected)

    # Each complex buffer variance has 2(N-1) real degrees of freedom.  The
    # reference is the mean of `used` independent buffers, hence the F ratio.
    test_dof=2*(samples-1)
    reference_dof=test_dof*used
    cfar_factor=float(fisher_f.ppf(1-false_alarm,test_dof,reference_dof))

    # Real receivers drift more than the ideal Gaussian model.  Model the
    # observed buffer-to-buffer log-power variation and use the stricter limit.
    z=float(norm.isf(false_alarm))
    empirical_factor=10**(z*spread_db/10)
    threshold_factor=max(cfar_factor,empirical_factor)
    threshold_power=noise_power*threshold_factor
    theoretical_pfa=float(fisher_f.sf(threshold_factor,test_dof,reference_dof))
    return dict(noise_dbfs=dbfs_from_power(noise_power),
                threshold_dbfs=dbfs_from_power(threshold_power),
                threshold_margin_db=10*math.log10(threshold_factor),
                spread_db=spread_db,buffers=len(powers),used_buffers=used,
                outliers=len(powers)-used,samples_per_buffer=samples,
                false_alarm_probability=float(false_alarm),
                theoretical_false_alarm=theoretical_pfa)


def calibrate_noise(settings, stop, duration=2.0):
    sdr=None
    try:
        sdr=connect(settings,False)
        count=max(8,math.ceil(duration*int(sdr.sample_rate)/int(sdr.rx_buffer_size)))
        powers=[]
        samples_per_buffer=None
        for _ in range(count):
            if stop.is_set():
                return None
            raw=np.asarray(sdr.rx()).ravel()
            if samples_per_buffer is None:
                samples_per_buffer=len(raw)
            elif len(raw)!=samples_per_buffer:
                raise RuntimeError('Pluto вернул IQ-буферы разного размера при калибровке')
            powers.append(iq_variance(raw))
        return noise_threshold(powers,samples_per_buffer)
    finally:
        if sdr is not None:
            sdr.rx_destroy_buffer()


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
        size=None
        cycle=0
        while not stop.is_set():
            cycle+=1
            tid=secrets.randbits(64)
            emit('log',f'Цикл {cycle}, передача {tid:016x}: {source.total} пакетов, {len(source.raw)} байт')
            emit('progress',(0,source.total,cycle))
            sent=end_sent=0
            # END is control, repeated 3 times. Each cycle has a fresh transfer ID.
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
                if end:
                    end_sent+=1
                else:
                    sent+=1
                    emit('progress',(sent,source.total,cycle))
            complete=sent==source.total and end_sent==3
            tail=('Отправлен END; '+('остановлено пользователем.' if stop.is_set()
                                     else 'начинаю следующий цикл.')) if complete else 'Остановлено пользователем.'
            emit('log',f'Цикл {cycle}: в SDR отправлено {sent}/{source.total} пакетов. '+tail)
            if not complete:
                break
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
    frames=queue.Queue(maxsize=rx_queue_capacity(settings))
    errors=queue.Queue()
    capture_stop=threading.Event()
    dropped=[0]
    squelched=[0]
    input_dbfs=[None]
    session=None
    decoder=StreamDecoder(settings['cfg'])
    threshold_dbfs=settings.get('squelch_dbfs')
    threshold_power=(None if threshold_dbfs is None else
                     ADC_FULL_SCALE**2*10**(float(threshold_dbfs)/10))
    try:
        sdr=connect(settings,False)
        emit('log',f'Pluto подключён. RX LO={int(sdr.rx_lo)} Гц. Приём до Stop.')
        if threshold_dbfs is not None:
            emit('log',f'Шумовой порог включён: {float(threshold_dbfs):.1f} dBFS.')
        def collect():
            offset=0
            gap=False
            tail=0
            pretrigger=deque(maxlen=2)
            def enqueue(item):
                nonlocal gap
                iq,clipping=item
                try:
                    frames.put_nowait((iq,gap,clipping))
                    gap=False
                except queue.Full:
                    dropped[0]+=1
                    gap=True
            try:
                while not capture_stop.is_set():
                    raw=np.asarray(sdr.rx(),np.complex64)
                    # fs/4 oscillator has exactly four states, no growing float phase.
                    mixer=np.array([1,-1j,-1,1j],np.complex64)[(np.arange(len(raw))+offset)%4]
                    iq=raw*mixer
                    offset=(offset+len(raw))%4
                    clipping=float(np.mean((abs(raw.real)>=32760)|(abs(raw.imag)>=32760)))
                    raw_power=iq_variance(raw)
                    input_dbfs[0]=dbfs_from_power(raw_power)
                    if threshold_power is None:
                        enqueue((iq,clipping))
                    elif raw_power>=threshold_power:
                        while pretrigger:
                            enqueue(pretrigger.popleft())
                        enqueue((iq,clipping))
                        tail=2
                    elif tail:
                        enqueue((iq,clipping))
                        tail-=1
                    else:
                        if len(pretrigger)==pretrigger.maxlen:
                            pretrigger.popleft()
                            squelched[0]+=1
                            gap=True
                        pretrigger.append((iq,clipping))
            except Exception as e:
                if not capture_stop.is_set():
                    errors.put(e)
        capture=threading.Thread(target=collect,name='Pluto RX capture',daemon=True)
        capture.start()
        last=0
        last_clip=0
        last_gap_log=0
        while not stop.is_set():
            if not errors.empty():
                raise RuntimeError(f'Ошибка чтения SDR: {errors.get()}')
            try:
                iq,gap,clipping=frames.get(timeout=.15)
            except queue.Empty:
                now=time.monotonic()
                if now-last>.5:
                    emit('health',dict(candidates=decoder.candidates,header_failures=decoder.header_failures,
                        queue=frames.qsize(),queue_capacity=frames.maxsize,drops=dropped[0],
                        squelched=squelched[0],input_dbfs=input_dbfs[0],squelch_dbfs=threshold_dbfs))
                    last=now
                continue
            if gap:
                decoder.reset()
                now=time.monotonic()
                if now-last_gap_log>=1:
                    emit('log',f'Перегрузка обработки: потеряно IQ-буферов {dropped[0]}. Поиск пакета заново.')
                    last_gap_log=now
            if clipping>.001 and time.monotonic()-last_clip>3:
                last_clip=time.monotonic()
                emit('log','Обнаружено ограничение I/Q. Уменьшите RX gain / уровень TX.')
            packets=decoder.feed(iq,stop)
            for packet in packets:
                # Continuous TX repeats transfers. Start only at packet zero so an
                # RX launched in the middle of a cycle waits for one complete result.
                if ((session is None or session.meta.transfer != packet.meta.transfer) and
                        (packet.meta.end or packet.meta.seq != 0)):
                    continue
                if session is None or session.meta.transfer != packet.meta.transfer:
                    if session is not None:
                        emit('log','Предыдущий результат: '+session.save(folder,True))
                    session=Reception(packet.meta,reference,settings['cfg'].mod)
                    suffix='' if session.reference_ok else ' — BER и SNR недоступны'
                    emit('log',f'Передача {packet.meta.transfer:016x}; пакетов: {packet.meta.total}; '+
                         session.reference_status+suffix)
                was_end=session.ended
                session.accept(packet)
                session.transport_drops=dropped[0]
                emit('snapshot',session.snapshot())
                if session.ended and not was_end:
                    emit('log','Получен END. Передача принята; RX останавливается.')
                    stop.set()
                    break
            now=time.monotonic()
            if now-last>.5:
                if session:
                    session.transport_drops=dropped[0]
                    emit('snapshot',session.snapshot())
                emit('health',dict(candidates=decoder.candidates,header_failures=decoder.header_failures,
                    queue=frames.qsize(),queue_capacity=frames.maxsize,drops=dropped[0],
                    squelched=squelched[0],input_dbfs=input_dbfs[0],squelch_dbfs=threshold_dbfs))
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
