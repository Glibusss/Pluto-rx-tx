from __future__ import annotations
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import ctypes
import errno
import json
import math
import os
import queue
import secrets
import threading
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.stats import f as fisher_f, norm
from modem import Config, SPS, make_frame, StreamDecoder
from session import Reception


RX_QUEUE_MIN = 16
RX_QUEUE_DEFAULT = 512
RX_QUEUE_MAX = 8192
ADC_FULL_SCALE = 2048.0
# ADI's TX HDL consumes signed 16-bit samples before dropping the low 4 bits.
# One common scale leaves headroom for OOK and QAM peaks as well as PSK/FSK.
TX_DIGITAL_FULL_SCALE = 32768.0
TX_AMPLITUDE_DEFAULT = 16384.0
TX_AMPLITUDE_MAX = 16384.0
NOISE_FALSE_ALARM = 1e-6
NOISE_OUTLIER_SIGMAS = 5.0
RX_SNAPSHOT_INTERVAL = 0.1
RX_READ_RETRY_LIMIT = 3
RX_READ_RETRY_DELAY = 0.2
PROCESSING_REVISION = 'pilot-timing-fft-v2'


def read_rx_buffer(sdr):
    """Recover errno hidden by old Windows pylibiio NULL-pointer checks.

    libiio sets C errno when buffer creation fails. Some Windows bindings read
    GetLastError instead; leave a valid error untouched and never reuse stale errno.
    """
    ctypes.set_errno(0)
    try:
        return sdr.rx()
    except OSError as error:
        native_errno = ctypes.get_errno()
        if error.errno == 0 and native_errno:
            raise OSError(native_errno,'libiio: '+os.strerror(native_errno)+
                          '; Python bindings reported '+str(error)) from error
        raise


def retryable_rx_error(error):
    return error.errno in (0,errno.EINTR,errno.EAGAIN,errno.EIO,errno.ETIMEDOUT,110,10060)


def sdr_runtime_info():
    """Record the executing Python and loaded bindings, without opening an SDR."""
    import sys
    from importlib.metadata import PackageNotFoundError, version
    result = dict(python=sys.version,python_executable=sys.executable)
    for package in ('pyadi-iio','pylibiio'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    binding = sys.modules.get('iio')
    if binding is not None:
        result['iio_module'] = getattr(binding,'__file__',None)
        result['libiio_version'] = getattr(binding,'version',None)
        result['libiio_library'] = getattr(getattr(binding,'_lib',None),'_name',None)
    return result


def tx_amplitude(settings):
    try:
        value = float(str(settings.get('tx_amplitude',TX_AMPLITUDE_DEFAULT)).replace(',','.'))
    except (TypeError,ValueError) as error:
        raise ValueError('Цифровая амплитуда TX: 1…16384') from error
    if not math.isfinite(value) or not 1<=value<=TX_AMPLITUDE_MAX:
        raise ValueError('Цифровая амплитуда TX: 1…16384')
    return value


def clipping_fraction(iq):
    """Fraction near the AD936x signed 12-bit I/Q rails, including a margin."""
    samples = np.asarray(iq)
    if not samples.size:
        return 0.
    limit = ADC_FULL_SCALE - 8
    return float(np.mean((abs(samples.real)>=limit) | (abs(samples.imag)>=limit)))


@lru_cache(maxsize=8)
def rx_mixer(length, offset):
    wave = np.array([1,-1j,-1,1j],np.complex64)[(np.arange(length)+offset)%4]
    wave.setflags(write=False)
    return wave


def tx_stream_sample_rate(sdr):
    device = getattr(sdr,'_txdac',None)
    channel = device.find_channel('voltage0',True) if device is not None else None
    if channel is None or 'sampling_frequency' not in channel.attrs:
        return None
    return int(channel.attrs['sampling_frequency'].value)


def configure_tx_packet_path(sdr, sample_rate):
    # BIST and loopback survive previous applications and can replace DMA samples.
    for name,value in (('bist_prbs','0'),('bist_tone','0 0 0 0'),('loopback','0')):
        attr = sdr._ctrl.debug_attrs.get(name)
        if attr is not None:
            attr.value = value
            if int(attr.value.split()[0]) != 0:
                raise RuntimeError(f'Pluto не отключил тестовый режим TX: {name}={attr.value}')
    channel = sdr._txdac.find_channel('voltage0',True)
    if channel is None or 'sampling_frequency' not in channel.attrs:
        raise RuntimeError('Pluto не предоставляет частоту потока IQ TX')
    # The DMA clock is separate from the PHY clock when FPGA interpolation is enabled.
    channel.attrs['sampling_frequency'].value = str(sample_rate)
    actual = tx_stream_sample_rate(sdr)
    if actual != sample_rate:
        raise RuntimeError(f'Pluto установил другую частоту потока IQ TX: {actual}; '
                           f'ожидается {sample_rate} отсчётов/с')


def experiment_settings(settings, sdr=None, tx=False):
    """Serializable requested settings and hardware values actually read back."""
    result = dict(processing_revision=PROCESSING_REVISION, phy_version=1,
                  cfg=asdict(settings['cfg']), samples_per_symbol=SPS,
                  payload_fec=False, header_decoder='hard-majority',
                  adc_full_scale=ADC_FULL_SCALE)
    names = ('uri','frequency','gain','gap_ms') if tx else (
        'uri','frequency','gain','queue_buffers','squelch_dbfs',
        'environment_dbfs','receiver_noise_dbfs')
    result.update({name:settings[name] for name in names if name in settings})
    if tx:
        amplitude = tx_amplitude(settings)
        result.update(tx_amplitude=amplitude,tx_digital_full_scale=TX_DIGITAL_FULL_SCALE,
                      tx_unit_symbol_dbfs=20*math.log10(amplitude/TX_DIGITAL_FULL_SCALE))
    else:
        result['queue_buffers'] = rx_queue_capacity(settings)
        result['gain_mode'] = 'manual'
        result['acquisition'] = 'first valid header; start RX before TX'
        result['rx_read_retry_limit'] = RX_READ_RETRY_LIMIT
    if sdr is not None:
        prefix = 'tx' if tx else 'rx'
        actual = {}
        for name in ('sample_rate',prefix+'_lo',prefix+'_rf_bandwidth',prefix+'_hardwaregain_chan0'):
            value = getattr(sdr,name,None)
            if value is not None:
                actual[name] = float(value) if 'gain' in name else int(value)

        if tx:
            stream_rate = tx_stream_sample_rate(sdr)
            if stream_rate is not None:
                actual['tx_stream_sample_rate'] = stream_rate
            controller = getattr(sdr,'_ctrl',None)
            if controller is not None:
                for name in ('bist_prbs','bist_tone','loopback'):
                    attr = controller.debug_attrs.get(name)
                    if attr is not None:
                        actual[name] = attr.value
        result['actual'] = actual
        result['runtime'] = sdr_runtime_info()
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_path(folder, role, suffix):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(folder)/(role+'_runs')/(stamp+'_'+secrets.token_hex(4)+suffix)
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def append_attempt(path, record):
    if path is not None:
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')


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
            raw=np.asarray(read_rx_buffer(sdr)).ravel()
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
    if tx:
        sdr.tx_hardwaregain_chan0=-89.75
    sdr._ctx.set_timeout(2000)
    sdr.sample_rate = cfg.sample_rate
    if int(sdr.sample_rate) != cfg.sample_rate:
        raise RuntimeError(f'Pluto установил другую sample rate: {sdr.sample_rate}')
    # Digital IF avoids AD936x DC cancellation eating the OOK carrier.
    lo = frequency-cfg.sample_rate//4
    if tx:
        sdr.tx_enabled_channels=[0]
        configure_tx_packet_path(sdr,cfg.sample_rate)
        sdr.tx_lo=lo
        sdr.tx_rf_bandwidth=cfg.sample_rate
        sdr.tx_cyclic_buffer=False
        sdr.tx_hardwaregain_chan0=settings['gain']
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
    journal=None
    attempt=None
    failure=None
    sent=end_sent=0
    size=None
    try:
        cfg=settings['cfg']
        amplitude=tx_amplitude(settings)
        sdr=connect(settings,True)
        emit('log',f'Pluto подключён. TX LO={int(sdr.tx_lo)} Гц, Fs={int(sdr.sample_rate)}')
        emit('log',f'Цифровая амплитуда TX={amplitude:g}; '
             f'единичный символ {20*math.log10(amplitude/TX_DIGITAL_FULL_SCALE):.2f} dBFS. '
             'Один масштаб для всех модуляций; TX gain задаётся отдельно в dB.')
        if settings.get('results_folder'):
            journal=run_path(settings['results_folder'],'tx','.jsonl')
            emit('log','Журнал попыток TX: '+str(journal))
        run_settings=experiment_settings(settings,sdr,tx=True)
        cycle=0
        while not stop.is_set():
            cycle+=1
            tid=secrets.randbits(64)
            sent=end_sent=0
            attempt=dict(event='start',time_utc=utc_now(),transfer=f'{tid:016x}',
                cycle=cycle,expected_packets=source.total,source_bytes=len(source.raw),
                kind='RGB' if source.kind else 'text',width=source.width,height=source.height,
                source_digest=source.digest.hex(),transmitter_settings=run_settings)
            append_attempt(journal,attempt)
            emit('log',f'Цикл {cycle}, передача {tid:016x}: {source.total} пакетов, {len(source.raw)} байт')
            emit('progress',(0,source.total,cycle))
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
                iq=(frame*amplitude*np.exp(2j*np.pi*.25*np.arange(len(frame)))).astype(np.complex64)
                size=len(iq)
                sdr.tx(iq)
                # Count submitted buffers even if Stop interrupts the subsequent wait.
                if end:
                    end_sent+=1
                else:
                    sent+=1
                # push completion need not mean last DAC sample; conservatively wait a full buffer.
                stop.wait(size/cfg.sample_rate + settings['gap_ms']/1000)
                if not end:
                    emit('progress',(sent,source.total,cycle))
            complete=sent==source.total and end_sent==3
            append_attempt(journal,dict(event='finish',time_utc=utc_now(),
                transfer=attempt['transfer'],status='complete' if complete else 'cancelled',
                submitted_data_packets=sent,submitted_end_packets=end_sent))
            attempt=None
            tail=('Отправлен END; '+('остановлено пользователем.' if stop.is_set()
                                     else 'начинаю следующий цикл.')) if complete else 'Остановлено пользователем.'
            emit('log',f'Цикл {cycle}: в SDR отправлено {sent}/{source.total} пакетов. '+tail)
            if not complete:
                break
    except Exception as error:
        failure=str(error)
        raise
    finally:
        try:
            if attempt is not None:
                append_attempt(journal,dict(event='finish',time_utc=utc_now(),
                    transfer=attempt['transfer'],status='failed' if failure else 'cancelled',
                    error=failure,submitted_data_packets=sent,submitted_end_packets=end_sent))
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
    run_started=utc_now()
    run_clock=time.monotonic()
    run_settings=experiment_settings(settings)
    results=[]
    failure=None
    clipped_samples=[0]
    captured_samples=[0]
    read_errors=[0]
    buffer_restarts=[0]
    read_error_events=[]
    last_snapshot=-float('inf')
    snapshot_dirty=False
    decoder=StreamDecoder(settings['cfg'])
    threshold_dbfs=settings.get('squelch_dbfs')
    threshold_power=(None if threshold_dbfs is None else
                     ADC_FULL_SCALE**2*10**(float(threshold_dbfs)/10))
    def update_transport():
        if session is not None:
            session.transport_drops=dropped[0]
            session.transport_read_errors=read_errors[0]
            session.transport_buffer_restarts=buffer_restarts[0]
    try:
        sdr=connect(settings,False)
        run_settings=experiment_settings(settings,sdr)
        emit('log',f'Pluto подключён. RX LO={int(sdr.rx_lo)} Гц. Приём до Stop.')
        if threshold_dbfs is not None:
            emit('log',f'Порог эфирного фона включён: {float(threshold_dbfs):.1f} dBFS.')
        def collect():
            offset=0
            gap=None
            tail=0
            pretrigger=deque(maxlen=2)
            def enqueue(item):
                nonlocal gap
                iq,clipping=item
                try:
                    frames.put_nowait((iq,gap,clipping))
                    gap=None
                except queue.Full:
                    dropped[0]+=1
                    gap='overflow'
            try:
                while not capture_stop.is_set():
                    try:
                        raw=read_rx_buffer(sdr)
                    except OSError as error:
                        if capture_stop.is_set() or stop.is_set():
                            break
                        read_errors[0]+=1
                        retry=retryable_rx_error(error) and buffer_restarts[0]<RX_READ_RETRY_LIMIT
                        read_error_events.append(dict(time_utc=utc_now(),
                            elapsed_seconds=time.monotonic()-run_clock,error_type=type(error).__name__,
                            errno=error.errno,message=str(error),retry=retry,
                            captured_samples=captured_samples[0],traceback=traceback.format_exc()))
                        if not retry:
                            raise
                        emit('log',f'Сбой чтения Pluto: {error}. Пересоздаю RX-буфер '
                             f'({buffer_restarts[0]+1}/{RX_READ_RETRY_LIMIT}); разрыв IQ учтён.')
                        sdr.rx_destroy_buffer()
                        buffer_restarts[0]+=1
                        # The new buffer is not contiguous with any retained samples.
                        pretrigger.clear()
                        tail=0
                        offset=0
                        gap='read_error'
                        if capture_stop.wait(RX_READ_RETRY_DELAY*buffer_restarts[0]):
                            break
                        continue
                    raw=np.asarray(raw,np.complex64)
                    # fs/4 oscillator has exactly four states, no growing float phase.
                    iq=raw*rx_mixer(len(raw),offset)
                    offset=(offset+len(raw))%4
                    clipping=clipping_fraction(raw)
                    clipped_samples[0]+=int(round(clipping*len(raw)))
                    captured_samples[0]+=len(raw)
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
                            gap='squelch'
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
                error=errors.get()
                raise RuntimeError(f'Ошибка чтения SDR после {buffer_restarts[0]} пересозданий RX-буфера: {error}. '
                    'Закройте программы, использующие этот Pluto, и переподключите USB. '
                    'Если ошибка повторяется, проверьте сетевой URI Pluto и версии libiio; '
                    'подробности сохранены в журнале RX.') from error
            try:
                iq,gap,clipping=frames.get(timeout=.15)
            except queue.Empty:
                now=time.monotonic()
                if session and snapshot_dirty and now-last_snapshot>=RX_SNAPSHOT_INTERVAL:
                    update_transport()
                    emit('snapshot',session.snapshot())
                    snapshot_dirty=False
                    last_snapshot=now
                if now-last>.5:
                    emit('health',dict(candidates=decoder.candidates,header_failures=decoder.header_failures,
                        queue=frames.qsize(),queue_capacity=frames.maxsize,drops=dropped[0],
                        squelched=squelched[0],input_dbfs=input_dbfs[0],squelch_dbfs=threshold_dbfs,
                        read_errors=read_errors[0],buffer_restarts=buffer_restarts[0]))
                    last=now
                continue
            if gap:
                decoder.reset()
                now=time.monotonic()
                if gap=='overflow' and now-last_gap_log>=1:
                    emit('log',f'Перегрузка обработки: потеряно IQ-буферов {dropped[0]}. Поиск пакета заново.')
                    last_gap_log=now
            if clipping>.001 and time.monotonic()-last_clip>3:
                last_clip=time.monotonic()
                emit('log','Обнаружено ограничение I/Q. Уменьшите RX gain / уровень TX.')
            packets=decoder.feed(iq,stop)
            for packet in packets:
                # Every header carries the expected count. Losing packet zero
                # must not exclude the rest of the same laboratory attempt.
                if session is None or session.meta.transfer != packet.meta.transfer:
                    if session is not None:
                        update_transport()
                        results.append(session.stats(True))
                        emit('log','Предыдущий результат: '+session.save(folder,True))
                    session=Reception(packet.meta,reference,settings['cfg'].mod,run_settings)
                    last_snapshot=-float('inf')
                    if packet.meta.seq != 0:
                        emit('log',f'Первый принятый заголовок: №{packet.meta.seq}. '
                             'Пакет №0 не принят; учитываю его потерю и продолжаю эту передачу.')
                    suffix='' if session.reference_ok else ' — BER и SNR недоступны'
                    emit('log',f'Передача {packet.meta.transfer:016x}; пакетов: {packet.meta.total}; '+
                         session.reference_status+suffix)
                was_end=session.ended
                session.accept(packet)
                update_transport()
                snapshot_dirty=True
                if session.ended and not was_end:
                    emit('log','Получен END. Передача принята; RX останавливается.')
                    stop.set()
                    break
            now=time.monotonic()
            if session and snapshot_dirty and not stop.is_set() and now-last_snapshot>=RX_SNAPSHOT_INTERVAL:
                emit('snapshot',session.snapshot())
                snapshot_dirty=False
                last_snapshot=now
            if now-last>.5:
                emit('health',dict(candidates=decoder.candidates,header_failures=decoder.header_failures,
                    queue=frames.qsize(),queue_capacity=frames.maxsize,drops=dropped[0],
                    squelched=squelched[0],input_dbfs=input_dbfs[0],squelch_dbfs=threshold_dbfs,
                    read_errors=read_errors[0],buffer_restarts=buffer_restarts[0]))
                last=now
    except Exception as error:
        failure=str(error)
        raise
    finally:
        capture_stop.set()
        if capture:
            capture.join(timeout=3)
        if capture is not None and capture.is_alive() and failure is None:
            failure='Драйвер RX не завершился за 3 с. Закройте приложение перед повторным запуском.'
        if sdr is not None and (capture is None or not capture.is_alive()):
            sdr.rx_destroy_buffer()
        if session:
            update_transport()
            emit('snapshot',session.snapshot(True))
            results.append(session.stats(True))
            emit('log','Результат: '+session.save(folder,True))
        summary=run_path(folder,'rx','.json')
        report=dict(started_utc=run_started,finished_utc=utc_now(),
            elapsed_seconds=time.monotonic()-run_clock,receiver_settings=run_settings,
            status='failed' if failure else 'finished' if results else 'no_valid_header',
            error=failure,transfers=results,per=results[-1]['per'] if results else None,
            ber=results[-1]['ber'] if results else None,transport_dropped_buffers=dropped[0],
            transport_read_errors=read_errors[0],transport_buffer_restarts=buffer_restarts[0],
            read_error_events=read_error_events,
            squelched_buffers=squelched[0],captured_samples=captured_samples[0],
            clipped_samples=clipped_samples[0],
            clipping_fraction=clipped_samples[0]/captured_samples[0] if captured_samples[0] else None,
            candidates=decoder.candidates,header_failures=decoder.header_failures)
        summary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        emit('log','Журнал запуска RX: '+str(summary))
        if not results and failure is None:
            emit('log','Валидных заголовков нет: ID передачи и PER неизвестны. Сопоставьте запуск с журналом TX.')
        if capture is not None and capture.is_alive():
            raise RuntimeError('Драйвер RX не завершился за 3 с. Закройте приложение перед повторным запуском.')
