import io
import tempfile
import unittest
from pathlib import Path
import signal
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
from scipy.signal import resample_poly

from hackrf_emitter import (EmitterConfig, build_command, build_cycle,
                            cleanup_temp_directory, complex_to_cs8, cycle_sample_count,
                            frequency_track,
                            iter_cycle_chunks, packet_source, write_cycle, WaveformCancelled,
                            discover_hackrf_devices, find_hackrf_info,
                            find_hackrf_transfer, load_iq_file, parse_hackrf_info,
                            run_transmitter, select_filter_bandwidth,
                            stop_hackrf_process, wait_for_hackrf_ready)
from modem import Config, MODS, PREAMBLE, frame_len, StreamDecoder
from session import Source
from tx_emitter import EmitterApp


class HackRFEmitterTests(unittest.TestCase):
    def test_defaults_are_safe_and_valid(self):
        cfg = EmitterConfig()
        cfg.validate()
        self.assertEqual(cfg.txvga_gain, 0)
        self.assertFalse(cfg.rf_amp)
        self.assertLess(cfg.amplitude, 0.5)

    def test_validation_rejects_invalid_rf_parameters(self):
        cases = [
            EmitterConfig(frequency_hz=999_999),
            EmitterConfig(sample_rate=3_000_000),
            EmitterConfig(bandwidth_hz=9_000_000),
            EmitterConfig(txvga_gain=48),
            EmitterConfig(amplitude=-.001),
            EmitterConfig(amplitude=float('inf')),
            EmitterConfig(sweep_period_ms=0),
            EmitterConfig(sweep_period_ms=1e308,tuning_mode='sweep'),
            EmitterConfig(pulse_off_ms=-1),
            EmitterConfig(packet_kind='invalid'),
            EmitterConfig(frequency_hz=1_000_000, bandwidth_hz=1_000_000),
        ]
        for cfg in cases:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                cfg.validate()

    def test_cs8_is_signed_interleaved_and_clipped(self):
        raw = complex_to_cs8(np.array([1 + 0j, -1 + .5j, 2 - 2j], np.complex64))
        values = np.frombuffer(raw, np.int8)
        np.testing.assert_array_equal(values, [127, 0, -127, 64, 127, -127])

    def test_pulse_cycle_has_exact_zero_off_interval(self):
        cfg = EmitterConfig(sample_rate=2_000_000, bandwidth_hz=400_000,
                            signal_mode='noise', operation_mode='pulse',
                            pulse_on_ms=5, pulse_off_ms=7)
        iq = build_cycle(cfg)
        on = int(cfg.sample_rate * cfg.pulse_on_ms / 1000)
        self.assertEqual(len(iq), cycle_sample_count(cfg))
        self.assertGreater(float(np.max(np.abs(iq[:on]))), 0.1)
        self.assertTrue(np.all(iq[on:] == 0))

    def test_sweep_track_stays_inside_requested_band(self):
        cfg = EmitterConfig(sample_rate=2_000_000, bandwidth_hz=500_000,
                            tuning_mode='sweep', sweep_period_ms=10)
        track = frequency_track(cfg, 20_000)
        self.assertLessEqual(float(np.ptp(track)), cfg.bandwidth_hz)
        self.assertGreater(float(np.ptp(track)), cfg.bandwidth_hz * 0.7)
        self.assertAlmostEqual(float(np.mean(track)), 0, places=7)

    def test_pulse_sweep_completes_one_pass_per_burst(self):
        cfg = EmitterConfig(sample_rate=2_000_000, bandwidth_hz=500_000,
                            tuning_mode='sweep', operation_mode='pulse',
                            sweep_period_ms=1000, pulse_on_ms=5, pulse_off_ms=5)
        track = frequency_track(cfg, 10_000)
        self.assertGreater(float(np.ptp(track)), cfg.bandwidth_hz * 0.7)

    def test_packet_mode_uses_existing_modem_for_all_modulations(self):
        for modulation in ('BPSK', 'QPSK', 'QAM-16', '2-FSK'):
            cfg = EmitterConfig(sample_rate=2_000_000, bandwidth_hz=500_000,
                                modulation=modulation)
            iq = build_cycle(cfg)
            with self.subTest(modulation=modulation):
                self.assertEqual(iq.dtype, np.complex64)
                self.assertEqual(len(iq), cycle_sample_count(cfg))
                self.assertGreaterEqual(len(iq),200_000)
                packet_samples=len(PREAMBLE)+frame_len(Config(modulation,True,False,2000000))
                self.assertEqual(len(iq)%packet_samples,0)
                self.assertGreater(np.count_nonzero(iq), len(iq) // 2)

    def test_packet_content_decodes_as_random_text_or_red_rgb_for_every_modulation(self):
        for kind in ('text','image'):
            for modulation in MODS:
                with self.subTest(kind=kind,modulation=modulation):
                    cfg=EmitterConfig(sample_rate=2000000,packet_kind=kind,modulation=modulation)
                    source=packet_source(cfg)
                    iq=build_cycle(cfg,source,63)
                    encoded=np.frombuffer(complex_to_cs8(iq),np.int8)
                    received=encoded[::2].astype(float)/127+1j*encoded[1::2].astype(float)/127
                    decoder=StreamDecoder(Config(modulation,True,False,2000000,0))
                    packets=decoder.feed(received)
                    self.assertGreater(len(packets),1)
                    self.assertTrue(all(p.crc_ok for p in packets))
                    self.assertTrue(all(p.payload==source.raw for p in packets))
                    self.assertEqual(packets[0].meta.digest,source.digest)
                    self.assertEqual(packets[0].meta.kind,int(kind=='image'))
                    self.assertEqual(len(decoder.buffer),0)
                    if kind=='image':
                        self.assertEqual((packets[0].meta.width,packets[0].meta.height),(16,16))
                        self.assertEqual(source.raw,bytes((255,0,0))*256)
                    else:
                        self.assertEqual((packets[0].meta.width,packets[0].meta.height),(0,0))
                        source.raw.decode('utf-8')

    def test_resampled_hackrf_frames_decode_at_pluto_rate(self):
        for rate in (8000000,20000000):
            for modulation in MODS:
                with self.subTest(rate=rate,modulation=modulation):
                    cfg=EmitterConfig(sample_rate=rate,modulation=modulation,packet_kind='image')
                    source=packet_source(cfg)
                    codes=np.frombuffer(complex_to_cs8(build_cycle(cfg,source,29)),np.int8)
                    base=codes[::2].astype(float)/127+1j*codes[1::2].astype(float)/127
                    received=resample_poly(base,2000000,rate)
                    decoder=StreamDecoder(Config(modulation,True,False,2000000,0))
                    packets=decoder.feed(received)
                    self.assertGreater(len(packets),1)
                    self.assertTrue(all(p.crc_ok and p.payload==source.raw for p in packets))
                    self.assertEqual(len(decoder.buffer),0)

    def test_gui_binds_packet_kind_and_extended_controls(self):
        values=dict(gain='47',frequency='2400',sample_rate='8',bandwidth='8',
                    amplitude='200',signal_mode='packet',modulation='BPSK',packet_kind='image',
                    tuning_mode='fixed',operation_mode='continuous',sweep_period='12000',
                    pulse_on='20000',pulse_off='0',iq_path='',iq_format='CS8',
                    serial='',executable='hackrf_transfer',rf_amp=True)
        app=SimpleNamespace(_number=EmitterApp._number,
                            **{name:Mock(get=Mock(return_value=value)) for name,value in values.items()})
        cfg=EmitterApp.settings(app)
        self.assertEqual(cfg.packet_kind,'image')
        self.assertEqual(cfg.amplitude,2)
        self.assertEqual(cfg.bandwidth_hz,cfg.sample_rate)
        self.assertEqual(cfg.txvga_gain,47)
        self.assertTrue(cfg.rf_amp)

    def test_each_start_generates_new_random_text(self):
        cfg=EmitterConfig(packet_kind='text')
        with patch('hackrf_emitter.secrets.token_hex',side_effect=['a'*768,'b'*768]):
            first,second=packet_source(cfg),packet_source(cfg)
        self.assertNotEqual(first.raw,second.raw)
        self.assertNotEqual(first.digest,second.digest)
        self.assertEqual(first.total,1)

    def test_extended_controls_allow_full_band_overdrive_and_long_cycles(self):
        for cfg in (EmitterConfig(bandwidth_hz=8000000,amplitude=2),
                    EmitterConfig(sweep_period_ms=12000,pulse_on_ms=20000,pulse_off_ms=0),
                    EmitterConfig(amplitude=0),EmitterConfig(amplitude=.0001)):
            with self.subTest(cfg=cfg):
                cfg.validate()
        cfg=EmitterConfig(bandwidth_hz=8000000)
        self.assertGreaterEqual(select_filter_bandwidth(cfg),cfg.bandwidth_hz)

    def test_overdrive_saturates_without_int8_wrap_and_zero_amplitude_is_silent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wave.cf32'
            np.array([1+0j,0+1j,0j,-1-1j],'<c8').tofile(path)
            for amplitude in (0,2,1e308):
                cfg=EmitterConfig(signal_mode='file',iq_path=str(path),iq_format='cf32',amplitude=amplitude)
                iq=build_cycle(cfg)
                self.assertTrue(np.all(np.isfinite(iq)))
                values=np.frombuffer(complex_to_cs8(iq),np.int8)
                if amplitude==0:
                    self.assertTrue(np.all(values==0))
                else:
                    np.testing.assert_array_equal(values,[127,0,0,127,0,0,-127,-127])

    def test_large_cycle_is_written_in_bounded_chunks_with_exact_silent_tail(self):
        cfg=EmitterConfig(sample_rate=2000000,operation_mode='pulse',pulse_on_ms=1200,
                          pulse_off_ms=1200,packet_kind='image')
        chunks=iter_cycle_chunks(cfg,chunk_samples=32768,source=packet_source(cfg),transfer=64)
        total=0
        for chunk in chunks:
            self.assertLessEqual(len(chunk),32768)
            total+=len(chunk)
        self.assertEqual(total,4800000)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'large.cs8'
            samples=write_cycle(cfg,path,source=packet_source(cfg),transfer=64,chunk_samples=32768)
            self.assertEqual(samples,4800000)
            self.assertEqual(path.stat().st_size,9600000)
            self.assertGreater(path.stat().st_size,8*1024**2)
            with path.open('rb') as stream:
                stream.seek(2400000*2)
                while chunk:=stream.read(65536):
                    self.assertFalse(any(chunk))

    def test_noise_filter_and_sweep_are_continuous_across_chunk_boundaries(self):
        for bandwidth in (500000,2000000):
            cfg=EmitterConfig(sample_rate=2000000,signal_mode='noise',bandwidth_hz=bandwidth,
                              tuning_mode='sweep',operation_mode='pulse',pulse_on_ms=20,pulse_off_ms=3)
            small=np.concatenate(list(iter_cycle_chunks(cfg,chunk_samples=1024)))
            large=np.concatenate(list(iter_cycle_chunks(cfg,chunk_samples=8192)))
            np.testing.assert_allclose(small,large,atol=1e-6,rtol=1e-6)
            self.assertTrue(np.all(small[40000:]==0))

    def test_stop_during_iq_generation_does_not_launch_hackrf(self):
        with (patch('hackrf_emitter.resolve_executable',return_value='hackrf_transfer.exe'),
              patch('hackrf_emitter.wait_for_hackrf_ready',return_value=(True,'ready')),
              patch('hackrf_emitter.write_cycle',side_effect=WaveformCancelled),
              patch('hackrf_emitter.subprocess.Popen') as popen):
            run_transmitter(EmitterConfig(),threading.Event(),lambda *event:None)
        popen.assert_not_called()

    def test_chunk_generator_honors_stop_during_reference_pass(self):
        stop=Mock()
        stop.is_set.side_effect=[False,False,True]
        cfg=EmitterConfig(sample_rate=2000000,signal_mode='noise',operation_mode='pulse',
                          pulse_on_ms=10000,pulse_off_ms=10000)
        with self.assertRaises(WaveformCancelled):
            list(iter_cycle_chunks(cfg,chunk_samples=1024,stop=stop))

    def test_loads_cs8_and_cf32_iq_files(self):
        with tempfile.TemporaryDirectory() as folder:
            cs8_path = Path(folder) / 'input.cs8'
            cs8_path.write_bytes(np.array([127, 0, -127, 64], np.int8).tobytes())
            cs8 = load_iq_file(cs8_path, 'cs8')
            np.testing.assert_allclose(cs8, [1 + 0j, -1 + (64 / 127) * 1j])

            cf32_path = Path(folder) / 'input.cf32'
            expected = np.array([.25 + .5j, -.75 - .125j], dtype='<c8')
            cf32_path.write_bytes(expected.tobytes())
            np.testing.assert_array_equal(load_iq_file(cf32_path, 'cf32'), expected)

    def test_iq_file_is_repeated_and_pulsed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.cf32'
            np.array([1 + 0j, 0 + 1j], dtype='<c8').tofile(path)
            cfg = EmitterConfig(sample_rate=2_000_000, bandwidth_hz=400_000,
                                signal_mode='file', iq_path=str(path), iq_format='cf32',
                                operation_mode='pulse', pulse_on_ms=1, pulse_off_ms=1,
                                amplitude=.5)
            iq = build_cycle(cfg)
            on = 2000
            self.assertEqual(len(iq), 4000)
            self.assertGreater(np.count_nonzero(iq[:on]), on - 4)
            self.assertTrue(np.all(iq[on:] == 0))

    def test_iq_file_validation_checks_binary_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'broken.cs8'
            path.write_bytes(b'123')
            cfg = EmitterConfig(signal_mode='file', iq_path=str(path), iq_format='cs8')
            with self.assertRaisesRegex(ValueError, 'кратным'):
                cfg.validate()

    def test_command_contains_explicit_tx_controls_and_repeat(self):
        cfg = EmitterConfig(serial='00000001', txvga_gain=17, rf_amp=True)
        command = build_command(cfg, Path('wave.cs8'), 'hackrf_transfer.exe')
        self.assertEqual(command[0], 'hackrf_transfer.exe')
        self.assertIn('-R', command)
        self.assertEqual(command[command.index('-x') + 1], '17')
        self.assertEqual(command[command.index('-a') + 1], '1')
        self.assertEqual(command[command.index('-d') + 1], '00000001')
        self.assertEqual(command[command.index('-b') + 1], str(select_filter_bandwidth(cfg)))

    def test_finds_hackrf_tools_from_explicit_install_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            transfer = Path(folder) / 'hackrf_transfer.exe'
            info = Path(folder) / 'hackrf_info.exe'
            transfer.write_bytes(b'fake')
            info.write_bytes(b'fake')
            self.assertEqual(Path(find_hackrf_transfer(str(transfer))), transfer.resolve())
            self.assertEqual(Path(find_hackrf_info(str(transfer))), info.resolve())

    def test_parses_and_discovers_multiple_hackrf_serials(self):
        output = ('Found HackRF\nIndex: 0\nSerial number: 0000000000000001\n\n'
                  'Found HackRF\nIndex: 1\nSerial number: ABCDEF0123456789\n')
        self.assertEqual(parse_hackrf_info(output),
                         ['0000000000000001', 'ABCDEF0123456789'])
        with tempfile.TemporaryDirectory() as folder:
            transfer = Path(folder) / 'hackrf_transfer.exe'
            info = Path(folder) / 'hackrf_info.exe'
            transfer.write_bytes(b'fake')
            info.write_bytes(b'fake')
            calls = []

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(returncode=0, stdout=output)

            used_info, serials, diagnostic = discover_hackrf_devices(
                str(transfer), runner=runner)
            self.assertEqual(Path(used_info), info.resolve())
            self.assertEqual(serials, ['0000000000000001', 'ABCDEF0123456789'])
            self.assertEqual(diagnostic, output)
            self.assertEqual(calls[0][0], [str(info.resolve())])

    def test_device_discovery_treats_no_boards_as_an_empty_result(self):
        with tempfile.TemporaryDirectory() as folder:
            transfer = Path(folder) / 'hackrf_transfer.exe'
            info = Path(folder) / 'hackrf_info.exe'
            transfer.write_bytes(b'fake')
            info.write_bytes(b'fake')

            def runner(_command, **_kwargs):
                return SimpleNamespace(returncode=1, stdout='No HackRF boards found.\n')

            _info, serials, _output = discover_hackrf_devices(
                str(transfer), runner=runner)
            self.assertEqual(serials, [])

    def test_device_discovery_reports_real_hackrf_info_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            transfer = Path(folder) / 'hackrf_transfer.exe'
            info = Path(folder) / 'hackrf_info.exe'
            transfer.write_bytes(b'fake')
            info.write_bytes(b'fake')

            def runner(_command, **_kwargs):
                return SimpleNamespace(returncode=1, stdout='hackrf_init() failed: USB error\n')

            with self.assertRaisesRegex(RuntimeError, 'кодом 1'):
                discover_hackrf_devices(str(transfer), runner=runner)

    def test_waits_until_selected_hackrf_is_released(self):
        with tempfile.TemporaryDirectory() as folder:
            transfer = Path(folder) / 'hackrf_transfer.exe'
            info = Path(folder) / 'hackrf_info.exe'
            transfer.write_bytes(b'fake')
            info.write_bytes(b'fake')
            results = iter([
                SimpleNamespace(returncode=1,
                                stdout='hackrf_open() failed: Access denied\n'),
                SimpleNamespace(
                    returncode=0,
                    stdout='Found HackRF\nSerial number: ABCDEF0123456789\n'),
            ])
            sleeps = []

            ready, output = wait_for_hackrf_ready(
                str(transfer), 'abcdef0123456789', attempts=2, interval=0.1,
                runner=lambda _command, **_kwargs: next(results),
                sleeper=sleeps.append)

            self.assertTrue(ready)
            self.assertIn('ABCDEF0123456789', output)
            self.assertEqual(sleeps, [0.1])

    def test_selected_hackrf_does_not_match_another_available_board(self):
        with tempfile.TemporaryDirectory() as folder:
            transfer = Path(folder) / 'hackrf_transfer.exe'
            info = Path(folder) / 'hackrf_info.exe'
            transfer.write_bytes(b'fake')
            info.write_bytes(b'fake')

            def runner(_command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout='Found HackRF\nSerial number: OTHER\n')

            ready, diagnostic = wait_for_hackrf_ready(
                str(transfer), 'WANTED', attempts=1, runner=runner)

            self.assertFalse(ready)
            self.assertIn('OTHER', diagnostic)

    def test_transmitter_does_not_start_while_hackrf_is_busy(self):
        cfg = EmitterConfig(executable='hackrf_transfer.exe')
        events = []
        with (patch('hackrf_emitter.resolve_executable',
                    return_value='hackrf_transfer.exe'),
              patch('hackrf_emitter.wait_for_hackrf_ready',
                    return_value=(False, 'Access denied')),
              patch('hackrf_emitter.subprocess.Popen') as popen):
            with self.assertRaisesRegex(RuntimeError, 'Access denied'):
                run_transmitter(cfg, threading.Event(),
                                lambda *event: events.append(event))

        popen.assert_not_called()
        self.assertEqual(events[0], ('status', 'Ожидание готовности HackRF…'))

    def test_transmitter_retries_transient_start_code_one(self):
        stop = threading.Event()
        events = []

        class ImmediateFailure:
            returncode = 1
            stdout = io.StringIO('hackrf_open() failed: Access denied\n')

            def poll(self):
                return self.returncode

        class RunningProcess:
            stdout = io.StringIO('')

            def __init__(self):
                self.returncode = None
                self.polls = 0

            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                self.polls += 1
                if self.polls >= 4:
                    stop.set()
                return None

            def send_signal(self, _value):
                self.returncode = 0

            def wait(self, timeout):
                return self.returncode

            def terminate(self):
                self.returncode = 0

        processes = [ImmediateFailure(), RunningProcess()]
        cfg = EmitterConfig(executable='hackrf_transfer.exe')
        with (patch('hackrf_emitter.resolve_executable',
                    return_value='hackrf_transfer.exe'),
              patch('hackrf_emitter.wait_for_hackrf_ready',
                    side_effect=[(True, 'ready'), (True, 'ready')]) as ready,
              patch('hackrf_emitter.write_cycle',side_effect=lambda _cfg,path,*args:
                    (Path(path).write_bytes(b'\0\0') or 1)),
              patch('hackrf_emitter.STARTUP_STABILITY_SECONDS', 0),
              patch('hackrf_emitter.START_RETRY_DELAYS', (0,)),
              patch('hackrf_emitter.subprocess.Popen',
                    side_effect=processes) as popen):
            run_transmitter(cfg, stop, lambda *event: events.append(event))

        self.assertEqual(popen.call_count, 2)
        self.assertEqual(ready.call_count, 2)
        self.assertTrue(any('повтор через' in value for kind, value in events
                            if kind == 'log'))
        self.assertIn(('log', 'HackRF освобождён и готов к повторному запуску.'),
                      events)

    def test_temp_cleanup_retries_a_windows_style_file_lock(self):
        attempts = []
        sleeps = []

        def remover(path):
            attempts.append(Path(path))
            if len(attempts) < 3:
                raise PermissionError(32, 'file is in use')

        self.assertTrue(cleanup_temp_directory(
            'temporary-waveform', remover=remover, sleeper=sleeps.append))
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [0.05, 0.15])

    def test_temp_cleanup_lock_never_escapes_to_gui(self):
        events = []

        def remover(_path):
            raise PermissionError(32, 'file is in use')

        self.assertFalse(cleanup_temp_directory(
            'temporary-waveform', lambda *event: events.append(event),
            remover=remover, sleeper=lambda _delay: None))
        self.assertEqual(events[0][0], 'log')
        self.assertIn('оставлен', events[0][1])

    @unittest.skipUnless(hasattr(signal, 'CTRL_BREAK_EVENT'), 'Windows signal')
    def test_hackrf_process_receives_graceful_ctrl_break(self):
        class FakeProcess:
            def __init__(self):
                self.running = True
                self.signals = []
                self.terminated = False

            def poll(self):
                return None if self.running else 0

            def send_signal(self, value):
                self.signals.append(value)

            def wait(self, timeout):
                self.running = False
                return 0

            def terminate(self):
                self.terminated = True

        process = FakeProcess()
        self.assertEqual(stop_hackrf_process(process, windows=True), 'ctrl-break')
        self.assertEqual(process.signals, [signal.CTRL_BREAK_EVENT])
        self.assertFalse(process.terminated)

    def test_unresponsive_process_is_reported_without_an_unhandled_timeout(self):
        class StuckProcess:
            pid = 123

            def poll(self):
                return None

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout):
                raise subprocess.TimeoutExpired('hackrf_transfer', timeout)

        self.assertEqual(stop_hackrf_process(StuckProcess(), windows=False),
                         'still-running')


if __name__ == '__main__':
    unittest.main(verbosity=2)
