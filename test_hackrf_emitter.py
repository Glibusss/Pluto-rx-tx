import tempfile
import unittest
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import numpy as np

from hackrf_emitter import (EmitterConfig, build_command, build_cycle,
                            cleanup_temp_directory, complex_to_cs8, cycle_sample_count,
                            frequency_track,
                            discover_hackrf_devices, find_hackrf_info,
                            find_hackrf_transfer, load_iq_file, parse_hackrf_info,
                            select_filter_bandwidth, stop_hackrf_process)


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
            EmitterConfig(bandwidth_hz=7_000_000),
            EmitterConfig(txvga_gain=48),
            EmitterConfig(amplitude=0),
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
                self.assertEqual(len(iq), 200_000)
                self.assertGreater(np.count_nonzero(iq), len(iq) // 2)

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
