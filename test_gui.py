import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from gui import (APP_ICON_BACKGROUND, APP_ICON_SIGNAL, App, app_icon_pixels,
                 gain_power_text)
from modem import Config


class GuiTests(unittest.TestCase):
    def test_real_windows_build_with_requested_tabs_and_adaptive_profiles(self):
        from tx_emitter import EmitterApp
        from modulations import MODS
        for tx in (False, True):
            app=App(tx)
            try:
                app.withdraw()
                app.update_idletasks()
                self.assertEqual([app.top_tabs.tab(t,'text') for t in app.top_tabs.tabs()],
                                 ['Выбор данных, выбор устройства','Настройки'])
                self.assertEqual([app.tabs.tab(t,'text') for t in app.tabs.tabs()],
                                 ['Принятые данные','Статистика','Спектр','Решения демодулятора','Лог'])
                for tab in app.top_tabs.tabs():
                    self.assertLessEqual(app.nametowidget(tab).winfo_reqheight(),int(app.top_tabs.cget('height')))
                from modem import make_frame, decode_frame, preamble_wave
                from session import Source, Reception
                from spectrum import calculate_spectrum
                import numpy as np
                cfg=Config()
                source=Source.text('Проверка принятого текста')
                packet=decode_frame(make_frame(source.meta(81,0),source.payload(0),cfg)[len(preamble_wave(cfg)):],cfg,0)
                reception=Reception(packet.meta,None,cfg.mod)
                reception.accept(packet)
                app.update_snapshot(reception.snapshot())
                self.assertIn('Проверка принятого текста',app.received_text.get('1.0','end'))
                frequency,power=calculate_spectrum(np.ones(512),1000000,2400000000)
                app.spectrum_trace=dict(frequency=frequency,power=power,signal_frequency=2400000000)
                app.draw_spectrum()
                self.assertGreater(len(app.spectrum_canvas.find_all()),10)
                for mod in MODS:
                    app.mod.set(mod)
                    app.update_idletasks()
                    self.assertEqual(app.settings()['cfg'].mod,mod)
                    self.assertEqual(len(app.phy.mode_widgets),0 if mod=='BPSK' else 1)
            finally:
                app.destroy()
        app=EmitterApp()
        try:
            app.withdraw()
            for mod in MODS:
                app.phy.mod.set(mod)
                app.update_idletasks()
                cfg=app.settings()
                self.assertEqual(cfg.modem_config().mod,mod)
                self.assertEqual(len(app.phy.mode_widgets),0 if mod=='BPSK' else 1)
            app.phy.mod.set('2-FSK')
            app.phy.fsk_deviation_mhz.set('0,1')
            self.assertEqual(app.settings().modem_config().fsk_deviation_hz,100000)
            self.assertTrue(any('МГц' in str(w.cget('text')) for w in app.phy.mode_frame.winfo_children()
                                if w.winfo_class()=='TLabel'))
            app.phy.mod.set('BPSK')
            app.phy.phase_deg.set('ignored')
            app.phy.fsk_deviation_mhz.set('ignored')
            self.assertEqual(app.settings().modem_config().mod,'BPSK')
            app.signal_mode.set('noise')
            app.update_modes()
            self.assertEqual(str(app.mod_widget.cget('state')),'disabled')
        finally:
            app.destroy()

    def test_custom_radio_icon_replaces_default_tk_feather(self):
        pixels=app_icon_pixels()
        self.assertEqual((len(pixels),len(pixels[0])),(32,32))
        colors={color for row in pixels for color in row}
        self.assertEqual(colors,{APP_ICON_BACKGROUND,APP_ICON_SIGNAL})
        self.assertGreater(sum(color==APP_ICON_SIGNAL for row in pixels for color in row),40)

    def test_pluto_gain_has_live_linear_power_description(self):
        self.assertEqual(gain_power_text('-30'),'линейно по мощности: ×0,001')
        self.assertEqual(gain_power_text('20'),'линейно по мощности: ×100')
        self.assertEqual(gain_power_text('bad'),'линейно по мощности: —')

    def test_tx_settings_include_selected_digital_amplitude(self):
        values=dict(mod='BPSK',pre=True,cp=False,rate='1',cfo='40000',gain='-10',
                    freq='2400',gap='20',queue_buffers='512',uri='usb:1.2.3',
                    folder='received',tx_amplitude='4096')
        app=SimpleNamespace(tx=True,environment_dbfs=None,receiver_noise_dbfs=None,
                            phy=Mock(config=Mock(return_value=Config())),
                            **{name:Mock(get=Mock(return_value=value)) for name,value in values.items()})
        settings=App.settings(app)
        self.assertEqual(settings['cfg'],Config())
        self.assertEqual(settings['tx_amplitude'],4096)
        self.assertEqual(settings['gain'],-10)
        app.tx_amplitude.get.return_value='20000'
        with self.assertRaisesRegex(ValueError,'амплитуда'):
            App.settings(app)

    def test_receiver_noise_calibration_requires_50_ohm_warning(self):
        app=SimpleNamespace(tx=False,worker=None,_start_calibration=Mock())
        with patch('gui.messagebox.askokcancel',return_value=False) as warning:
            App.calibrate_receiver_noise(app)
        warning.assert_called_once()
        self.assertIn('50 Ω',warning.call_args.args[1])
        app._start_calibration.assert_not_called()

        with patch('gui.messagebox.askokcancel',return_value=True):
            App.calibrate_receiver_noise(app)
        app._start_calibration.assert_called_once_with('receiver_noise')

    def test_environment_calibration_needs_no_terminator_confirmation(self):
        app=SimpleNamespace(_start_calibration=Mock())
        App.calibrate_environment(app)
        app._start_calibration.assert_called_once_with('environment')


if __name__=='__main__':
    unittest.main(verbosity=2)
