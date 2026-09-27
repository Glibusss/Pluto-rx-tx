import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from gui import (APP_ICON_BACKGROUND, APP_ICON_SIGNAL, App, app_icon_pixels,
                 gain_power_text)
from modem import Config


class GuiTests(unittest.TestCase):
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
