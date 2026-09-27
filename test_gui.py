import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from gui import (APP_ICON_BACKGROUND, APP_ICON_SIGNAL, App, app_icon_pixels,
                 gain_power_text)


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
