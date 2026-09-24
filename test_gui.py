import unittest

from gui import (APP_ICON_BACKGROUND, APP_ICON_SIGNAL, app_icon_pixels,
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


if __name__=='__main__':
    unittest.main(verbosity=2)
