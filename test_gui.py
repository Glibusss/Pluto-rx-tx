import unittest

from gui import gain_power_text


class GuiTests(unittest.TestCase):
    def test_pluto_gain_has_live_linear_power_description(self):
        self.assertEqual(gain_power_text('-30'),'линейно по мощности: ×0,001')
        self.assertEqual(gain_power_text('20'),'линейно по мощности: ×100')
        self.assertEqual(gain_power_text('bad'),'линейно по мощности: —')


if __name__=='__main__':
    unittest.main(verbosity=2)
