import unittest

import numpy as np

from spectrum_debugger import calculate_spectrum


class SpectrumDebuggerTests(unittest.TestCase):
    def test_tone_frequency_and_level(self):
        size=32768
        rate=1_000_000
        center=2_400_000_000
        offset=rate//8
        time=np.arange(size)/rate
        iq=1024*np.exp(2j*np.pi*offset*time)
        frequency,power=calculate_spectrum(iq,rate,center)
        peak=int(np.argmax(power))
        self.assertAlmostEqual(frequency[peak],center+offset,delta=rate/size)
        self.assertAlmostEqual(10*np.log10(power[peak]),-6.0206,delta=.05)

    def test_rejects_too_short_input(self):
        with self.assertRaises(ValueError):
            calculate_spectrum(np.zeros(8),1_000_000,2_400_000_000)


if __name__=='__main__':
    unittest.main(verbosity=2)
