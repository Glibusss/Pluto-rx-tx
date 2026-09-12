import threading
import unittest
from unittest.mock import patch
import numpy as np
import radio
from modem import Config, StreamDecoder
from session import Source

class FakeTX:
    def __init__(self):
        self.calls=[]
        self.tx_lo=2399750000
        self.sample_rate=1000000
        self.destroyed=False
        self.tx_hardwaregain_chan0=-30
    def tx(self,iq):self.calls.append(iq.copy())
    def tx_destroy_buffer(self):self.destroyed=True

class NoWait:
    def is_set(self):return False
    def wait(self,duration):return False

class RadioTests(unittest.TestCase):
    def test_tx_once_end_and_zero_flush(self):
        sdr=FakeTX();source=Source.text('Hello Pluto')
        settings=dict(cfg=Config(),gap_ms=0)
        with patch.object(radio,'connect',return_value=sdr),patch.object(radio.time,'sleep'):
            radio.transmit(settings,source,NoWait(),lambda *x:None)
        self.assertEqual(len(sdr.calls),5) # 1 data + 3 END + final zero buffer
        self.assertTrue(sdr.destroyed)
        self.assertEqual(sdr.tx_hardwaregain_chan0,-89.75)
        self.assertTrue(np.all(sdr.calls[-1]==0))
        decoder=StreamDecoder(settings['cfg'])
        packets=[]
        for iq in sdr.calls:
            iq=iq/4096*np.exp(-2j*np.pi*.25*np.arange(len(iq)))
            packets+=decoder.feed(iq)
        self.assertEqual(len(packets),4)
        self.assertEqual(packets[0].payload,source.raw)
        self.assertEqual(sum(p.meta.end for p in packets),3)

    def test_tx_cleanup_on_failure(self):
        sdr=FakeTX()
        source=Source.text('Hello')
        calls=[0]
        def fail_first(iq):
            calls[0]+=1
            if calls[0]==1:raise OSError('hardware mock fault')
        sdr.tx=fail_first
        with patch.object(radio,'connect',return_value=sdr),patch.object(radio.time,'sleep'):
            with self.assertRaises(OSError):
                radio.transmit(dict(cfg=Config(),gap_ms=0),source,NoWait(),lambda *x:None)
        self.assertTrue(sdr.destroyed)
        self.assertEqual(sdr.tx_hardwaregain_chan0,-89.75)

if __name__=='__main__':unittest.main(verbosity=2)
