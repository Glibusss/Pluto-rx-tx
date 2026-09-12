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

class StopAfterWaits:
    def __init__(self,count):self.count=count;self.waits=0
    def is_set(self):return self.waits>=self.count
    def wait(self,duration):self.waits+=1;return self.is_set()

class RadioTests(unittest.TestCase):
    def test_tx_repeats_with_new_transfer_id_until_stop(self):
        sdr=FakeTX();source=Source.text('Hello Pluto')
        settings=dict(cfg=Config(),gap_ms=0)
        events=[]
        with patch.object(radio,'connect',return_value=sdr),patch.object(radio.time,'sleep'):
            radio.transmit(settings,source,StopAfterWaits(8),lambda *x:events.append(x))
        self.assertEqual(len(sdr.calls),9) # 2 × (1 data + 3 END) + final zero buffer
        self.assertTrue(sdr.destroyed)
        self.assertEqual(sdr.tx_hardwaregain_chan0,-89.75)
        self.assertTrue(np.all(sdr.calls[-1]==0))
        decoder=StreamDecoder(settings['cfg'])
        packets=[]
        for iq in sdr.calls:
            iq=iq/4096*np.exp(-2j*np.pi*.25*np.arange(len(iq)))
            packets+=decoder.feed(iq)
        self.assertEqual(len(packets),8)
        data_packets=[p for p in packets if not p.meta.end]
        self.assertEqual([p.payload for p in data_packets],[source.raw,source.raw])
        self.assertNotEqual(data_packets[0].meta.transfer,data_packets[1].meta.transfer)
        self.assertEqual(sum(p.meta.end for p in packets),6)
        self.assertIn(('progress',(0,source.total,2)),events)

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
