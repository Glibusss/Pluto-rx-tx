import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import radio
from modem import Config, StreamDecoder, PREAMBLE, decode_frame, make_frame
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

class FakeRX:
    def __init__(self):
        self.rx_lo=2399750000
        self.destroyed=False
    def rx(self):return np.zeros(64,np.complex64)
    def rx_destroy_buffer(self):self.destroyed=True

class NoWait:
    def is_set(self):return False
    def wait(self,duration):return False

class StopAfterWaits:
    def __init__(self,count):self.count=count;self.waits=0
    def is_set(self):return self.waits>=self.count
    def wait(self,duration):self.waits+=1;return self.is_set()

class RadioTests(unittest.TestCase):
    def test_noise_power_and_automatic_threshold(self):
        self.assertAlmostEqual(radio.power_dbfs(np.full(32,radio.ADC_FULL_SCALE)),0)
        result=radio.noise_threshold([-81,-80,-80,-79.5,-80.5])
        self.assertAlmostEqual(result['noise_dbfs'],-80)
        self.assertGreaterEqual(result['threshold_dbfs'],result['noise_dbfs']+3)

    def test_noise_calibration_reads_pluto_and_cleans_up(self):
        sdr=FakeRX()
        sdr.sample_rate=1_000_000
        sdr.rx_buffer_size=32768
        with patch.object(radio,'connect',return_value=sdr):
            result=radio.calibrate_noise(dict(cfg=Config()),threading.Event(),duration=.01)
        self.assertEqual(result['buffers'],8)
        self.assertTrue(sdr.destroyed)

    def test_rx_queue_capacity_is_configurable_and_bounded(self):
        self.assertEqual(radio.rx_queue_capacity({}),512)
        self.assertEqual(radio.rx_queue_capacity({'queue_buffers':'8192'}),8192)
        for value in (15,8193,'bad'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                radio.rx_queue_capacity({'queue_buffers':value})

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

    def test_rx_waits_for_packet_zero_and_stops_on_end(self):
        cfg=Config()
        source=Source.text('Hello RX')
        tid=72
        skipped=decode_frame(make_frame(source.meta(tid,source.total,True),b'',cfg)[len(PREAMBLE):],cfg,0)
        data=decode_frame(make_frame(source.meta(tid+1,0),source.payload(0),cfg)[len(PREAMBLE):],cfg,0)
        end=decode_frame(make_frame(source.meta(tid+1,source.total,True),b'',cfg)[len(PREAMBLE):],cfg,0)
        decoder=unittest.mock.Mock()
        decoder.candidates=decoder.header_failures=0
        decoder.feed.side_effect=[[skipped,data,end]]
        stop=threading.Event();events=[];sdr=FakeRX()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(radio,'connect',return_value=sdr),patch.object(radio,'StreamDecoder',return_value=decoder):
                radio.receive(dict(cfg=cfg),None,folder,stop,lambda *x:events.append(x))
            saved=list(Path(folder).glob('transfer_*'))
        self.assertTrue(stop.is_set())
        self.assertTrue(sdr.destroyed)
        self.assertEqual(len(saved),1)
        self.assertTrue(any(kind=='log' and 'RX останавливается' in value for kind,value in events))

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
