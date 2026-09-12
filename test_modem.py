import unittest
import numpy as np
from modem import *

class ModemTests(unittest.TestCase):
    def test_supported_qam_modes(self):
        self.assertNotIn('QAM-8',MODS)
        self.assertEqual([mode for mode in MODS if mode.startswith('QAM-')],
                         ['QAM-4','QAM-16','QAM-64'])
        points,bits_per_symbol=constellation('QAM-64')
        self.assertEqual((len(points),bits_per_symbol),(64,6))
        self.assertAlmostEqual(float(np.mean(abs(points)**2)),1)

    def test_all_modes(self):
        rng = np.random.default_rng(881)
        payload = rng.integers(0,256,PAYLOAD,dtype=np.uint8).tobytes()
        for mod in MODS:
            for cp in (False, True):
                for pre in (False, True):
                    cfg = Config(mod, pre, cp)
                    m = Meta(1, 999, 0, 1, PAYLOAD, 16, 16, PAYLOAD, b'x'*16)
                    wave = make_frame(m, payload, cfg)[len(PREAMBLE) if pre else 0:]
                    r = decode_frame(wave, cfg, 0)
                    with self.subTest(mod=mod,cp=cp,pre=pre):
                        self.assertEqual(r.payload, payload)
                        self.assertTrue(r.crc_ok)
    def test_stream_offsets_cfo_noise(self):
        rng = np.random.default_rng(4)
        for mod in MODS:
            cfg = Config(mod, False, False)
            data = rng.integers(0,256,PAYLOAD,dtype=np.uint8).tobytes()
            m = Meta(1, 17, 0, 1, PAYLOAD, 16, 16, PAYLOAD, b'k'*16)
            wave = np.r_[np.zeros(137),make_frame(m,data,cfg),np.zeros(1000)]
            wave = .7*wave*np.exp(1j*(.9+2*np.pi*17321*np.arange(len(wave))/cfg.sample_rate))
            wave += .012*(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave)))
            dec = StreamDecoder(cfg)
            packets=[]
            for chunk in np.array_split(wave, 19):
                packets += dec.feed(chunk)
            with self.subTest(mod=mod):
                self.assertEqual(len(packets),1)
                self.assertEqual(packets[0].payload,data)
                self.assertTrue(packets[0].crc_ok)

    def test_stream_near_cfo_search_edges(self):
        rng=np.random.default_rng(71)
        data=rng.integers(0,256,PAYLOAD,dtype=np.uint8).tobytes()
        for rate in (1_000_000,2_000_000):
            for cfo in (-39_000,39_000):
                cfg=Config('BPSK',True,False,rate,40_000)
                m=Meta(1,83,0,1,PAYLOAD,16,16,PAYLOAD,b'z'*16)
                wave=np.r_[np.zeros(211),make_frame(m,data,cfg),np.zeros(900)]
                wave=.8*wave*np.exp(1j*(.4+2*np.pi*cfo*np.arange(len(wave))/rate))
                wave+=.008*(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave)))
                decoder=StreamDecoder(cfg)
                packets=[]
                for chunk in np.array_split(wave,13):packets+=decoder.feed(chunk)
                with self.subTest(rate=rate,cfo=cfo):
                    self.assertEqual(len(packets),1)
                    self.assertEqual(packets[0].payload,data)
                    self.assertTrue(packets[0].crc_ok)

if __name__ == '__main__':
    unittest.main(verbosity=2)
