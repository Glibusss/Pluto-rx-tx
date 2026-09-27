import unittest
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import correlate
from modem import *

class ModemTests(unittest.TestCase):
    def test_cached_search_matches_reference_correlator(self):
        rng=np.random.default_rng(649)
        for preamble in (False,True):
            decoder=StreamDecoder(Config('BPSK',preamble,False))
            for positions in (1,17,1024,32768):
                segment=(rng.normal(size=positions+len(decoder.short)-1)+
                         1j*rng.normal(size=positions+len(decoder.short)-1)).astype(np.complex64)
                size=next_fast_len(len(segment)+len(decoder.short)-1)
                spectrum=fft(segment,size)
                for freq in (decoder.grid[0],0.,decoder.grid[-1],12345.):
                    template=decoder.short*np.exp(2j*np.pi*freq*np.arange(len(decoder.short))/decoder.cfg.sample_rate)
                    expected=correlate(segment,template,mode='valid',method='fft')
                    actual=ifft(spectrum*decoder._template_fft(freq,size))[
                        len(decoder.short)-1:len(decoder.short)-1+positions]
                    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-11)

    def test_reset_reacquires_different_cfo_and_stream_alignment(self):
        cfg=Config('QAM-16',True,True)
        data=np.random.default_rng(403).integers(0,256,PAYLOAD,dtype=np.uint8).tobytes()
        decoder=StreamDecoder(cfg)
        for transfer,cfo in ((31,-33000),(32,37000)):
            decoder.reset()
            meta=Meta(1,transfer,0,1,PAYLOAD,16,16,PAYLOAD,b'q'*16)
            wave=np.r_[np.zeros(307),make_frame(meta,data,cfg),np.zeros(19)]
            wave=wave*np.exp(2j*np.pi*cfo*np.arange(len(wave))/cfg.sample_rate)
            packets=[]
            for chunk in np.array_split(wave,7):
                packets+=decoder.feed(chunk)
            self.assertEqual(len(packets),1)
            self.assertEqual(packets[0].meta.transfer,transfer)
            self.assertEqual(packets[0].payload,data)
            self.assertTrue(packets[0].crc_ok)

    def test_silence_keeps_overlap_and_does_not_hide_next_packet(self):
        cfg=Config('QAM-64',True,False)
        meta=Meta(1,47,0,1,PAYLOAD,16,16,PAYLOAD,b's'*16)
        data=bytes([92])*PAYLOAD
        decoder=StreamDecoder(cfg)
        for _ in range(4):
            self.assertEqual(decoder.feed(np.zeros(32768,np.complex64)),[])
        self.assertLess(len(decoder.buffer),decoder.length)
        packets=[]
        for chunk in np.array_split(make_frame(meta,data,cfg),9):
            packets+=decoder.feed(chunk)
        self.assertEqual(len(packets),1)
        self.assertEqual(packets[0].payload,data)
        self.assertTrue(packets[0].crc_ok)

    def test_decision_error_energies(self):
        signal,error=decision_error_energies(np.array([-1+.1j,.8+0j]),'BPSK')
        self.assertAlmostEqual(signal,2.)
        self.assertAlmostEqual(error,.05)
        signal,error=decision_error_energies(np.array([[.9,.1],[.2,.8]]),'2-FSK')
        self.assertAlmostEqual(signal,2.)
        self.assertAlmostEqual(error,.1)

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
                        self.assertEqual(r.sir_vector_count,PAYLOAD*8//constellation(mod)[1])
                        self.assertGreater(r.sir_signal_energy,0)
                        self.assertGreaterEqual(r.sir_interference_energy,0)
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
