import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.signal import firwin, lfilter, resample_poly
from modem import *
from session import Source, Reception

class IntegrationTests(unittest.TestCase):
    def test_noisy_stream_all_switches(self):
        rng=np.random.default_rng(291)
        data=rng.integers(0,256,PAYLOAD,dtype=np.uint8).tobytes()
        m=Meta(1,429,0,1,PAYLOAD,16,16,PAYLOAD,b'k'*16)
        for mod in MODS:
            for cp in (False,True):
                for pre in (False,True):
                    cfg=Config(mod,pre,cp)
                    wave=np.r_[np.zeros(87),make_frame(m,data,cfg),np.zeros(900)]
                    # RF/front-end response, phase, CFO and AWGN. Independent stream boundaries.
                    wave=lfilter(firwin(9,.65),[1],wave)
                    wave=.7*wave*np.exp(1j*(.3+2*np.pi*12731*np.arange(len(wave))/cfg.sample_rate))
                    wave+=.006*(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave)))
                    dec=StreamDecoder(cfg)
                    packets=[]
                    for chunk in np.array_split(wave,11):packets+=dec.feed(chunk)
                    with self.subTest(mod=mod,cp=cp,pre=pre):
                        self.assertEqual(len(packets),1)
                        self.assertEqual(packets[0].payload,data)
                        self.assertTrue(packets[0].crc_ok)

    def test_two_frames_and_end(self):
        cfg=Config('QAM-16',True,True)
        source=Source.text('Привет, Pluto!\n'*70)
        frames=[]
        for seq in range(source.total):
            frames.extend([np.zeros(19),make_frame(source.meta(332,seq),source.payload(seq),cfg)])
        frames.append(make_frame(source.meta(332,source.total,True),b'',cfg))
        frames.append(np.zeros(100))
        stream=np.concatenate(frames)
        d=StreamDecoder(cfg)
        packets=[]
        for start in range(0,len(stream),13017):packets+=d.feed(stream[start:start+13017])
        self.assertEqual(len(packets),source.total+1)
        rec=Reception(packets[0].meta,source,cfg.mod)
        for p in packets:rec.accept(p)
        self.assertEqual(bytes(rec.data),source.raw)
        self.assertEqual(rec.stats()['ber'],0)
        self.assertEqual(rec.stats()['per'],0)
        self.assertTrue(rec.ended)

    def test_text_snapshot_grows_only_to_received_data(self):
        source=Source.text('A'*PAYLOAD+'Привет')
        cfg=Config()
        packets=[decode_frame(make_frame(source.meta(19,seq),source.payload(seq),cfg)[len(PREAMBLE):],cfg,0)
                 for seq in range(source.total)]
        rec=Reception(packets[0].meta,source,cfg.mod)
        rec.accept(packets[0])
        first=rec.snapshot()
        self.assertEqual(first['preview_size'],PAYLOAD)
        self.assertEqual(first['data'][:first['preview_size']],source.raw[:PAYLOAD])
        rec.accept(packets[1])
        final=rec.snapshot()
        self.assertEqual(final['preview_size'],len(source.raw))
        self.assertEqual(final['data'][:final['preview_size']],source.raw)

    def test_rgb_errors_losses_reference_and_save(self):
        rgb=np.random.default_rng(1).integers(0,256,(17,19,3),dtype=np.uint8)
        src=Source(1,rgb.tobytes(),19,17)
        self.assertEqual(src.total,4)
        cfg=Config()
        rec=Reception(src.meta(31,0),src,'BPSK')
        for seq in (0,1,3):
            packet=decode_frame(make_frame(src.meta(31,seq),src.payload(seq),cfg)[len(PREAMBLE):],cfg,0)
            if seq==1:
                corrupt=bytearray(packet.payload);corrupt[0]^=1
                packet.payload=bytes(corrupt);packet.crc_ok=False
            rec.accept(packet)
        stat=rec.stats(True)
        self.assertEqual(stat['bit_errors'],1)
        self.assertEqual(stat['compared_bits'],3*768*8)
        self.assertEqual(stat['ber'],1/(3*768*8))
        self.assertEqual(stat['per'],.5)
        self.assertEqual(stat['ber_coverage'],.75)
        arr=np.frombuffer(rec.data,np.uint8).reshape(17,19,3)
        np.testing.assert_array_equal(arr[16,:16],128)
        self.assertEqual(arr[0,16,0],int(rgb[0,16,0])^1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(rec.save(folder,True))
            self.assertTrue((path/'received.png').exists())
            self.assertIn('LOST',(path/'packets.csv').read_text(encoding='utf-8-sig'))
        wrong=Source.text('other')
        other=Reception(src.meta(31,0),wrong,'BPSK')
        self.assertIsNone(other.stats()['ber'])
        self.assertIsNone(other.stats()['snr_estimate_db'])
        self.assertEqual(other.stats()['reference_status'],'тип эталона не совпадает (текст/RGB)')

    def test_reference_status_explains_image_mismatch(self):
        first=Source(1,bytes([0])*3*16*16,16,16)
        changed=Source(1,bytes([1])*3*16*16,16,16)
        reception=Reception(first.meta(8,0),changed,'BPSK')
        self.assertFalse(reception.reference_ok)
        self.assertIn('SHA-256',reception.stats()['reference_status'])
        no_reference=Reception(first.meta(9,0),None,'BPSK')
        self.assertEqual(no_reference.stats()['reference_status'],'эталон не выбран')

    def test_noise_does_not_create_packets(self):
        rng=np.random.default_rng(992)
        dec=StreamDecoder(Config('BPSK',False,False))
        wave=rng.normal(size=100000)+1j*rng.normal(size=100000)
        self.assertEqual(dec.feed(wave),[])
        self.assertLess(len(dec.buffer),dec.length)

if __name__=='__main__':unittest.main(verbosity=2)
