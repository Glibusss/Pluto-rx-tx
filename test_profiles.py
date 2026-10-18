import unittest
from dataclasses import asdict, replace
from unittest.mock import patch
import numpy as np
from modem import (Config, Meta, make_frame, decode_frame, StreamDecoder, frame_len,
                   preamble_wave, sync_wave, pilot_wave, tile_shape)
from modulations import MODS, SPS, constellation
from session import Source, Reception
from hackrf_emitter import EmitterConfig, packet_source, _packet_template


class ProfileTests(unittest.TestCase):
    def test_selected_modulation_carries_every_service_section(self):
        for mod in MODS:
            cfg = Config(mod, phase_deg=27, ook_low=.2, fsk_cycles=1.5)
            for known in (preamble_wave(cfg), sync_wave(cfg), pilot_wave(cfg)):
                with self.subTest(mod=mod, samples=len(known)):
                    if mod == '2-FSK':
                        ratios = known.reshape(-1,SPS)[:,1] / known.reshape(-1,SPS)[:,0]
                        expected=np.exp(2j*np.pi*cfg.fsk_cycles*np.array([-1,1])/SPS)
                        self.assertLess(float(np.min(abs(ratios[:,None]-expected),axis=1).max()),1e-6)
                    else:
                        points = constellation(mod,cfg)[0]
                        z = known.reshape(-1,SPS)[:,0]
                        self.assertLess(float(np.min(abs(z[:,None]-points),axis=1).max()),1e-6)
                        if mod in ('QPSK','8-PSK','QAM-4','QAM-16','QAM-64'):
                            self.assertGreater(float(np.max(abs(z.imag))),.1)

    def test_nondefault_profile_decodes_stream_and_end_for_all_modes(self):
        for mod in MODS:
            cfg = Config(mod,True,True,cfo_range=25000,preamble_symbols=96,
                         preamble_seed=478,cp_symbols=8,pilot_symbols=24,pilot_seed=63,
                         data_symbols=64,payload_bytes=195,phase_deg=23,ook_low=.2,fsk_cycles=1.5)
            source = Source.text('Профиль '+mod+' АБВ'*45,payload_bytes=cfg.payload_bytes)
            frames = [make_frame(source.meta(148,i),source.payload(i),cfg) for i in range(source.total)]
            frames += [make_frame(source.meta(148,source.total,True),b'',cfg)]
            self.assertTrue(all(len(frame)==frame_len(cfg)+len(preamble_wave(cfg)) for frame in frames))
            wave = np.r_[np.zeros(37),*frames,np.zeros(300)]
            wave *= .7*np.exp(1j*(.3+2*np.pi*13721*np.arange(len(wave))/cfg.sample_rate))
            decoder = StreamDecoder(cfg)
            packets = []
            for chunk in np.array_split(wave,17):
                packets += decoder.feed(chunk)
            with self.subTest(mod=mod):
                self.assertEqual(len(packets),source.total+1)
                self.assertTrue(all(packet.crc_ok for packet in packets))
                result = Reception(packets[0].meta,source,mod,{'cfg':asdict(cfg)})
                for packet in packets:
                    result.accept(packet)
                self.assertEqual(bytes(result.data),source.raw)
                self.assertEqual(result.stats()['bit_errors'],0)
                self.assertTrue(result.ended)

    def test_image_packet_size_and_exact_lost_edge_pixels(self):
        for size in (65,192,768,1536):
            cfg = Config(payload_bytes=size)
            source = Source(1,bytes([33])*19*17*3,19,17,payload_bytes=size)
            result = Reception(source.meta(72,0),source,'BPSK')
            expected_received = 0
            tw,th = tile_shape(size)
            cols = (19+tw-1)//tw
            for seq in range(1,source.total):
                wave=make_frame(source.meta(72,seq),source.payload(seq),cfg)[len(preamble_wave(cfg)):]
                packet=decode_frame(wave,cfg,0)
                self.assertTrue(packet.crc_ok)
                result.accept(packet)
                expected_received += min(tw,19-(seq%cols)*tw)*min(th,17-(seq//cols)*th)
            stats=result.stats(True)
            self.assertEqual(stats['lost_packets'],1)
            self.assertEqual(stats['lost_pixels'],19*17-expected_received)
            self.assertEqual(stats['corrupted_bits'],0)
            self.assertEqual(stats['corrupted_pixels'],0)

    def test_profile_mismatch_is_rejected_by_header(self):
        cfg=Config()
        source=Source.text('profile')
        raw=source.meta(42,0).encode(cfg)
        for changed in (replace(cfg,pilot_seed=12),replace(cfg,payload_bytes=1536),
                        replace(cfg,preamble_symbols=512),replace(cfg,data_symbols=256)):
            with self.assertRaises(ValueError):
                Meta.decode(raw,changed)

    def test_unrelated_large_reference_does_not_break_small_packet_reception(self):
        source=Source.text('small',payload_bytes=48)
        unrelated=Source.text('x'*100000)
        reception=Reception(source.meta(43,0),unrelated,'BPSK')
        self.assertFalse(reception.reference_ok)
        self.assertIn('размер',reception.reference_status)

    def test_invalid_profiles_are_rejected(self):
        for change in ({'pilot_symbols':0},{'preamble_symbols':20},{'data_symbols':0},
                       {'payload_bytes':65536},{'cp_symbols':200,'data_symbols':32},
                       {'phase_deg':float('nan')},{'ook_low':1},{'fsk_cycles':4}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                Config(**change).validate()

    def test_hackrf_uses_the_same_custom_profile_at_both_phy_rates(self):
        for rate in (1000000,2000000):
            for mod in MODS:
                profile=Config(mod,True,True,rate,0,preamble_symbols=80,cp_symbols=8,
                               pilot_symbols=16,data_symbols=64,payload_bytes=192,
                               phase_deg=13,ook_low=.2,fsk_cycles=1.5)
                cfg=EmitterConfig(sample_rate=rate*2,modulation=mod,packet_kind='image',packet_config=profile)
                cfg.validate()
                source=packet_source(cfg)
                emitted=_packet_template(cfg,source,45)
                from scipy.signal import resample_poly
                received=resample_poly(emitted,1,2)
                packets=StreamDecoder(profile).feed(np.r_[received,np.zeros(80)])
                with self.subTest(rate=rate,mod=mod):
                    self.assertEqual(len(packets),1)
                    self.assertTrue(packets[0].crc_ok)
                    self.assertEqual(packets[0].payload,source.raw)


if __name__ == '__main__':
    unittest.main()
