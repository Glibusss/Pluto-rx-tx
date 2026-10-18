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
            cfg = Config(mod, phase_deg=27, ook_low=.2, fsk_deviation_hz=187_500)
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
                         data_symbols=64,payload_bytes=195,phase_deg=23,ook_low=.2,fsk_deviation_hz=187_500)
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

    def test_bpsk_has_no_adjustable_phase_or_inactive_modulation_parameters(self):
        cfg=Config(payload_bytes=192)
        other=replace(cfg,phase_deg=87,ook_low=.7,fsk_deviation_hz=100_000)
        source=Source.text('Fixed BPSK',payload_bytes=cfg.payload_bytes)
        a=make_frame(source.meta(94,0),source.raw,cfg)
        b=make_frame(source.meta(94,0),source.raw,other)
        self.assertEqual(cfg.profile_tag(),other.profile_tag())
        np.testing.assert_array_equal(a,b)
        decoded=decode_frame(a[len(preamble_wave(cfg)):],other,0)
        self.assertTrue(decoded.crc_ok)
        self.assertEqual(decoded.payload,source.raw)

    def test_fsk_deviation_hz_controls_tones_and_decodes_nonorthogonal_signals(self):
        from modulations import fsk
        for rate in (1_000_000,2_000_000):
            for deviation in (50_000.,100_000.,187_500.):
                cfg=Config('2-FSK',sample_rate=rate,payload_bytes=192,
                           fsk_deviation_hz=deviation,pilot_symbols=64)
                cfg.validate()
                known=fsk.wave([-1,1],SPS,cfg.fsk_cycles).reshape(2,SPS)
                actual=np.angle(known[:,1]/known[:,0])*rate/(2*np.pi)
                np.testing.assert_allclose(actual,[-deviation,deviation],atol=1e-6)
                source=Source.text('FSK frequency in Hz '*10,payload_bytes=192)
                wave=make_frame(source.meta(95,0),source.payload(0),cfg)
                wave *= .8*np.exp(1j*(.4+2*np.pi*13_711*np.arange(len(wave))/rate))
                packets=StreamDecoder(cfg).feed(np.r_[wave,np.zeros(100)])
                with self.subTest(rate=rate,deviation=deviation):
                    self.assertEqual(len(packets),1)
                    packet=packets[0]
                    self.assertTrue(packet.crc_ok)
                    self.assertEqual(packet.payload,source.payload(0))
                    self.assertLess(packet.crc_error_energy/packet.crc_signal_energy,1e-8)
                    rec=Reception(packet.meta,source,cfg.mod,{'cfg':asdict(cfg)})
                    rec.accept(packet)
                    self.assertEqual(rec.stats()['bit_errors'],0)
                    self.assertLess(rec.error_sum/rec.signal_sum,1e-8)
                other=replace(cfg,fsk_deviation_hz=deviation+1000)
                self.assertNotEqual(cfg.profile_tag(),other.profile_tag())

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

    def test_inactive_preamble_and_cp_do_not_change_frame_or_decoding(self):
        from itertools import product
        for mod,preamble,cp in product(MODS,(False,True),(False,True)):
            cfg=Config(mod,preamble,cp,payload_bytes=192)
            changes={}
            if not preamble:
                changes.update(preamble_symbols=512,preamble_seed=842)
            if not cp:
                changes['cp_symbols']=4
            receiver_cfg=replace(cfg,**changes)
            source=Source.text('inactive fields',payload_bytes=192)
            frame=make_frame(source.meta(83,0),source.payload(0),cfg)
            other=make_frame(source.meta(83,0),source.payload(0),receiver_cfg)
            with self.subTest(mod=mod,preamble=preamble,cp=cp):
                self.assertEqual(cfg.profile_tag(),receiver_cfg.profile_tag())
                np.testing.assert_array_equal(frame,other)
                decoded=decode_frame(frame[len(preamble_wave(cfg)):],receiver_cfg,0)
                self.assertTrue(decoded.crc_ok)
                self.assertEqual(decoded.payload,source.raw)

    def test_active_service_parameters_and_switches_change_profile(self):
        cfg=Config(cp=True)
        changes=({'cp_symbols':4},{'preamble_symbols':512},{'preamble_seed':842},
                 {'preamble':False},{'cp':False})
        source=Source.text('active fields')
        raw=source.meta(84,0).encode(cfg)
        for changed in changes:
            other=replace(cfg,**changed)
            with self.subTest(changed=changed):
                self.assertNotEqual(cfg.profile_tag(),other.profile_tag())
                with self.assertRaises(ValueError):
                    Meta.decode(raw,other)

    def test_invalid_profiles_are_rejected(self):
        for change in ({'pilot_symbols':0},{'preamble_symbols':20},{'data_symbols':0},
                       {'payload_bytes':65536},{'cp_symbols':200,'data_symbols':32},
                       {'mod':'QPSK','phase_deg':float('nan')},{'mod':'ASK / OOK','ook_low':1},
                       {'mod':'2-FSK','fsk_deviation_hz':0},
                       {'mod':'2-FSK','fsk_deviation_hz':250_000},
                       {'mod':'2-FSK','fsk_deviation_hz':float('nan')}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                Config(**change).validate()

    def test_hackrf_uses_the_same_custom_profile_at_both_phy_rates(self):
        for rate in (1000000,2000000):
            for mod in MODS:
                profile=Config(mod,True,True,rate,0,preamble_symbols=80,cp_symbols=8,
                               pilot_symbols=16,data_symbols=64,payload_bytes=192,
                               phase_deg=13,ook_low=.2,fsk_deviation_hz=rate*1.5/SPS)
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
