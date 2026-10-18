import unittest
import math
import numpy as np
from dataclasses import asdict
from quality import pilot_energies, ratio_db, evm_percent
from modem import (Config, make_frame, decode_frame, preamble_wave, sync_wave,
                   header_blocks, block_len)
from modulations import MODS, SPS
from session import Source, Reception


class QualityTests(unittest.TestCase):
    def test_finite_window_powers_track_known_awgn_without_decisions(self):
        rng=np.random.default_rng(478)
        for fsk in (False,True):
            signal=disturbance=0.
            n=32
            gain=.7+.3j
            sigma=.12
            for _ in range(1500):
                labels=rng.integers(0,2,n)
                known=np.column_stack((labels==0,labels==1)).astype(complex) if fsk else (2*labels-1)
                noise=sigma*(rng.normal(size=known.shape)+1j*rng.normal(size=known.shape))
                a,b,count=pilot_energies(gain*known+noise,known)
                signal+=a
                disturbance+=b
            expected_signal=1500*n*abs(gain)**2
            expected_error=1500*n*(2 if fsk else 1)*2*sigma**2
            self.assertAlmostEqual(signal/expected_signal,1,delta=.015)
            self.assertAlmostEqual(disturbance/expected_error,1,delta=.015)
            self.assertAlmostEqual(ratio_db(signal,disturbance,1),
                                   10*math.log10(expected_signal/expected_error),delta=.15)

    def test_no_reference_needed_and_estimate_degrades_with_noise(self):
        rng=np.random.default_rng(94)
        for mod in MODS:
            cfg=Config(mod,False,False,payload_bytes=192)
            source=Source.text('A'*192,payload_bytes=192)
            values=[]
            for sigma in (.01,.06):
                wave=make_frame(source.meta(4,0),source.payload(0),cfg)
                wave=wave+sigma*(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave)))
                packet=decode_frame(wave,cfg,0)
                rec=Reception(packet.meta,None,mod,{'cfg':asdict(cfg)})
                rec.accept(packet)
                stats=rec.stats()
                with self.subTest(mod=mod,sigma=sigma):
                    self.assertIsNone(stats['ber'])
                    self.assertIsNone(stats['corrupted_bits'])
                    self.assertIsNotNone(stats['sinr_estimate_db'])
                    self.assertGreater(stats['pilot_sample_count'],0)
                    self.assertEqual(stats['crc_payload_evm_percent'] is not None, packet.crc_ok)
                values.append(stats['sinr_estimate_db'])
            self.assertGreater(values[0]-values[1],10)

    def test_payload_only_burst_cannot_be_hidden_by_valid_pilots(self):
        cfg=Config('BPSK',False,False,payload_bytes=192,data_symbols=64)
        source=Source.text('A'*192,payload_bytes=192)
        wave=make_frame(source.meta(5,0),source.payload(0),cfg)
        body_start=len(sync_wave(cfg))+header_blocks(cfg)*block_len(cfg)
        start=body_start+cfg.pilot_symbols*SPS+5*SPS
        wave[start:start+SPS]*=-1
        packet=decode_frame(wave,cfg,0)
        self.assertFalse(packet.crc_ok)
        self.assertEqual(packet.crc_symbol_count,0)
        rec=Reception(packet.meta,None,cfg.mod)
        rec.accept(packet)
        stats=rec.stats()
        self.assertGreater(stats['sinr_estimate_db'],40)
        self.assertIsNone(stats['crc_payload_evm_percent'])
        self.assertEqual(stats['crc_quality_packet_coverage'],0)
        self.assertIn('между пилотами',stats['quality_warning'])
        self.assertEqual(stats['crc_packets'],1)

    def test_receiver_estimates_do_not_depend_on_reference_or_duplicates(self):
        cfg=Config('QAM-16',False)
        source=Source.text('independent reference')
        packet=decode_frame(make_frame(source.meta(6,0),source.payload(0),cfg),cfg,0)
        results=[]
        for reference in (None,source,Source.text('different')):
            rec=Reception(packet.meta,reference,cfg.mod)
            rec.accept(packet)
            before=rec.stats()
            rec.accept(packet)
            after=rec.stats()
            self.assertEqual(before['pilot_sample_count'],after['pilot_sample_count'])
            self.assertEqual(after['duplicates'],1)
            results.append((after['sinr_estimate_db'],after['crc_payload_evm_percent']))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[1],results[2])

    def test_unavailable_quality_and_zero_noise_are_finite(self):
        self.assertIsNone(ratio_db(0,0,0))
        self.assertIsNone(evm_percent(0,0,0))
        self.assertEqual(pilot_energies(np.zeros(8),np.zeros(8)),(0.,0.,0))
        self.assertEqual(pilot_energies([np.nan]*8,np.ones(8)),(0.,0.,0))
        self.assertTrue(math.isfinite(ratio_db(1,0,32)))
        self.assertEqual(evm_percent(1,0,32),0)


if __name__ == '__main__':
    unittest.main()
