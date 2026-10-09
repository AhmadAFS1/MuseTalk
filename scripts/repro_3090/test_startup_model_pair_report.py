import copy
import unittest
import startup_model_pair_report as report


class PairReport(unittest.TestCase):
    def fixture(self):
        trials=[]
        for mode in report.SEQUENCE:
            trials.append(dict(status='PASS_DIAGNOSTIC',schema='owned_fresh_process_startup_model_probe_v1',
                skip_eager_unet=mode,source_pins={'source':'a'*64},input_capture_sha256='b'*64,
                model_startup_profile={'skip_eager_unet':mode=='1','avatar_vae_encoder_retained':True},
                imports_s=1.,manager_init_s=9. if mode=='0' else 3.,import_init_wall_s=10. if mode=='0' else 4.,
                peak_allocated_vram_bytes=123,**{k:dict(shape=v,dtype='torch.float16',nonfinite_count=0,sha256='c'*64) for k,v in report.SHAPES.items()}))
        watch=[dict(status='ENROLLMENT_PENDING',target_sha256='d'*64),
               *[dict(status=x) for x in sorted(report.ACKS)],dict(status='COMPLETE',returncode=0)]
        return trials,[copy.deepcopy(watch) for _ in range(4)]

    def test_exact_four_trial_pair(self):
        result=report.summarize(*self.fixture())
        self.assertEqual(result['mean_manager_init_saving_s'],6.)
        self.assertFalse(result['release_ready'] or result['startup_acceptance'])

    def test_wrong_sequence_missing_ack_failed_child(self):
        for case in ('sequence','ack','child','missing'):
            trials,watches=self.fixture()
            if case=='sequence': trials[0]['skip_eager_unet']='1'
            elif case=='ack': watches[0].pop(1)
            elif case=='child': watches[0][-1]['returncode']=1
            else: trials.pop()
            with self.assertRaises(ValueError): report.summarize(trials,watches)

    def test_nonfinite_timing_or_output_and_different_input(self):
        for field,value in (('manager_init_s',float('nan')),('input_capture_sha256','f'*64),('decoded',{'nonfinite_count':1})):
            trials,watches=self.fixture();trials[1][field]=value
            with self.assertRaises(ValueError): report.summarize(trials,watches)


if __name__=='__main__': unittest.main()
