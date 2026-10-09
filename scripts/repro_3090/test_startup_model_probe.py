"""Synthetic CPU guard/identity tests; no startup/GPU speed claim."""
import hashlib
from pathlib import Path
import types
import unittest
from unittest import mock

import startup_model_probe as probe
import watch_owned_startup_565 as watch


class StartupProbe(unittest.TestCase):
    def test_default_off_before_import_or_write(self):
        with self.assertRaisesRegex(ValueError,'explicit canonical guard'):
            probe.main(['--skip-eager','1','--out','/not-created.json'])

    def test_strict_same_profile_only_one_mode_changes(self):
        with mock.patch.object(Path,'read_text',return_value='MUSETALK_UNET_BACKEND=trt_stagewise\nMUSETALK_TRT_FALLBACK=0\n'):
            a,b = probe.flags('0'),probe.flags('1')
        self.assertEqual({k for k in a if a[k]!=b[k]},{'MUSETALK_SKIP_EAGER_UNET'})
        self.assertEqual(b['MUSETALK_UNET_STAGEWISE_PROBE_TOL'],'0')
        self.assertEqual(b['AVATAR_S3_ENABLED'],'0')
        with self.assertRaises(ValueError): probe.flags('yes')

    def test_entire_prior_watch_and_one_new_target_bound(self):
        self.assertEqual(hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),watch.CHILD_SHA)
        module = watch.base()
        old = module.reviewed_watch()
        module.configure(old,{'worker_hostname':'owned','gpu_uuid':'GPU-test','deadline_utc':'2099-01-01T00:00:00Z'})
        self.assertEqual(old.CANONICAL_TARGETS['startup_model_probe.py'],watch.CHILD_SHA)
        self.assertEqual(old.QUERY_SECONDS,3)
        self.assertEqual(old.FINISH_SECONDS,10)
        self.assertIn('validate_unet_backend.py',old.CANONICAL_TARGETS)


if __name__=='__main__': unittest.main()
