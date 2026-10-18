"""Offline captured-evidence audit fixtures only; no pip, GPU or network."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
from verify_dependency_optimization_evidence import verify

REPO=Path(__file__).resolve().parent
EVIDENCE=REPO/'docs/dependency_optimization_20261015'


@contextlib.contextmanager
def copied_fixture():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);dest=root/'docs/evidence';dest.mkdir(parents=True)
        for name in ('results.json','canary-results.json','final-screen-results.json','wheelhouse-manifest-a2.json'):
            (dest/name).write_bytes((EVIDENCE/name).read_bytes())
        path=root/'configs/dependencies/rtx3090-cp310-cu121-screening.json';path.parent.mkdir(parents=True)
        path.write_bytes((REPO/'configs/dependencies'/path.name).read_bytes())
        yield dest


class EvidenceAuditTests(unittest.TestCase):
    def test_current_captured_evidence_consistent(self):
        value=verify(EVIDENCE)
        self.assertEqual(value['state'],'captured-evidence-consistent')
        self.assertEqual(value['controlled_identical_videos'],28)
        self.assertEqual(value['owned_resources_remaining'],0)
        self.assertGreater(value['winner_reduction_percent'],55)

    def test_failed_s3_cannot_be_silently_relabelled(self):
        with copied_fixture() as folder:
            p=folder/'canary-results.json';value=json.loads(p.read_text())
            value['extra_test_receipts']['s3-full-a1']['state']='passed'
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'failure was lost'):
                verify(folder)

    def test_missing_cleanup_cannot_pass(self):
        with copied_fixture() as folder:
            p=folder/'final-screen-results.json';value=json.loads(p.read_text())
            value['cleanup']['provider_absence_verified']=False;p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'cleanup missing'):
                verify(folder)


if __name__=='__main__':unittest.main()
