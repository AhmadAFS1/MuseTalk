"""Fresh Vast workers must deploy immutable commit pins, without network/rental fixtures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BOOTSTRAP = Path(__file__).resolve().parents[1] / 'orchestrator/template/onstart_cmd.sh'


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = self.root / 'origin'
        self.workspace = self.root / 'worker workspace'
        self.origin.mkdir()
        self.env = {**os.environ, 'WORKSPACE': str(self.workspace), 'MUSETALK_REPO_URL': str(self.origin),
                    'GIT_CONFIG_GLOBAL': str(self.root / 'gitconfig'), 'GIT_CONFIG_NOSYSTEM': '1',
                    'GIT_AUTHOR_NAME': 'fixture', 'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
                    'GIT_COMMITTER_NAME': 'fixture', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid'}
        self.git('init', '-q', '-b', 'main')
        handoff = self.origin / 'character_factory/character_worker/vast_onstart_character.sh'
        handoff.parent.mkdir(parents=True)
        handoff.write_text('printf "handoff" > "$WORKSPACE/handoff.txt"\n')
        self.git('add', '.')
        self.git('commit', '-q', '-m', 'base')
        self.base = self.git('rev-parse', 'HEAD')
        self.git('checkout', '-q', '-b', 'reviewed-feature')
        (self.origin / 'revision.txt').write_text('reviewed\n')
        self.git('add', '.')
        self.git('commit', '-q', '-m', 'reviewed')
        self.reviewed = self.git('rev-parse', 'HEAD')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.origin), *args], env=self.env,
                                       stderr=subprocess.DEVNULL, text=True).strip()

    def boot(self, ref):
        return subprocess.run(['bash', str(BOOTSTRAP)], env={**self.env, 'MUSETALK_REF': ref},
                              capture_output=True, text=True, timeout=20)

    def assert_deployed(self, ref):
        head = subprocess.check_output(['git', '-C', str(self.workspace / 'MuseTalk'), 'rev-parse', 'HEAD'],
                                       env=self.env, text=True).strip()
        self.assertEqual(head, ref)
        self.assertEqual((self.workspace / 'handoff.txt').read_text(), 'handoff')
        self.assertFalse(list(self.workspace.glob('.character-checkout.*')))

    def test_fresh_immutable_sha_pin_and_existing_checkout_update(self):
        first = self.boot(self.base)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assert_deployed(self.base)
        second = self.boot(self.reviewed)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assert_deployed(self.reviewed)
        failed = self.boot('missing-ref')
        self.assertNotEqual(failed.returncode, 0)
        self.assert_deployed(self.reviewed)

    def test_branch_ref_remains_supported(self):
        result = self.boot('reviewed-feature')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_deployed(self.reviewed)

    def test_invalid_pin_leaves_no_checkout_and_non_git_data_is_preserved(self):
        bad = self.boot('missing-ref')
        self.assertNotEqual(bad.returncode, 0)
        self.assertFalse((self.workspace / 'MuseTalk').exists())
        self.assertFalse(list(self.workspace.glob('.character-checkout.*')))
        target = self.workspace / 'MuseTalk'
        target.mkdir()
        (target / 'keep.txt').write_text('keep\n')
        result = self.boot(self.reviewed)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((target / 'keep.txt').read_text(), 'keep\n')


if __name__ == '__main__':
    unittest.main()
