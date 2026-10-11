"""Execute the real on-start validation with synthetic model fixtures, no GPU."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class StartupModelContractTests(unittest.TestCase):
    def validate(self, missing=None):
        tree = ast.parse((ROOT / 'scripts/musetalk_install_state.py').read_text())
        models = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id in {'SERVER_MODEL_FILES', 'AVATAR_PREP_MODEL_FILES'}
                    for target in node.targets):
                models.extend(ast.literal_eval(node.value))
        self.assertTrue(models)
        source = (ROOT / 'scripts/vast_onstart.sh').read_text()
        function = source[source.index('run_post_setup_validation() {'):
                          source.index('\nimmutable_runtime_policy() {')]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'venv/bin').mkdir(parents=True)
            (root / 'venv/bin/python').symlink_to(sys.executable)
            for name in models:
                if name != missing:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'SYNTHETIC TEST MODEL, NOT USABLE')
            script = '''set -Eeuo pipefail
REPO_ROOT="$1"
VENV_PATH="$1/venv"
IMMUTABLE_BOOT=1
ONSTART_POST_VALIDATE_IMPORTS=0
log() { printf '%s\\n' "$*"; }
die() { log "$*"; exit 7; }
legacy_recipe_selected() { return 1; }
avatar_prep_requested() { return 0; }
env_flag_is_true() { [[ "$1" == 1 ]]; }
'''
            return subprocess.run(['bash', '-c', script + function + '\nrun_post_setup_validation',
                                   'contract-test', str(root)], capture_output=True, text=True, timeout=10)

    def test_full_serving_preparation_contract_passes_without_training_syncnet(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('All post-setup validation checks passed', result.stdout)

    def test_missing_required_preparation_or_server_model_still_fails_immutable_boot(self):
        for name in ('models/dwpose/dw-ll_ucoco_384.pth', 'models/face_detection/s3fd.pth',
                     'models/musetalkV15/unet.pth', 'models/taesd/diffusion_pytorch_model.safetensors'):
            with self.subTest(name=name):
                result = self.validate(name)
                self.assertEqual(result.returncode, 7)
                self.assertIn('MISSING: ' + name, result.stdout)
                self.assertIn('Immutable image post-setup validation failed', result.stdout)


if __name__ == '__main__':
    unittest.main()
