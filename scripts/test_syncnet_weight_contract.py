"""CPU-only checks: API model contract and actual installer flag forwarding."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SyncNetContractTests(unittest.TestCase):
    def test_training_checkpoint_is_not_an_api_or_preparation_requirement(self):
        tree = ast.parse((ROOT / "scripts/musetalk_install_state.py").read_text())
        names = {"SERVER_MODEL_FILES", "AVATAR_PREP_MODEL_FILES", "TRAINING_MODEL_FILES"}
        lists = {target.id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name) and target.id in names}
        api = lists["SERVER_MODEL_FILES"] + lists["AVATAR_PREP_MODEL_FILES"]
        self.assertEqual(len(api), 13)
        self.assertEqual(len(set(api)), 13)
        self.assertNotIn("models/syncnet/latentsync_syncnet.pt", api)
        self.assertEqual(lists["TRAINING_MODEL_FILES"], ["models/syncnet/latentsync_syncnet.pt"])
        self.assertEqual(lists["AVATAR_PREP_MODEL_FILES"],
                         ["models/dwpose/dw-ll_ucoco_384.pth", "models/face_detection/s3fd.pth"])

    def test_actual_installer_function_forwards_default_and_explicit_flags(self):
        source = (ROOT / "scripts/install_musetalk.sh").read_text()
        function = "download_weights_step() {" + source.split("download_weights_step() {", 1)[1].split(
            "\nphase_weights() {", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "download_weights.sh").write_text(
                "printf '%s\\n' \"$DOWNLOAD_SYNCNET_WEIGHTS\" \"$DOWNLOAD_AVATAR_PREP_WEIGHTS\" "
                "\"$DOWNLOAD_MUSETALK_V1_WEIGHTS\" \"$DOWNLOAD_TAESD_WEIGHTS\" \"$DOWNLOAD_KOKORO_WEIGHTS\"\n")
            for explicit in (None, "1", "0", "true"):
                env = dict(os.environ, REPO_ROOT=tmp, VENV_PATH=str(root / "venv"),
                           AVATAR_PREP="1", KOKORO="1", CONSTRAINTS="fixture-constraints")
                env.pop("DOWNLOAD_SYNCNET_WEIGHTS", None)
                if explicit is not None:
                    env["DOWNLOAD_SYNCNET_WEIGHTS"] = explicit
                with self.subTest(explicit=explicit):
                    result = subprocess.run(["bash", "-c", function + "\ndownload_weights_step"],
                                            env=env, check=True, text=True, capture_output=True)
                    self.assertEqual(result.stdout.splitlines(), [explicit or "0", "1", "0", "1", "1"])


if __name__ == "__main__":
    unittest.main()
