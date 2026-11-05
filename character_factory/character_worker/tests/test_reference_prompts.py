"""Missing character metadata must not invent clothes or scenery across poses."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'character_factory/h3_avatar_workflow'))
from create_avatar import prompt
from three_pose_prompts import pose_prompt, VERSION


class ReferencePromptTest(unittest.TestCase):
    def setUp(self):
        self.profile = {'label': 'Japanese woman', 'age': 26, 'gender': 'woman',
                        'hair': 'straight black hair tucked behind her shoulders', 'tone': 'warm-light skin',
                        'facial_hair': 'none'}

    def test_missing_background_and_wardrobe_preserve_portrait_across_poses(self):
        for pose in ('idle', 'talking', 'smiling'):
            text = pose_prompt(self.profile, pose)
            self.assertNotIn('home office', text)
            self.assertNotIn('navy', text)
            self.assertIn('reference image', text)
        talking = prompt(self.profile)
        self.assertIn('the exact background in the reference image', talking)
        self.assertIn('the exact clothing in the reference image', talking)

    def test_explicit_accepted_wardrobe_and_background_are_byte_identical(self):
        profile = {**self.profile, 'wardrobe': 'navy crew-neck knit top', 'background': 'softly blurred home office'}
        # Captured from the accepted v1 builder before changing its fallback defaults.
        self.assertEqual(hashlib.sha256(prompt(profile).encode()).hexdigest(),
                         '7beb172997666828a8c532be9ef3266c385be5c3c2ce20146fc63ab346155a21')

    def test_empty_metadata_falls_back_to_reference_and_explicit_gray_is_preserved(self):
        for empty in ('', None):
            text = prompt({**self.profile, 'wardrobe': empty, 'background': empty})
            self.assertIn('the exact background in the reference image', text)
            self.assertNotIn('navy', text)
        text = prompt({**self.profile, 'wardrobe': 'a muted teal shirt', 'background': 'plain gray studio background'})
        self.assertIn('a muted teal shirt, plain gray studio background.', text)
        self.assertNotIn('home office', text)

    def test_worker_recipe_version_tracks_prompt_change(self):
        lock = json.loads((ROOT / 'character_factory/character_worker/locks/character_worker.lock.json').read_text())
        self.assertEqual(lock['recipe']['name'], VERSION)
        self.assertEqual(VERSION, 'h3-three-pose-20260927-v2-reference-defaults')


if __name__ == '__main__':
    unittest.main()
