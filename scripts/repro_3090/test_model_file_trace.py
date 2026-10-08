import unittest
import tempfile
from pathlib import Path
from analyze_fullfile_model_trace import model_files, scan


class ModelTraceTests(unittest.TestCase):
    def test_training_split_keeps_historical_trace_inventory_complete(self):
        actual = model_files()
        self.assertEqual(len(actual), 14)
        self.assertIn('models/syncnet/latentsync_syncnet.pt', actual)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'old_install_state.py'
            source.write_text('SERVER_MODEL_FILES = ' + repr(actual[:11]) + '\n'
                              'AVATAR_PREP_MODEL_FILES = ' + repr(actual[11:]) + '\n')
            self.assertEqual(set(model_files(source)), set(actual))

    def test_control_completed_failure_unfinished_and_metadata_are_separate(self):
        model = 'models/syncnet/latentsync_syncnet.pt'
        lines = [f'11 openat(AT_FDCWD, "/workspace/MuseTalk/{model}", O_RDONLY) = 3\n',
                 f'12 openat(AT_FDCWD, "{model}", O_RDONLY) = -1 ENOENT\n',
                 f'12 openat(AT_FDCWD, "{model}", O_RDONLY <unfinished ...>\n',
                 f'12 newfstatat(AT_FDCWD, "{model}", {{st_mode=0}}, 0) = 0\n',
                 '12 openat(AT_FDCWD, "latentsync_syncnet.pt", O_RDONLY) = 4\n']
        total, rows = scan(lines, [model], 11)
        self.assertEqual(total, 5)
        self.assertEqual(rows[model]['path_matching_lines'], 4)
        self.assertEqual(rows[model]['literal_basename_matching_lines'], 5)
        self.assertEqual(rows[model]['successful_completed_opens'], 1)
        self.assertEqual(rows[model]['control_pid_completed_opens'], 1)
        self.assertEqual(rows[model]['other_pid_completed_opens'], 0)
        self.assertEqual(rows[model]['matching_unfinished_lines'], 1)


if __name__ == '__main__':
    unittest.main()
