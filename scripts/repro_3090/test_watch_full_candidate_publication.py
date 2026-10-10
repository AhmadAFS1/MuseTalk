import unittest
import watch_full_candidate_publication as watch


class PublicationWaitTests(unittest.TestCase):
    def test_only_exact_run_and_request_are_observed(self):
        run = {'id': 123, 'head_sha': 'a' * 40, 'name': 'MuseTalk private full candidate',
               'status': 'in_progress', 'conclusion': None}
        jobs = [{'name': 'candidate', 'steps': [{'name': next(iter(watch.STEPS)), 'status': 'in_progress'}]}]
        self.assertEqual(watch.snapshot(run, jobs, 123, 'a' * 40)['phases'], ['CPU_CONTRACTS'])
        for field, value in (('id', 124), ('head_sha', 'b' * 40), ('name', 'other'), ('status', 'unknown')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                watch.snapshot({**run, field: value}, jobs, 123, 'a' * 40)
        with self.assertRaises(ValueError):
            watch.snapshot(run, [{'name': 'other'}], 123, 'a' * 40)

    def test_untrusted_step_names_are_not_reflected(self):
        run = {'id': 123, 'head_sha': 'a' * 40, 'name': 'MuseTalk private full candidate',
               'status': 'in_progress', 'conclusion': None}
        jobs = [{'name': 'candidate', 'steps': [{'name': 'untrusted body', 'status': 'in_progress'}]}]
        self.assertEqual(watch.snapshot(run, jobs, 123, 'a' * 40)['phases'], ['RUNNER_SETUP_OR_CHECKOUT'])


if __name__ == '__main__':
    unittest.main()
