"""A silent native process must not defeat worker cancellation or absolute expiry."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import subprocess
import signal
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline

class SilentSubprocessTest(unittest.TestCase):
    def make_job(self,root,options):
        (root/'job.json').write_text(json.dumps({'job_id':'deadline-test','request':{'character_id':'person','options':options}}));return pipeline.Job(root)
    def test_silent_process_stops_at_absolute_job_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=self.make_job(root,{'deadline_epoch':time.time()+0.6});started=time.monotonic()
            with self.assertRaises(pipeline.StageFailed) as raised:
                pipeline.run_logged(job,[sys.executable,'-c','import time;time.sleep(60)'],cwd=root,env=dict(os.environ),timeout=60,log_name='silent.log')
            self.assertEqual(raised.exception.stage,'deadline');self.assertLess(time.monotonic()-started,5);self.assertIsNone(job.process)
    def test_silent_process_stops_when_user_cancels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=self.make_job(root,{});timer=threading.Timer(0.4,job.cancel_event.set);timer.start()
            try:
                with self.assertRaises(pipeline.JobCancelled):
                    pipeline.run_logged(job,[sys.executable,'-c','import time;time.sleep(60)'],cwd=root,env=dict(os.environ),timeout=60,log_name='silent.log')
            finally:timer.cancel()
            self.assertIsNone(job.process)
    def test_orphaned_child_is_killed_even_after_original_parent_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pidfile=root/'child.pid'
            child="import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);open("+repr(str(pidfile))+",'w').write(str(os.getpid()));time.sleep(60)"
            parent="import subprocess,sys;subprocess.Popen([sys.executable,'-c',"+repr(child)+"])"
            proc=subprocess.Popen([sys.executable,'-c',parent],start_new_session=True)
            proc.wait(timeout=3)
            deadline=time.monotonic()+3
            while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
            pid=int(pidfile.read_text())
            try:
                pipeline.terminate(proc,grace=.2)
                time.sleep(.1)
                state=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'),state)
            finally:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
if __name__=='__main__':unittest.main()
