"""Exact-zero upload policy through real timeline, prep/router and ordered compose."""
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.hls_gpu_scheduler import HLSGPUStreamScheduler
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank
from scripts.webrtc_audio_timeline import prepare_webrtc_audio_timeline
from scripts.webrtc_pose_router import LivePoseVideoRouter
from test_motion_current_runtime import current_fixture, make_job, make_scheduler, pixels
from test_webrtc_exact_silence import write_wav
from test_webrtc_pose_router import FakeDecoder


class ExactSilenceTimelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_default_trimming_accepts_zero_without_trimming_or_changing_media(self):
        path=write_wav(self.root/'zero.wav', np.zeros(24000, np.int16))
        before=path.read_bytes()
        timeline=prepare_webrtc_audio_timeline(path, enabled=True)
        self.assertTrue(timeline.exact_silence)
        self.assertTrue(timeline.to_dict()['exact_silence'])
        self.assertEqual(timeline.media_path, str(path))
        self.assertEqual((timeline.original_duration_seconds,timeline.media_duration_seconds),(1.,1.))
        self.assertEqual((timeline.trim_start_seconds,timeline.trim_end_seconds),(0.,1.))
        self.assertFalse(timeline.normalized)
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse((self.root/'zero_timeline.wav').exists())

    def test_original_tiny_nonzero_is_not_reclassified_after_int16_decode(self):
        data=np.zeros(24000,np.float32); data[12000]=1e-20
        path=write_wav(self.root/'tiny.wav',data)
        timeline=prepare_webrtc_audio_timeline(path, enabled=False)
        self.assertFalse(timeline.exact_silence)
        self.assertEqual(timeline.media_path,str(path))
        # The existing audible-activity policy still rejects near-silent clips
        # with default trimming; the new feature never converts them to silence.
        with self.assertRaisesRegex(ValueError,'no sustained activity'):
            prepare_webrtc_audio_timeline(path,enabled=True)

    def test_normal_voiced_timeline_retains_existing_trim_and_nonzero_policy(self):
        voice=(12000*np.sin(2*np.pi*440*np.arange(24000)/24000)).astype(np.int16)
        data=np.concatenate((np.zeros(24000,np.int16),voice,np.zeros(24000,np.int16)))
        path=write_wav(self.root/'voice.wav',data); before=path.read_bytes()
        timeline=prepare_webrtc_audio_timeline(path, enabled=True)
        self.assertFalse(timeline.exact_silence)
        self.assertTrue(timeline.normalized)
        self.assertAlmostEqual(timeline.trim_start_seconds,.92,places=2)
        self.assertAlmostEqual(timeline.trim_end_seconds,2.04,places=2)
        self.assertAlmostEqual(timeline.media_duration_seconds,1.12,places=2)
        self.assertEqual(path.read_bytes(),before)
        self.assertTrue(Path(timeline.media_path).is_file())

    def test_api_forwards_only_server_timeline_fact(self):
        tree=ast.parse((Path(__file__).parent/'api_server.py').read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='submit_webrtc_stream']
        self.assertEqual(len(calls),1)
        value=next(kw.value for kw in calls[0].keywords if kw.arg=='exact_silence')
        self.assertIsInstance(value,ast.Attribute)
        self.assertEqual((value.value.id,value.attr),('audio_timeline','exact_silence'))


class ExactSilencePreparationTest(unittest.TestCase):
    def prepare(self, exact, *, cancel=False, motion=True):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root=Path(temp.name); paths={}
        for pose in ('default',IDLE,TALK,SMILE):
            path=root/(pose+'.mp4');path.write_bytes(b'fixture');paths[pose]=str(path)
        bank=MotionBank(current_fixture()) if motion else None
        with patch('scripts.motion_transitions.configured_bank',return_value=bank):
            router=LivePoseVideoRouter(paths,decoder_factory=FakeDecoder)
        self.addCleanup(router.close)
        router.set_prepared_pose_ids(set(paths))
        avatar=SimpleNamespace(coord_list_cycle=[None]*41)
        total=80  # Long enough to require talking for ordinary speech.
        processor=SimpleNamespace(
            get_audio_feature=Mock(return_value=([torch.zeros((1,2))],64000)),
            encode_whisper_feature=Mock(return_value=(torch.zeros((1,2)),total)),
            build_audio_prompts=Mock(side_effect=lambda **kw:torch.zeros((kw['end_frame']-kw['start_frame'],2,3))))
        session=SimpleNamespace(session_id='silence_prep',avatar_id='idle',batch_size=8,
            prepared_pose_avatar_ids={},active_pose_plan={'version':2,'clock':'audio_progress',
            'segments':[{'at_permille':0,'pose_id':TALK}],'on_complete':IDLE,'switch_mode':'next_boundary'},
            live_pose_router=router, idle_track=SimpleNamespace(get_pose_status=lambda:{'current_pose_id':IDLE}),
            chunk_duration=4)
        scheduler=HLSGPUStreamScheduler.__new__(HLSGPUStreamScheduler)
        scheduler.manager=SimpleNamespace(unet_dtype=torch.float32,_get_or_load_avatar=Mock(return_value=avatar),
            audio_processor=processor,device='cpu',whisper=None,
            args=SimpleNamespace(audio_padding_length_left=2,audio_padding_length_right=2))
        executor=ThreadPoolExecutor(max_workers=2);self.addCleanup(executor.shutdown)
        scheduler.prep_subtask_executor=executor
        scheduler.startup_chunk_count=0;scheduler.startup_chunk_duration_seconds=0
        scheduler._apply_positional_encoding_cpu=lambda x:x
        scheduler._set_request_status=Mock();scheduler._finish_before_enqueue=Mock()
        scheduler.jobs={};scheduler.preparing_requests={'zero'};scheduler.condition=threading.Condition()
        event=threading.Event()
        if cancel:event.set()
        original_plan=copy.deepcopy(session.active_pose_plan)
        with patch('torch.cuda.is_available',return_value=False):
            scheduler._prepare_job(session,'zero','unchanged-original.wav',20,0.,event,None,None,
                                   'webrtc',None,None,None,time.time(),exact)
        return scheduler,session,processor,original_plan

    def test_zero_keeps_original_duration_frame_count_and_routes_only_idle(self):
        scheduler,session,processor,original=self.prepare(True)
        scheduler._finish_before_enqueue.assert_not_called()
        job=scheduler.jobs['zero']
        self.assertTrue(job.exact_silence)
        self.assertEqual((job.total_frames,job.conditioning_ready_frames),(80,80))
        self.assertEqual((job.audio_path,job.generation_fps),('unchanged-original.wav',20))
        self.assertEqual({s.pose_id for s in session.live_pose_router.snapshots_for_range(0,80,20)},{IDLE})
        self.assertEqual(session.active_pose_plan,original)
        processor.encode_whisper_feature.assert_called_once()

    def test_nonzero_retains_existing_talking_plan_and_frame_count(self):
        scheduler,session,processor,_=self.prepare(False)
        scheduler._finish_before_enqueue.assert_not_called()
        self.assertFalse(scheduler.jobs['zero'].exact_silence)
        self.assertEqual(scheduler.jobs['zero'].total_frames,80)
        self.assertIn(TALK,{s.pose_id for s in session.live_pose_router.snapshots_for_range(0,80,20)})

    def test_no_motion_bank_does_not_replace_existing_source_plan(self):
        scheduler,session,_,original=self.prepare(True,motion=False)
        scheduler._finish_before_enqueue.assert_not_called()
        self.assertEqual(session.live_pose_router._pose_plan_request['segments'],original['segments'])

    def test_cancel_before_prep_never_enqueues_or_decodes(self):
        scheduler,_,processor,_=self.prepare(True,cancel=True)
        self.assertEqual(scheduler.jobs,{})
        processor.get_audio_feature.assert_not_called()
        self.assertEqual(scheduler._finish_before_enqueue.call_args.args[6],'cancelled')

    def test_internal_flag_forwards_to_preparation_without_session_global_state(self):
        scheduler=HLSGPUStreamScheduler.__new__(HLSGPUStreamScheduler)
        scheduler.condition=threading.Condition();scheduler.jobs={};scheduler.preparing_requests=set()
        scheduler.max_pending_jobs=8;scheduler.prep_executor=Mock()
        accepted=scheduler.submit_webrtc_stream(SimpleNamespace(), 'zero','source.wav',20,
            threading.Event(),None,None,None,None,exact_silence=True)
        self.assertTrue(accepted)
        args=scheduler.prep_executor.submit.call_args.args
        self.assertEqual(args[-1],True)
        self.assertEqual(args[3],'source.wav')


class ExactSilenceCompositionTest(unittest.TestCase):
    def run_batch(self, exact, *, bank=True, cancel=False):
        job=make_job(MotionBank(current_fixture()) if bank else None,[(IDLE,12),(IDLE,13)])
        job.exact_silence=exact
        scheduler=make_scheduler(job);received=[]
        job.frame_batch_callback=lambda frames,start,total:received.extend((start+i,total,f.copy())for i,f in enumerate(frames))
        scheduler._dispatch_compose_batch(job,[pixels(201),pixels(202)],0)
        if cancel:job.cancel_event.set()
        scheduler.compose_executor.finish(0)
        scheduler._drain_completed_composes()
        return job,received

    def test_entire_zero_emits_exact_raw_frames_and_zero_generated_support(self):
        job,received=self.run_batch(True)
        self.assertEqual([(start,total)for start,total,_ in received],[(1,20),(2,20)])
        for (_,_,frame),source in zip(received,(12,13)):
            np.testing.assert_array_equal(frame,pixels(source))
        self.assertEqual(job.webrtc_last_pose_alpha['bounds'],(0,0,0,0))
        self.assertEqual(job.webrtc_last_pose_alpha['values'].shape,(0,0))
        self.assertEqual(job.composed_frame_idx,2)

    def test_nonzero_never_replaces_generated_mouth_even_if_some_frames_quiet(self):
        job,received=self.run_batch(False)
        for (_,_,frame),generated in zip(received,(201,202)):
            np.testing.assert_array_equal(frame,pixels(generated))
        self.assertEqual(job.composed_frame_idx,2)

    def test_legacy_no_bank_returns_its_selected_raw_source_without_new_pose(self):
        job,received=self.run_batch(True,bank=False)
        for (_,_,frame),source in zip(received,(12,13)):
            np.testing.assert_array_equal(frame,pixels(source))
        self.assertEqual(job.webrtc_last_pose_id,IDLE)

    def test_cancelled_zero_batch_cannot_publish_or_restore_history(self):
        job,received=self.run_batch(True,cancel=True)
        self.assertEqual(received,[])
        self.assertEqual(job.composed_frame_idx,0)
        self.assertIsNone(job.webrtc_last_raw_pose_frame)


if __name__=='__main__':unittest.main()
