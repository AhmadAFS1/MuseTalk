#!/usr/bin/env python3
"""Loopback-only cache preparation API. No teacher UNet, Whisper or decoder warmup."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import shutil
import threading
import tempfile
from types import SimpleNamespace

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool


class PreparationManager:
    def __init__(self):
        import torch
        from musetalk.models.vae import VAE
        from musetalk.utils.face_parsing import FaceParsing
        from scripts.api_avatar import APIAvatar
        from scripts.avatar_s3_store import AvatarS3Store
        self.avatar_type = APIAvatar
        self.lock = threading.Lock()
        # Match the established cache encoder's precision/backend policy. The
        # VAE posterior sampling algorithm is unchanged; no teacher is executed.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        self.vae = VAE(model_path='./models/sd-vae', use_float16=True)
        self.vae.vae.eval().requires_grad_(False)
        self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        self.store = AvatarS3Store.from_env(version='v15')
        if not self.store.enabled:
            raise RuntimeError('Preparation requires configured S3 persistence')
        self.args = SimpleNamespace(version='v15', gpu_id=0, extra_margin=10, parsing_mode='jaw',
                                    left_cheek_width=90, right_cheek_width=90)

    def stats(self):
        return {'mode': 'avatar-preparation/1', 'teacher_loaded': False, 'whisper_loaded': False,
                'renderer_available': False, 'avatar_s3': self.store.get_stats()}

    def prepare(self, avatar_id, source, idle, batch_size, force_recreate):
        with self.lock:
            avatar = self.avatar_type(avatar_id=avatar_id, video_path=str(source), bbox_shift=0,
                                     batch_size=batch_size, vae=self.vae, unet=None, pe=None, fp=self.fp,
                                     args=self.args, preparation=True, force_recreate=force_recreate,
                                     idle_video_path=str(idle) if idle else None)
            uploaded = self.store.upload_avatar_dir(avatar_id, Path(avatar.avatar_path))
            if not uploaded:
                raise RuntimeError('Prepared cache was not durably uploaded')
            return {'status': 'success', 'avatar_id': avatar_id, 'already_prepared': False,
                    's3_uploaded': True, 's3_key': self.store._object_key(avatar_id),
                    'preprocessing_contract': 'avatar-preparation/1', 'teacher_executed': False}


def create_app(manager_factory=PreparationManager):
    app = FastAPI(title='MuseTalk cache preparation only')
    @app.on_event('startup')
    def startup():
        app.state.manager = manager_factory()
    @app.get('/stats')
    def stats():
        return app.state.manager.stats()
    @app.post('/avatars/prepare')
    async def prepare(avatar_id: str, video_file: UploadFile = File(...),
                      idle_video_file: UploadFile | None = File(None), bbox_shift: int = 0,
                      batch_size: int = 8, force_recreate: bool = False):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', avatar_id) or '..' in avatar_id:
            raise HTTPException(400, 'Invalid avatar identity')
        if bbox_shift != 0 or not 1 <= batch_size <= 32:
            raise HTTPException(400, 'v15 preparation requires bbox_shift=0 and batch_size1..32')
        root = Path(tempfile.mkdtemp(prefix='avatar-prepare-'))
        def save(upload, role):
            if upload is None: return None
            target = root / f'{avatar_id}_{role}_source.mp4'
            with target.open('wb') as out: shutil.copyfileobj(upload.file, out)
            return target
        source, idle = save(video_file, 'talking'), save(idle_video_file, 'idle')
        try:
            return await run_in_threadpool(app.state.manager.prepare, avatar_id, source, idle, batch_size, force_recreate)
        except Exception as exc:
            raise HTTPException(500, f'Cache preparation failed: {type(exc).__name__}') from exc
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return app


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', '::1'])
    parser.add_argument('--port', type=int, default=8198)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
