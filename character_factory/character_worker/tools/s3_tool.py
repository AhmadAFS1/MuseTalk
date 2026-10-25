#!/usr/bin/env python3
"""Small S3 helper for the character worker (run in the MuseTalk venv, which has boto3).

  s3_tool.py head s3://bucket/key                 -> {"exists": bool, "size": int, "metadata": {...}}
  s3_tool.py get  s3://bucket/key /local/path
  s3_tool.py put  /local/file s3://bucket/key [--metadata k=v ...]
  s3_tool.py put-dir /local/dir s3://bucket/prefix/ [--include-suffix .json .jpg ...]

Credentials come from the environment (AWS_* exported from the worker runtime env).
Prints one JSON object. Exit 0 on success, 3 when an object is missing (head), 1 on error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def split(uri: str) -> tuple[str, str]:
    if not uri.startswith('s3://'):
        raise ValueError(f'not an s3 uri: {uri}')
    bucket, _, key = uri[5:].partition('/')
    if not bucket or not key:
        raise ValueError(f'incomplete s3 uri: {uri}')
    return bucket, key


def client():
    import boto3
    from botocore.config import Config
    region = os.getenv('AVATAR_S3_REGION') or os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION') or 'us-east-1'
    return boto3.client('s3', region_name=region, config=Config(connect_timeout=10, read_timeout=120,
                                                                 retries={'max_attempts': 4, 'mode': 'standard'}))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    h = sub.add_parser('head'); h.add_argument('uri')
    g = sub.add_parser('get'); g.add_argument('uri'); g.add_argument('dest', type=Path)
    u = sub.add_parser('put'); u.add_argument('src', type=Path); u.add_argument('uri'); u.add_argument('--metadata', nargs='*', default=[])
    d = sub.add_parser('put-dir'); d.add_argument('src', type=Path); d.add_argument('uri'); d.add_argument('--include-suffix', nargs='*')
    args = p.parse_args()
    try:
        s3 = client()
        if args.command == 'head':
            bucket, key = split(args.uri)
            try:
                head = s3.head_object(Bucket=bucket, Key=key)
            except Exception as exc:  # botocore ClientError
                code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
                if code in ('404', 'NoSuchKey', 'NotFound'):
                    print(json.dumps({'exists': False, 'uri': args.uri}))
                    return 3
                raise
            print(json.dumps({'exists': True, 'uri': args.uri, 'size': head.get('ContentLength'),
                              'metadata': head.get('Metadata', {}), 'etag': head.get('ETag'),
                              'last_modified': head['LastModified'].isoformat() if head.get('LastModified') else None}))
            return 0
        if args.command == 'get':
            bucket, key = split(args.uri)
            args.dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.dest.with_name(args.dest.name + '.partial')
            s3.download_file(bucket, key, str(tmp))
            tmp.replace(args.dest)
            print(json.dumps({'uri': args.uri, 'dest': str(args.dest), 'size': args.dest.stat().st_size}))
            return 0
        if args.command == 'put':
            bucket, key = split(args.uri)
            extra = {}
            meta = dict(kv.split('=', 1) for kv in args.metadata if '=' in kv)
            if meta:
                extra['Metadata'] = meta
            s3.upload_file(str(args.src), bucket, key, ExtraArgs=extra or None)
            print(json.dumps({'uri': args.uri, 'size': args.src.stat().st_size}))
            return 0
        if args.command == 'put-dir':
            bucket, prefix = split(args.uri.rstrip('/') + '/x')
            prefix = prefix[:-1]
            uploaded = []
            for path in sorted(x for x in args.src.rglob('*') if x.is_file()):
                if args.include_suffix and path.suffix.lower() not in args.include_suffix:
                    continue
                key = prefix + path.relative_to(args.src).as_posix()
                s3.upload_file(str(path), bucket, key)
                uploaded.append({'key': key, 'size': path.stat().st_size})
            print(json.dumps({'prefix': f's3://{bucket}/{prefix}', 'uploaded': len(uploaded), 'bytes': sum(u['size'] for u in uploaded), 'files': uploaded}))
            return 0
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 1
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
