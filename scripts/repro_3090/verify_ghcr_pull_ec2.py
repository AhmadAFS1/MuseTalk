#!/usr/bin/env python3
"""Read-only manifest pull using EC2's existing Secrets Manager pull-only token.

No image layers, GPU rental, Docker login file or credentials are persisted.
This verifies the deployment credential separately from CI's GITHUB_TOKEN.
"""
import argparse
import base64
import hashlib
import json
import logging
import re
import sys
import urllib.request

IMAGE = 'ghcr.io/ahmadafs1/musetalk-rtx3090'
BACKEND_SECRET = 'arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/api-keys-RLmuzo'
PACKAGE = 'https://api.github.com/users/AhmadAFS1/packages/container/musetalk-rtx3090'
ACCEPT = 'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'


def require(ok):
    if not ok:
        raise ValueError('Private pull verification rejected; details suppressed')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Registry redirects are forbidden')


def read_response(opener, request):
    with opener(request, timeout=20) as response:
        require(response.status == 200)
        raw = response.read(1024 * 1024 + 1)
        require(0 < len(raw) <= 1024 * 1024)
        return raw, response.headers


def verify(image, username, token, opener=None):
    require(re.fullmatch(re.escape(IMAGE) + r'@sha256:[0-9a-f]{64}', image)
            and username == 'AhmadAFS1' and re.fullmatch(r'ghp_[A-Za-z0-9]{30,255}', token))
    opener = opener or urllib.request.build_opener(NoRedirect()).open
    raw, headers = read_response(opener, urllib.request.Request(PACKAGE,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'}))
    package = json.loads(raw)
    require(package.get('name') == 'musetalk-rtx3090' and package.get('visibility') == 'private'
            and {s.strip() for s in headers.get('X-OAuth-Scopes', '').split(',') if s.strip()} == {'read:packages'})
    basic = base64.b64encode((username + ':' + token).encode()).decode()
    raw, _ = read_response(opener, urllib.request.Request(
        'https://ghcr.io/token?service=ghcr.io&scope=repository:ahmadafs1/musetalk-rtx3090:pull',
        headers={'Authorization': 'Basic ' + basic}))
    registry_token = json.loads(raw).get('token')
    require(isinstance(registry_token, str) and registry_token and len(registry_token) <= 65536)
    digest = image.split('@', 1)[1]
    raw, headers = read_response(opener, urllib.request.Request(
        'https://ghcr.io/v2/ahmadafs1/musetalk-rtx3090/manifests/' + digest,
        headers={'Authorization': 'Bearer ' + registry_token, 'Accept': ACCEPT}))
    require('sha256:' + hashlib.sha256(raw).hexdigest() == digest
            and headers.get('Docker-Content-Digest') == digest)
    manifest = json.loads(raw)
    require(manifest.get('schemaVersion') == 2 and manifest.get('mediaType') in ACCEPT.split(', ')
            and isinstance(manifest.get('layers'), list) and manifest['layers']
            and all(type(layer.get('size')) is int and layer['size'] >= 0 for layer in manifest['layers']))
    return {'schema': 'musetalk_deployment_pull_verification_v1', 'image': image,
            'status': 'PASS', 'credential_source': 'EC2_IAM_ROLE_SECRETS_MANAGER',
            'token_scopes': ['read:packages'], 'private_visibility': 'VERIFIED',
            'manifest_pull': 'PASS', 'compressed_layer_bytes': sum(layer['size'] for layer in manifest['layers']),
            'layer_downloaded': False, 'gpu_rental': False, 'credentials_persisted': False}


def main():
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    require(re.fullmatch(re.escape(IMAGE) + r'@sha256:[0-9a-f]{64}', args.image))
    import boto3
    from botocore.config import Config
    session = boto3.Session(region_name='us-east-1')
    require(session.get_credentials().method == 'iam-role')
    config = Config(ignore_configured_endpoint_urls=True, connect_timeout=5, read_timeout=20,
                    retries={'total_max_attempts': 2, 'mode': 'standard'})
    identity = session.client('sts', config=config).get_caller_identity()
    require(identity['Account'] == '211125449207' and identity['Arn'].startswith(
        'arn:aws:sts::211125449207:assumed-role/linguaEc2role/'))
    secret = session.client('secretsmanager', config=config).get_secret_value(SecretId=BACKEND_SECRET)
    payload = json.loads(secret['SecretString'])
    result = verify(args.image, payload.get('VAST_MUSETALK_GHCR_USERNAME', ''),
                    payload.get('VAST_MUSETALK_GHCR_PULL_TOKEN', ''))
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'NOT_VERIFIED', 'error_type': type(error).__name__, 'details_suppressed': True}))
        sys.exit(2)
