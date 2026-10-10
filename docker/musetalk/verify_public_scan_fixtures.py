"""Read-only public provenance inspection; output fingerprints, never key text."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ghcr
import release

def fetch(url, headers=None, limit=150 * 1024**2):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=90) as response:
        data = response.read(limit + 1)
        assert len(data) <= limit
        return data

def rules(data):
    return [name for name, pattern in [('private-key-material', ghcr.PRIVATE_KEY_MATERIAL),
                                      *zip(ghcr.SECRET_RULES[1:], release.SECRET_PATTERNS[1:])]
            if pattern.search(data)]

rows = []
for project in ('boto3', 'botocore'):
    metadata_url = f'https://pypi.org/pypi/{project}/1.42.97/json'
    metadata = json.loads(fetch(metadata_url))
    wheel = next(item for item in metadata['urls'] if item['filename'].endswith('.whl'))
    payload = fetch(wheel['url'])
    assert hashlib.sha256(payload).hexdigest() == wheel['digests']['sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            findings = rules(data)
            if findings:
                rows.append({'path': 'opt/musetalk/venv/lib/python3.10/site-packages/' + name,
                             'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data),
                             'rules': findings, 'public_artifact_url': wheel['url'],
                             'public_artifact_sha256': hashlib.sha256(payload).hexdigest(),
                             'public_artifact_member': name, 'metadata_url': metadata_url})

def self_test_constants(source):
    constants = []
    for section in re.findall(rb'static const char \w+\[\]\s*=\s*(.*?);', source, re.S):
        strings = re.findall(rb'"(?:[^"\\]|\\.)*"', section)
        value = b''.join(ast.literal_eval('b' + s.decode('ascii')) for s in strings)
        if ghcr.PRIVATE_KEY_MATERIAL.search(value):
            constants.append(value)
    return constants

# Scan the entire exact wheel, not only the file implicated by a failed build.
# Never print the embedded PEM text. This is credential provenance, not a
# conclusion about the separate license/source obligations of bundled FFmpeg.
metadata_url = 'https://pypi.org/pypi/av/16.1.0/json'
metadata = json.loads(fetch(metadata_url))
wheel = next(item for item in metadata['urls']
             if item['filename'] == 'av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl')
payload = fetch(wheel['url'])
assert hashlib.sha256(payload).hexdigest() == wheel['digests']['sha256']
av_source_url = 'https://raw.githubusercontent.com/gnutls/gnutls/3.8.11/lib/crypto-selftests-pk.c'
av_source = fetch(av_source_url)
assert hashlib.sha256(av_source).hexdigest() == '6e596be00754107fd7f7d1f132c2dc8cd0ff12bbdfc6de08c162a1dd5eedc7e8'
av_constants = self_test_constants(av_source)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        data = archive.read(name)
        findings = rules(data)
        if findings:
            pem_blocks = re.findall(rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----\r?\n.*?-----END (?:[A-Z]+ )?PRIVATE KEY-----\r?\n', data, re.S)
            assert len(pem_blocks) == 8 and all(pem in av_constants for pem in pem_blocks)
            rows.append({'path': 'opt/musetalk/venv/lib/python3.10/site-packages/' + name,
                         'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data),
                         'rules': findings, 'public_artifact_url': wheel['url'],
                         'public_artifact_sha256': hashlib.sha256(payload).hexdigest(),
                         'public_artifact_member': name, 'metadata_url': metadata_url,
                         'test_source_url': av_source_url,
                         'test_source_sha256': hashlib.sha256(av_source).hexdigest(),
                         'embedded_pem_blocks': len(pem_blocks),
                         'all_pem_blocks_match_public_self_tests': True})

source_url = 'https://raw.githubusercontent.com/gnutls/gnutls/3.7.3/lib/crypto-selftests-pk.c'
source = fetch(source_url)
constants = self_test_constants(source)

token = json.loads(fetch('https://auth.docker.io/token?service=registry.docker.io&scope=repository:nvidia/cuda:pull'))['token']
headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json'}
digest = release.CUDA_RUNTIME_BASE.split('@')[1]
def manifest(digest):
    data = fetch('https://registry-1.docker.io/v2/nvidia/cuda/manifests/' + digest, headers)
    assert 'sha256:' + hashlib.sha256(data).hexdigest() == digest
    return json.loads(data)
index = manifest(digest)
if 'manifests' in index:
    digest = next(m['digest'] for m in index['manifests'] if m['platform'].get('os') == 'linux' and m['platform'].get('architecture') == 'amd64')
    index = manifest(digest)
target = 'usr/lib/x86_64-linux-gnu/libgnutls.so.30.31.0'
found = False
for layer in index['layers']:
    if layer['size'] > 150 * 1024**2:
        continue
    payload = fetch('https://registry-1.docker.io/v2/nvidia/cuda/blobs/' + layer['digest'], headers)
    assert 'sha256:' + hashlib.sha256(payload).hexdigest() == layer['digest']
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r|*') as archive:
        for member in archive:
            if member.name.removeprefix('./') == target and member.isreg():
                data = archive.extractfile(member).read()
                pem_blocks = re.findall(rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----\r?\n.*?-----END (?:[A-Z]+ )?PRIVATE KEY-----\r?\n', data, re.S)
                assert pem_blocks and all(pem in constants for pem in pem_blocks)
                rows.append({'path': target, 'sha256': hashlib.sha256(data).hexdigest(),
                             'size_bytes': len(data), 'rules': rules(data),
                             'public_image': release.CUDA_RUNTIME_BASE, 'linux_amd64_manifest': digest,
                             'registry_layer_digest': layer['digest'],
                             'test_source_url': source_url, 'test_source_sha256': hashlib.sha256(source).hexdigest(),
                             'embedded_pem_blocks': len(pem_blocks), 'all_pem_blocks_match_public_self_tests': True})
                found = True
                break
    if found:
        break
assert found
expected = ghcr.scan_exceptions()
observed = {(item['path'], item['sha256']): frozenset(item['rules']) for item in rows}
assert observed == expected, 'Public fixtures differ from reviewed exact-file exceptions'
assert hashlib.sha256(source).hexdigest() == json.loads(Path(__file__).with_name('credential_scan_exceptions.json').read_text())['files'][0]['provenance']['test_source_sha256']
print(json.dumps({'schema': 'musetalk_public_scanner_provenance_v1', 'files': rows}, indent=2))
