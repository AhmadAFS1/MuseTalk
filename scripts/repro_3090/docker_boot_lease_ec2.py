#!/usr/bin/env python3
"""One bounded standalone private-image boot test from EC2, never production rollout.

Secrets remain in memory. Preflight writes non-secret plans only. Create reuses
the durable single-attempt observer and installs exact owned expiry first.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import ec2_startup_runner as ec2

IMAGE = 'ghcr.io/ahmadafs1/musetalk-rtx3090'
WORKER_SECRET = 'arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-Dof4b8'
SOURCE = 'c06624da9d7cebd6aa8f3dd6ad4a0dc8306ec6d6'
LABEL = 'musetalk-r5-3090-dev-20261010-docker-c06624d'
DISK = 64
CAP = .0015


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_verification(value):
    require(value.get('schema') == 'musetalk_full_candidate_verification_v1'
            and value.get('source_revision') == SOURCE
            and re.fullmatch(re.escape(IMAGE) + r'@sha256:[0-9a-f]{64}', str(value.get('image', '')))
            and value.get('published') is True and value.get('serving_image') is True
            and value.get('workflow_conclusion') == 'success'
            and value.get('independent_pull') == 'PASS' and value.get('offline_cpu_check') == 'PASS'
            and value.get('private_visibility') == 'VERIFIED' and value.get('anonymous_pull') == 'DENIED'
            and value.get('promotion_eligible') is False, 'Full private image verification required')
    return value['image']


def filters():
    return {'limit': 100, 'type': 'ondemand', 'rentable': {'eq': True}, 'verified': {'eq': True},
            'num_gpus': {'eq': 1}, 'gpu_name': {'eq': 'RTX 3090'},
            'cpu_cores_effective': {'gte': 16}, 'cpu_ram': {'gte': 32000}, 'cpu_ghz': {'gte': 2},
            'disk_space': {'gte': DISK}, 'allocated_storage': DISK,
            'inet_down': {'gte': 500}, 'inet_down_cost': {'lte': CAP}, 'inet_up_cost': {'lte': CAP},
            'dph_total': {'lte': .30}, 'storage_cost': {'lte': .40},
            'reliability': {'gte': .90}, 'duration': {'gte': 6 * 3600}, 'order': [['dph_total', 'asc']]}


def qualifying(offer):
    for key, cap in (('inet_down_cost', CAP), ('inet_up_cost', CAP), ('storage_cost', .40)):
        value = offer.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= cap:
            return False
    prices = [offer[k] for k in ('dph_total', 'dph_total_adj', 'discounted_total_per_hour') if offer.get(k) is not None]
    if not prices or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= .30 for v in prices):
        return False
    return (offer.get('gpu_name') == 'RTX 3090' and offer.get('num_gpus') == 1
            and offer.get('verification') == 'verified' and offer.get('disk_space', 0) >= DISK
            and offer.get('cpu_cores_effective', 0) >= 16 and offer.get('cpu_ram', 0) >= 32000)


def runtime_env(payload):
    # No callbacks, registry/publisher keys, arbitrary executable overrides or
    # EC2 credentials. Existing avatar caches may be read, not newly uploaded.
    allow = ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN',
             'AWS_DEFAULT_REGION', 'AWS_REGION', 'AVATAR_S3_BUCKET', 'AVATAR_S3_REGION',
             'AVATAR_S3_PREFIX', 'AVATAR_S3_KEY_PREFIX')
    env = {name: str(payload[name]) for name in allow if payload.get(name)}
    require(env.get('AWS_ACCESS_KEY_ID') and env.get('AWS_SECRET_ACCESS_KEY'), 'Worker S3 credentials missing')
    env.update({'MUSETALK_RUNTIME_CONFIG_SOURCE': 'injected', 'MUSETALK_CANDIDATE_STANDALONE': '1',
                'LINGUA_CONTROL_PLANE_ENABLED': '0', 'LINGUA_WORKER_CALLBACK_REQUIRED': '0',
                'AVATAR_S3_ENABLED': '1' if env.get('AVATAR_S3_BUCKET') else '0'})
    return env


def registry_credentials():
    from services.secrets_loader import _load_secret_payload
    from botocore.config import Config
    import boto3
    import requests
    backend = _load_secret_payload(os.environ['AWS_SECRETS_ID'], os.environ.get('AWS_SECRETS_REGION', 'us-east-1'))
    username, token = backend.get('VAST_MUSETALK_GHCR_USERNAME'), backend.get('VAST_MUSETALK_GHCR_PULL_TOKEN')
    require(username == 'AhmadAFS1' and re.fullmatch(r'ghp_[A-Za-z0-9]{30,255}', str(token or '')), 'Separate pull-only credential required')
    response = requests.get('https://api.github.com/users/AhmadAFS1/packages/container/musetalk-rtx3090',
                            headers={'Authorization': 'Bearer ' + token}, timeout=15, allow_redirects=False)
    require(response.status_code == 200 and response.json().get('visibility') == 'private'
            and {s.strip() for s in response.headers.get('X-OAuth-Scopes', '').split(',') if s.strip()} == {'read:packages'},
            'Private visibility/read-only pull scope not verified')
    session = boto3.Session(region_name='us-east-1')
    require(session.get_credentials().method == 'iam-role', 'Worker secret must be read with EC2 role')
    secret = session.client('secretsmanager', config=Config(ignore_configured_endpoint_urls=True)).get_secret_value(SecretId=WORKER_SECRET)
    return f'-u {username} -p {token} ghcr.io', runtime_env(json.loads(secret['SecretString']))


def exclusive(path, text):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), 'w') as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())


def install_expiry(root, deadline):
    unit = LABEL + '-expiry'
    command = f'/home/ec2-user/lingua/venv/bin/python -B {Path(__file__).absolute()} expire --run-dir {root}'
    service = Path('/etc/systemd/system') / (unit + '.service')
    exclusive(service, '[Unit]\nDescription=Owned private MuseTalk Docker test expiry\n[Service]\nType=oneshot\nUser=root\nWorkingDirectory=/home/ec2-user\n'
              + f'ExecStart={command}\nRestart=on-failure\nRestartSec=30s\n')
    stamp = dt.datetime.fromisoformat(deadline).strftime('%Y-%m-%d %H:%M:%S UTC')
    exclusive(service.with_suffix('.timer'), '[Unit]\nDescription=Bound four-hour Docker test\n[Timer]\n'
              + f'OnCalendar={stamp}\nAccuracySec=1s\nPersistent=true\nUnit={unit}.service\n[Install]\nWantedBy=timers.target\n')
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', '--now', unit + '.timer'], check=True, capture_output=True)
    require(subprocess.check_output(['systemctl', 'is-active', unit + '.timer'], text=True).strip() == 'active', 'Expiry timer inactive')
    actual = subprocess.check_output(['systemctl', 'show', unit + '.service', '-p', 'ExecStart', '--value'], text=True)
    require(command in actual, 'Expiry service readback mismatch')
    actual = subprocess.check_output(['systemctl', 'cat', unit + '.timer'], text=True)
    require('OnCalendar=' + stamp in actual, 'Expiry timer deadline readback mismatch')
    return {'deadline_utc': deadline, 'timer_active': True, 'service_exec_verified': True, 'force': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'create', 'sample', 'expire'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--verification', type=Path)
    args = parser.parse_args()
    root = args.run_dir.absolute()
    require(os.getuid() == 0 and root == Path('/home/ec2-user/.local/state') / LABEL
            and not root.is_symlink(), 'Exact EC2-owned test directory required')
    os.umask(0o077)
    ec2.load_credentials()
    from services.vast_client import VastClient
    client, observer = VastClient(), ec2.load_observer()
    if args.mode == 'preflight':
        require(args.verification and not root.exists(), 'Fresh test plan and full image proof required')
        proof = json.loads(args.verification.read_text())
        image = validate_verification(proof)
        login, env = registry_credentials()
        offers = [row for row in client.search_offers(filters()) if qualifying(row)]
        require(offers, 'No offer meets GPU, allocated disk and both transfer-price caps')
        offer = min(offers, key=lambda row: max(row.get(k, 0) for k in ('dph_total', 'dph_total_adj', 'discounted_total_per_hour')))
        require(not any(row.get('label') == LABEL for row in client.list_instances()), 'Owned label already exists; reconcile')
        root.mkdir(mode=0o700)
        observer.atomic_json(root / 'verification.json', proof)
        observer.atomic_json(root / 'offer.json', {'observed_at_utc': observer.utc_now(), 'allocated_storage_gb': DISK, 'offer': offer})
        observer.atomic_json(root / 'plan.json', {'count': 1, 'offer_id': offer['id'], 'create_request': {
            'client_id': 'me', 'image': image, 'label': LABEL, 'disk': DISK, 'runtype': 'args',
            'env': {'-p 8000:8000': '1'}, 'cancel_unavail': True}})
        print(json.dumps({'status': 'PREFLIGHT_ONLY', 'offer_id': offer['id'], 'machine_id': offer['machine_id'],
                          'image': image, 'transfer_cap_usd_per_gb': CAP, 'cloud_writes': False}))
        return
    if args.mode == 'create':
        plan = observer.load_json(root / 'plan.json')
        image = validate_verification(observer.load_json(root / 'verification.json'))
        require(plan['create_request']['image'] == image, 'Plan image mismatch')
        offer = observer.load_json(root / 'offer.json')
        require(qualifying(offer['offer']), 'Offer no longer cost-qualified')
        require(not any(row.get('label') == LABEL for row in client.list_instances()), 'Existing label; never recreate')
        login, env = registry_credentials()
        deadline = (dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(hours=4)).isoformat()
        budget = {'total_usd': 12, 'max_hourly_usd': .30, 'planned_lifetime_hours': 4.5, 'max_lifetime_hours': 4.5,
                  'max_inet_down_usd_per_gb': CAP, 'max_inet_up_usd_per_gb': CAP,
                  'reserved_inet_down_gb': 100, 'reserved_inet_up_gb': 20, 'additional_reserve_usd': 4,
                  'approval_reference': 'User authorized new EC2-created 3090 Docker startup/app test for a few hours; no production rollout.',
                  'lifetime_enforcer_reference': 'Independent exact owned four-hour expiry timer verified before the one provider PUT; 4.5h cost reserve.'}
        reservation = observer.validate_budget(budget, offer, plan)
        observer.initialize(root, LABEL, 'https://console.vast.ai', resource_deadline_utc=deadline, image_digest=image)
        observer.atomic_json(root / 'expiry-enforcement.json', install_expiry(root, deadline))
        def create_api(_base, _path, _token, **kwargs):
            # Single-attempt observer records submitting BEFORE this one PUT.
            # Never persist the enriched request, body/error text or raw result.
            request = {**plan['create_request'], 'image_login': login,
                       'env': {**plan['create_request']['env'], **env}}
            response = client.create_instance(offer_id=plan['offer_id'], create_request=request)
            require(response.get('success') is True and str(response.get('new_contract', '')).isdigit(), 'Provider acceptance not proved; reconcile')
            return {'success': True, 'result': {'actions': [{'action': 'create', 'label': LABEL,
                    'instance_id': str(response['new_contract'])}]}}
        ledger = observer.create_once(root, plan, budget, offer, root / 'budget-ledger.json', 'in-memory-provider-adapter', api_fn=create_api)
        print(json.dumps({'status': ledger['state'], 'instance_id': ledger['instance_id'], 'image': image,
                          'deadline_utc': deadline, 'reserved_usd': reservation['reserved_usd'], 'cache_state': 'unknown'}))
        return
    ledger = observer.load_json(root / 'startup-ledger.json')
    instances = client.list_instances()
    if args.mode == 'expire':
        def destroy_api(_base, _path, _token, **kwargs):
            response = client.destroy_instance(kwargs['payload']['instance_id'])
            return {'success': response.get('success') is True, 'result': {'action': 'destroy'}}
        outcome = ec2.expire_owned(ledger, ledger['resource_deadline_utc'], instances, destroy_api, 'provider-adapter', observer)
        if outcome['status'] == 'destroy':
            outcome['provider_absence_verified'] = not any(str(row.get('id')) == ledger['instance_id'] for row in client.list_instances())
            require(outcome['provider_absence_verified'], 'Destroy absence not verified')
        observer.atomic_json(root / 'resource-expiry.json', outcome)
        print(json.dumps(outcome))
        return
    rows = [row for row in instances if row.get('label') == LABEL]
    require(len(rows) == 1 and str(rows[0]['id']) == ledger['instance_id']
            and rows[0].get('gpu_name') == 'RTX 3090' and rows[0].get('num_gpus') == 1, 'Provider ownership mismatch')
    row = rows[0]
    safe = {key: row.get(key) for key in ('id', 'label', 'gpu_name', 'num_gpus', 'machine_id', 'actual_status',
             'cur_state', 'intended_status', 'public_ipaddr', 'ports', 'image_uuid', 'image_runtype', 'start_date')}
    safe['observed_at_utc'] = observer.utc_now()
    observer.atomic_json(root / 'provider-latest.json', safe)
    print(json.dumps(safe))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'FAILED_CHECK_LEDGER_NO_CREATE_RETRY', 'error_type': type(exc).__name__}))
        sys.exit(2)
