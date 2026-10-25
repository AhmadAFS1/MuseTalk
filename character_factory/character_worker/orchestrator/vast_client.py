#!/usr/bin/env python3
"""Minimal Vast.ai REST client (standard library) for the character-creation orchestrator.

Endpoints mirror what the official CLI does (vast.py): offers are searched with
POST /api/v0/bundles/, an instance is rented with PUT /api/v0/asks/{offer_id}/ carrying
`template_hash_id`, instances are read with GET /api/v0/instances/ and destroyed with
DELETE /api/v0/instances/{id}/. The API key goes in the query string, as the CLI sends it.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE = 'https://console.vast.ai/api/v0'
# the CLI multiplies these query fields (GB on the command line, MB in the API)
FIELD_MULTIPLIER = {'cpu_ram': 1000, 'gpu_ram': 1000, 'gpu_total_ram': 1000, 'duration': 86400.0}
FIELD_ALIAS = {'cuda_vers': 'cuda_max_good', 'dph': 'dph_total', 'reliability': 'reliability2'}
OPS = {'>=': 'gte', '<=': 'lte', '>': 'gt', '<': 'lt', '!=': 'neq', '==': 'eq', '=': 'eq', 'in': 'in', 'notin': 'notin'}
QUERY_TOKEN = re.compile(r'([a-zA-Z0-9_]+)\s*(>=|<=|!=|==|=|>|<|\s+in\s+|\s+notin\s+)\s*(\[[^\]]*\]|"[^"]*"|\S+)')


class VastError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status, self.body = status, body


def numeric_driver_version(text: str) -> int:
    """'580.65.6' -> 580065006, '595.84' -> 595084000 (the encoding Vast compares against)."""
    parts = (str(text).split('.') + ['0', '0'])[:3]
    return int(''.join(p.zfill(3) for p in parts))


def driver_at_least(text: str | None, minimum: str) -> bool:
    try:
        return numeric_driver_version(text or '0') >= numeric_driver_version(minimum)
    except ValueError:
        return False


def parse_query(spec: str) -> dict:
    """CLI-style offer query ('gpu_ram>=12 cpu_ram>=30 driver_version>=580.65.6 rented=False') -> API dict."""
    result: dict = {}
    for field, op, value in QUERY_TOKEN.findall(spec or ''):
        op_name = OPS[op.strip()]
        field = FIELD_ALIAS.get(field, field)
        raw = value.strip()
        if op_name in ('in', 'notin'):
            parsed: object = [v.strip().strip('"').replace('_', ' ') for v in raw.strip('[]').split(',') if v.strip()]
        elif field == 'driver_version':
            parsed = numeric_driver_version(raw)
        elif raw in ('true', 'True'):
            parsed = True
        elif raw in ('false', 'False'):
            parsed = False
        else:
            try:
                parsed = float(raw)
                if field in FIELD_MULTIPLIER:
                    parsed = parsed * FIELD_MULTIPLIER[field]
            except ValueError:
                parsed = raw.strip('"').replace('_', ' ')
        result.setdefault(field, {})[op_name] = parsed
    return result


def mapped_port(instance: dict, internal_port: int) -> tuple[str, int] | None:
    ports = instance.get('ports') or {}
    entries = ports.get(f'{internal_port}/tcp') or []
    ip = instance.get('public_ipaddr')
    if not entries or not ip:
        return None
    try:
        return ip, int(entries[0]['HostPort'])
    except (KeyError, TypeError, ValueError):
        return None


def load_api_key(explicit: str | None = None) -> str:
    for candidate in (explicit, os.environ.get('VAST_API_KEY')):
        if candidate:
            return candidate.strip()
    for path in (Path.home() / '.config' / 'vastai' / 'vast_api_key', Path.home() / '.vast_api_key'):
        if path.is_file():
            return path.read_text().strip()
    raise VastError('no Vast API key: set VAST_API_KEY or create ~/.config/vastai/vast_api_key')


class VastClient:
    def __init__(self, api_key: str | None = None, base: str = DEFAULT_BASE, timeout: int = 60, retries: int = 4, sleep=time.sleep):
        self.api_key = load_api_key(api_key)
        self.base = base.rstrip('/')
        self.timeout, self.retries, self.sleep = timeout, retries, sleep

    def _request(self, method: str, path: str, params: dict | None = None, body: dict | None = None):
        query = {**(params or {}), 'api_key': self.api_key}
        url = f'{self.base}{path}?{urllib.parse.urlencode(query)}'
        data = None if body is None else json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode()
                    return json.loads(text) if text else {}
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors='replace')
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    last = VastError(f'{method} {path}: HTTP {exc.code}', exc.code, text)
                    self.sleep(min(30, 2 ** attempt))
                    continue
                raise VastError(f'{method} {path}: HTTP {exc.code}: {text[:400]}', exc.code, text) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = VastError(f'{method} {path}: {exc}')
                if attempt < self.retries:
                    self.sleep(min(30, 2 ** attempt))
                    continue
        raise last or VastError(f'{method} {path}: failed')

    # ---- offers / instances
    def search_offers(self, query: dict, *, order=(('dph_total', 'asc'),), limit: int = 50, offer_type: str = 'on-demand',
                      storage_gb: float = 5.0) -> list:
        body = {**query, 'order': [list(o) for o in order], 'type': offer_type, 'limit': int(limit), 'allocated_storage': float(storage_gb)}
        body.setdefault('verified', {'eq': True})
        body.setdefault('external', {'eq': False})
        body.setdefault('rentable', {'eq': True})
        body.setdefault('rented', {'eq': False})
        response = self._request('POST', '/bundles/', body=body)
        return response.get('offers') or []

    def create_instance(self, offer_id: int, *, template_hash: str, disk_gb: float, label: str, env: dict | None = None,
                        image: str | None = None, onstart_cmd: str | None = None, extra: dict | None = None) -> int:
        body: dict = {'client_id': 'me', 'template_hash_id': template_hash, 'disk': float(disk_gb), 'label': label}
        if env:
            body['env'] = env
        if image:
            body['image'] = image
        if onstart_cmd:
            body['onstart'] = onstart_cmd
        if extra:
            body.update(extra)
        response = self._request('PUT', f'/asks/{int(offer_id)}/', body=body)
        if not response.get('success') or not response.get('new_contract'):
            raise VastError(f'create instance failed: {response}')
        return int(response['new_contract'])

    def list_instances(self) -> list:
        return self._request('GET', '/instances/', params={'owner': 'me'}).get('instances') or []

    def get_instance(self, instance_id: int) -> dict | None:
        try:
            return self._request('GET', f'/instances/{int(instance_id)}/', params={'owner': 'me'}).get('instances')
        except VastError as exc:
            if exc.status == 404:
                return None
            raise

    def destroy_instance(self, instance_id: int) -> bool:
        try:
            response = self._request('DELETE', f'/instances/{int(instance_id)}/', body={})
        except VastError as exc:
            if exc.status == 404:
                return True
            raise
        return bool(response.get('success', True))

    def set_label(self, instance_id: int, label: str) -> None:
        self._request('PUT', f'/instances/{int(instance_id)}/', body={'label': label})

    def request_logs(self, instance_id: int, tail: int = 2000, wait_s: float = 20.0) -> str:
        response = self._request('PUT', f'/instances/request_logs/{int(instance_id)}/', body={'tail': str(tail)})
        url = response.get('result_url')
        if not url:
            return json.dumps(response)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    return resp.read().decode(errors='replace')
            except (urllib.error.HTTPError, urllib.error.URLError):
                self.sleep(2)
        return ''
