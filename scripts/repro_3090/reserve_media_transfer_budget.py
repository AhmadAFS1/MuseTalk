#!/usr/bin/env python3
"""Reserve $2 more of the already approved $30; no spend or rental action.

Fixed owned EC2 experiment ledger, SHA-bound input, lock and protected backup.
Fails closed on unknown state, changed reservations, existing backup or cap.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

BASE = Path('/home/ec2-user/.local/state/musetalk-r5-20261008-tools')
LEDGER = BASE / 'shared-budget-ledger.json'
EXPECTED_SHA = '548584fcaf402f6ab398dfc639d33e624bbfcc7909cc5dec8d5c55674cdd094a'
RESERVATION = 'musetalk-r5-20261008-native48-media-transfer-contingency'
EXPECTED_LABELS = {'musetalk-r5-3090-dev-20261008', 'musetalk-r5-20261008-additional-aws-audit'}
BACKUP = BASE / 'shared-budget-before-native48-media-transfer.json'
RECEIPT = BASE / 'native48-media-transfer-reservation.json'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def reserve(data):
    require(data.get('schema') == 'musetalk_experiment_budget_v1' and data.get('total_usd') == 30,
            'approved budget differs')
    rows = data.get('reservations') or {}
    require(set(rows) == EXPECTED_LABELS, 'ledger reservations changed')
    before = sum(row['reserved_usd'] for row in rows.values())
    require(abs(before - 7.5847395833333335) < 1e-9 and before + 2 <= 30, 'reservation total differs or cap exceeded')
    rows[RESERVATION] = {'reserved_usd': 2.0, 'scope': 'Additional private rejected-native48 artifact fresh-GET/request/storage contingency; no free AWS transfer allowance assumed',
                         'new_rental_authorized': False, 'actual_invoice_usd': None,
                         'authorization': 'Existing user $30 total experiment cap; bookkeeping hold only, not new spend approval'}
    return before, before + 2


def write_exclusive(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def main():
    require(str(Path.cwd()) == '/home/ec2-user' and os.getuid() == 0, 'fixed protected EC2 ledger owner context required')
    require(not BASE.is_symlink() and BASE.stat().st_mode & 0o777 == 0o700, 'protected ledger parent missing')
    require(not BACKUP.exists() and not RECEIPT.exists(), 'prior reservation operation exists; reconcile instead')
    with LEDGER.with_suffix('.lock').open('a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(not LEDGER.is_symlink() and LEDGER.stat().st_uid == 0 and stat.S_ISREG(LEDGER.stat().st_mode)
                and LEDGER.stat().st_mode & 0o777 == 0o600, 'protected ledger type/mode invalid')
        raw = LEDGER.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == EXPECTED_SHA, 'ledger SHA changed; do not overwrite')
        data = json.loads(raw)
        before, after = reserve(data)
        write_exclusive(BACKUP, raw)
        updated = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        staging = BASE / 'shared-budget-ledger-native48-staging.json'
        write_exclusive(staging, updated)
        require(hashlib.sha256(LEDGER.read_bytes()).hexdigest() == EXPECTED_SHA, 'ledger changed under lock')
        os.replace(staging, LEDGER)
        receipt = {'schema': 'owned_native48_budget_reservation_v1', 'total_cap_usd': 30,
                   'reserved_before_usd': before, 'reserved_after_usd': after,
                   'previous_ledger_sha256': EXPECTED_SHA, 'new_ledger_sha256': hashlib.sha256(updated).hexdigest(),
                   'reservation': RESERVATION, 'cloud_calls': False, 'paid_resource_created': False,
                   'actual_invoice_usd': None, 'free_aws_transfer_allowance_assumed': False}
        write_exclusive(RECEIPT, (json.dumps(receipt, indent=2) + '\n').encode())
        print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
