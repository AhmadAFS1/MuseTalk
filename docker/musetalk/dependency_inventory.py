#!/usr/bin/env python3
"""Read-only installed dependency fingerprints; not a license acceptance decision."""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess


def fingerprint(path, root):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': path.relative_to(root).as_posix(), 'size_bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def inventory(root, site_packages, dpkg_rows):
    root = root.resolve()
    site_packages = site_packages.resolve()
    if not inside(site_packages, root):
        raise ValueError('Site packages outside inventory root')
    distributions = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        notices, native, missing_native = [], [], []
        record_text = distribution.read_text('RECORD')
        # Newer importlib versions can omit missing files from .files; retain
        # raw RECORD ownership so intentionally pruned payloads remain visible.
        members = [row[0] for row in csv.reader(io.StringIO(record_text)) if row] if record_text is not None else distribution.files or []
        for member in dict.fromkeys(members):
            path = Path(distribution.locate_file(member))
            # RECORD may name scripts outside site-packages; only bounded package
            # payloads are inspected. Direct-URL/auth metadata is never reported.
            if not inside(path, site_packages):
                continue
            path = path.resolve()
            name = path.name.lower()
            notice = (re.search(r'(?:^|[._-])(?:licen[cs]es?|copying|notices?|copyright)(?:[._-]|$)', name) is not None
                      or any(part.lower() in {'license', 'licenses', 'licence', 'licences', 'notices'}
                             for part in path.relative_to(site_packages).parts[:-1]))
            library = re.search(r'\.so(?:\.|$)', name) is not None
            if not (notice or library):
                continue
            if not path.is_file():
                if library:
                    missing_native.append(path.relative_to(root).as_posix())
                continue
            record = fingerprint(path, root)
            if notice:
                notices.append(record)
            if library:
                native.append(record)
        label = distribution.metadata.get('License-Expression') or distribution.metadata.get('License') or ''
        distributions.append({'name': distribution.metadata['Name'], 'version': distribution.version,
                              'declared_license': label if re.fullmatch(r'[A-Za-z0-9() +.\-]{1,160}', label) else None,
                              'license_metadata_sha256': hashlib.sha256(label.encode()).hexdigest(),
                              'installed_notice_files': sorted(notices, key=lambda x: x['path']),
                              'native_library_files': sorted(native, key=lambda x: x['path']),
                              'missing_record_native_files': sorted(missing_native)})
    os_notices = []
    doc_root = root / 'usr/share/doc'
    for path in sorted(doc_root.glob('*/copyright')):
        if inside(path, doc_root) and path.is_file():
            os_notices.append(fingerprint(path, root))
    packages = []
    for line in dpkg_rows.splitlines():
        fields = line.split('\t')
        if len(fields) != 4:
            raise ValueError('Unexpected dpkg inventory row')
        packages.append(dict(zip(('binary_package', 'version', 'source_package', 'source_version'), fields)))
    return {'schema': 'musetalk_installed_dependency_inventory_v1',
            'pip_distributions': sorted(distributions, key=lambda x: (x['name'].lower(), x['version'])),
            'dpkg_packages': sorted(packages, key=lambda x: x['binary_package']),
            'os_notice_files': os_notices, 'publication_review_accepted': False,
            'limitation': 'Installed byte/notice fingerprints and package source identities only; no source delivery or legal/GPU acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'))
    parser.add_argument('--site-packages', type=Path, default=Path('/opt/musetalk/venv/lib/python3.10/site-packages'))
    args = parser.parse_args()
    if args.root != Path('/'):
        raise ValueError('CLI inventory must run inside its own image root')
    rows = subprocess.check_output(['dpkg-query', '-W', '-f', '${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n'], text=True)
    print(json.dumps(inventory(args.root, args.site_packages, rows), indent=2))


if __name__ == '__main__':
    main()
