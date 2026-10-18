"""Stdlib-only safeguards; no package installs, GPU checks, AWS calls or rentals."""
import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import types
import unittest
import tempfile
import time
from unittest import mock


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


w = module("python_s3_venv_worker")
e = module("python_s3_venv_ec2")
d = module("python_s3_venv_detached")
b0 = module("python_s3_venv_source_library")


class ArchiveSafety(unittest.TestCase):
    def test_runtime_env_preserves_only_reviewed_public_launch_fields(self):
        raw=b'VAST_TCP_PORT_1455=50145\0VAST_UDP_PORT_3478=50347\0PUBLIC_IPADDR=192.0.2.1\0UNRELATED_SECRET=do-not-copy\0LINGUA_WORKER_TOKEN=do-not-copy\0'
        with mock.patch.dict(w.os.environ,{},clear=True), mock.patch.object(Path,'read_bytes',return_value=raw):
            env=w.worker_runtime_env()
        self.assertEqual(env['VAST_TCP_PORT_1455'],'50145')
        self.assertEqual(env['VAST_UDP_PORT_3478'],'50347')
        self.assertEqual(env['PUBLIC_IPADDR'],'192.0.2.1')
        self.assertNotIn('UNRELATED_SECRET',env)
        self.assertNotIn('LINGUA_WORKER_TOKEN',env)

    def test_restore_space_checked_before_environment_replacement(self):
        self.assertEqual(w.restore_disk_budget(20*1024**3, 10*1024**3, 5*1024**3), 17*1024**3)
        with self.assertRaises(ValueError):
            w.restore_disk_budget(16*1024**3, 10*1024**3, 5*1024**3)

    def test_multipart_upload_needs_only_archive_range_not_restore_manifest(self):
        out = io.StringIO()
        response = mock.MagicMock()
        response.__enter__.return_value = types.SimpleNamespace(status=200, headers={'ETag':'"test"'})
        c = {'url':'https://lingua-musetalk-s3-storage.s3.amazonaws.com'+w.PREFIX+'object',
             'bytes':3, 'offset':0, 'part':1}
        with mock.patch.object(Path, 'is_file', return_value=True), mock.patch.object(Path, 'open', return_value=io.BytesIO(b'abc')), \
             mock.patch.object(w.urllib.request, 'urlopen', return_value=response), mock.patch.object(w.shutil, 'disk_usage') as disk, \
             mock.patch('sys.stdout', out), mock.patch.object(w,'save'):
            w.upload_part(c)
        self.assertEqual(json.loads(out.getvalue())['result'], 'upload_part')
        disk.assert_not_called()

    def member(self, name, kind=tarfile.REGTYPE, link=""):
        m = tarfile.TarInfo(name)
        m.type, m.linkname = kind, link
        return m

    def test_regular_and_internal_links(self):
        for m in [self.member(w.VENV.name + "/lib/a.so"),
                  self.member(w.VENV.name + "/bin/python", tarfile.SYMTYPE, "python3.10"),
                  self.member(w.VENV.name + "/lib/b.so", tarfile.LNKTYPE, w.VENV.name + "/lib/a.so")]:
            w.audit_member(m, "/test/base/python")

    def test_path_traversal_and_other_root(self):
        for name in ["/tmp/a", "other/a", w.VENV.name + "/../a", "../a"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                w.audit_member(self.member(name), "/test/base/python")

    def test_device_and_escaping_links(self):
        for m in [self.member(w.VENV.name + "/a", tarfile.CHRTYPE),
                  self.member(w.VENV.name + "/a", tarfile.SYMTYPE, "../outside"),
                  self.member(w.VENV.name + "/a", tarfile.SYMTYPE, "/etc/passwd"),
                  self.member(w.VENV.name + "/a", tarfile.SYMTYPE, str(w.VENV) + "/../../../etc/passwd"),
                  self.member(w.VENV.name + "/bin/python", tarfile.LNKTYPE, "/test/base/python")]:
            with self.subTest(name=m.name, link=m.linkname), self.assertRaises(ValueError):
                w.audit_member(m, "/test/base/python")

    def test_base_python_symlink_allowed_only_in_bin(self):
        w.audit_member(self.member(w.VENV.name + "/bin/python", tarfile.SYMTYPE, "/test/base/python"), "/test/base/python")
        with self.assertRaises(ValueError):
            w.audit_member(self.member(w.VENV.name + "/lib/python", tarfile.SYMTYPE, "/test/base/python"), "/test/base/python")

    def test_signed_url_exact_hosts_prefix(self):
        for host in ["lingua-musetalk-s3-storage.s3.amazonaws.com", "lingua-musetalk-s3-storage.s3.us-east-1.amazonaws.com"]:
            w.check_signed_url("https://" + host + w.PREFIX + "object?X-Amz-Signature=redacted")

    def test_signed_url_rejects_other_destinations(self):
        host = "lingua-musetalk-s3-storage.s3.amazonaws.com"
        for url in ["http://" + host + w.PREFIX + "a", "https://" + host + ".evil.example" + w.PREFIX + "a",
                    "https://user@" + host + w.PREFIX + "a", "https://" + host + ":444" + w.PREFIX + "a",
                    "https://" + host + "/other/a", "https://" + host + w.PREFIX + "%2e%2e/a",
                    "https://" + host + w.PREFIX + "a#fragment"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                w.check_signed_url(url)


class ControllerSafety(unittest.TestCase):
    def test_approved_quote_is_specific_and_bounded(self):
        o = {"id":e.APPROVED_OFFER, "machine_id":e.APPROVED_MACHINE, "host_id":e.APPROVED_HOST,
             "geolocation":e.APPROVED_LOCATION, "dph_total":.2311111111111111, "dph_total_adj":.2727777777777778,
             "inet_down_cost":e.DOWNLOAD_CAP, "inet_up_cost":e.UPLOAD_CAP}
        self.assertTrue(e.approved_offer(o, lambda value: "dph_total_adj" not in value))
        for key, value in [("machine_id",999), ("id",0), ("geolocation","Texas, US"), ("inet_up_cost",.006),
                           ("dph_total",.301), ("dph_total_adj",.301), ("inet_down_cost",float("nan"))]:
            with self.subTest(key=key):
                self.assertFalse(e.approved_offer(dict(o, **{key:value}), lambda value:True))

    def test_worker_evidence_stage_does_not_collide(self):
        record = {"stage": "VALIDATION_PASS", "seconds": 1}
        events = []
        def event(stage, **fields):
            events.append((stage, fields))
        b = types.SimpleNamespace(require=w.require, event=event)
        ledger = {"deadline_utc": (dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10)).isoformat()}
        with mock.patch.object(e, "ssh_arguments", return_value=["ssh"]), mock.patch.object(e.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], 0, json.dumps(record).encode(), b"")):
            result = e.worker_sync(b, {}, {"mode": "build"}, 120, ledger)
        self.assertEqual(result, record)
        self.assertEqual(events[0][0], "WORKER_INVOCATION_RESULT")
        self.assertEqual(events[1][1]["worker_stage"], "VALIDATION_PASS")

    def test_wrapper_acknowledges_and_reports_source_syntax_failure(self):
        ledger = {"deadline_utc": "2099-01-01T00:00:00+00:00"}
        # Execute only a synthetic invalid source, never the actual Linux helper.
        with mock.patch.object(Path, "read_text", return_value="broken syntax !"):
            script = e.worker_script({"mode": "build"}, ledger)
        r = subprocess.run([__import__('sys').executable, "-B", "-c", script], capture_output=True)
        records = [json.loads(line) for line in r.stdout.splitlines()]
        self.assertEqual(r.returncode, 2)
        self.assertEqual(records[0]["stage"], "WORKER_STARTED")
        self.assertEqual(records[1]["error_type"], "SyntaxError")
        self.assertEqual(r.stderr, b"")

    def test_ssh_diagnostics_never_include_raw_secret_or_signed_url(self):
        secret = b"https://example.com/?X-Amz-Signature=private-value"
        r = subprocess.CompletedProcess([], 255, b"", b"Permission denied (publickey) " + secret)
        facts = e.invocation_diagnostics(r)
        self.assertEqual(facts["error_categories"], ["publickey_authentication"])
        self.assertNotIn("private-value", json.dumps(facts))
        self.assertFalse(facts["raw_stderr_saved"])

    def test_failed_worker_still_persists_transport_facts(self):
        events = []
        b = types.SimpleNamespace(require=w.require, event=lambda stage, **fields:events.append((stage, fields)))
        ledger = {"deadline_utc": (dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10)).isoformat()}
        with mock.patch.object(e, "ssh_arguments", return_value=["ssh"]), mock.patch.object(e.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], 255, b"", b"Connection refused")), self.assertRaises(ValueError):
            e.worker_sync(b, {}, {"mode": "build"}, 120, ledger)
        self.assertEqual(events[0][1]["returncode"], 255)
        self.assertEqual(events[0][1]["error_categories"], ["connection_refused"])

    def test_retry_has_separate_controller_worker_and_s3_namespace(self):
        self.assertTrue(e.LABEL.endswith('-a3'))
        self.assertEqual(e.LABEL, w.LABEL)
        self.assertIn('/' + e.LABEL + '/', w.PREFIX)
        self.assertNotEqual(e.ROOT.name, 'musetalk-python-s3-venv-20261011')

    def test_ssh_uses_keepalives_for_silent_long_operations(self):
        b = types.SimpleNamespace(require=w.require)
        args = e.ssh_arguments({'ssh_host':'host.example','ssh_port':22}, b)
        self.assertIn('ServerAliveInterval=5', args)
        self.assertIn('ServerAliveCountMax=6', args)

    def test_empty_transport_failure_retries_probe_but_not_restore(self):
        b = types.SimpleNamespace(require=w.require, event=lambda *a,**kw:None)
        ledger = {'deadline_utc':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=15)).isoformat()}
        fail = subprocess.CompletedProcess([],255,b'',b'Connection closed by peer')
        good = subprocess.CompletedProcess([],0,b'{"result":"probe"}\n',b'')
        with mock.patch.object(e,'ssh_arguments',return_value=['ssh']), mock.patch.object(e.time,'sleep'), \
             mock.patch.object(e.subprocess,'run',side_effect=[fail,good]) as run:
            self.assertEqual(e.worker_sync(b,{}, {'mode':'probe'},60,ledger)['result'],'probe')
        self.assertEqual(run.call_count,2)
        with mock.patch.object(e,'ssh_arguments',return_value=['ssh']), \
             mock.patch.object(e.subprocess,'run',return_value=fail) as run, self.assertRaises(ValueError):
            e.worker_sync(b,{}, {'mode':'restore'},60,ledger)
        self.assertEqual(run.call_count,1)

    def test_worker_probe_is_read_only(self):
        out = io.StringIO()
        with mock.patch('sys.stdout', out), mock.patch.object(w, 'command') as run:
            w.probe()
        facts = json.loads(out.getvalue())
        self.assertEqual(facts['result'], 'probe')
        self.assertFalse(facts['writes_or_installs'])
        run.assert_not_called()

    def test_invocation_timeout_is_recorded(self):
        events = []
        b = types.SimpleNamespace(require=w.require, event=lambda stage, **fields:events.append((stage, fields)))
        ledger = {'deadline_utc': (dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10)).isoformat()}
        with mock.patch.object(e, 'ssh_arguments', return_value=['ssh']), mock.patch.object(e.subprocess, 'run', side_effect=
                               subprocess.TimeoutExpired(['ssh'], 20)), self.assertRaises(subprocess.TimeoutExpired):
            e.worker_sync(b, {}, {'mode':'probe'}, 20, ledger)
        self.assertEqual(events[0][0], 'WORKER_INVOCATION_TIMEOUT')

    def test_expired_lease_never_sends_ssh(self):
        b = types.SimpleNamespace(require=w.require)
        ledger = {"deadline_utc": (dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=1)).isoformat()}
        with mock.patch.object(e.subprocess, "run") as run, self.assertRaises(ValueError):
            e.worker(b, {}, {}, 120, ledger)
        run.assert_not_called()

    def test_bad_endpoint_rejected(self):
        b = types.SimpleNamespace(require=w.require)
        for row in [{"ssh_host":"host;echo", "ssh_port":22}, {"ssh_host":"host", "ssh_port":70000},
                    {"public_ipaddr":"127.0.0.1", "ports":{"22/tcp":[{"HostPort":22}]}}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                e.ssh_arguments(row, b)


class DetachedSafety(unittest.TestCase):
    def test_rest_smoke_sets_unique_output_name_without_network_or_installs(self):
        responses=[(202,{'status':'queued'}),(200,{'status':'ready'}),
                   (200,{'request_id':'test123'}),(200,{'status':'completed','result':{'success':True}})]
        fake_requests=types.SimpleNamespace(get=lambda *a,**kw:types.SimpleNamespace(status_code=200,content=b'synthetic-video'))
        metadata={'streams':[{'codec_type':'video','nb_read_frames':'10'},{'codec_type':'audio'}]}
        with tempfile.TemporaryDirectory(prefix='musetalk-smoke-name-') as root, \
             mock.patch.object(b0,'ROOT',Path(root)), mock.patch.object(b0,'probe',side_effect=responses) as probe, \
             mock.patch.object(b0,'save'), mock.patch.object(b0,'save_probe_receipt'), mock.patch.object(b0,'event'), \
             mock.patch.object(b0.uuid,'uuid4',return_value=types.SimpleNamespace(hex='a'*32)), \
             mock.patch.object(b0.subprocess,'run'), \
             mock.patch.object(b0.subprocess,'check_output',return_value=json.dumps(metadata).encode()), \
             mock.patch.dict(__import__('sys').modules,{'requests':fake_requests}):
            b0.smoke('https://synthetic.invalid',types.SimpleNamespace(read_bytes=lambda:b'synthetic-audio'))
        self.assertIn('&output_name=startup_aaaaaaaaaaaa',probe.call_args_list[2].args[1])

    def test_invalid_job_id_never_creates_script(self):
        for value in ['../escape','a'*31,'z'*32]:
            with self.assertRaises(ValueError): d.launcher('/tmp/test',value,'pass')

    def test_real_detached_stdlib_child_and_exclusive_relaunch(self):
        with tempfile.TemporaryDirectory(prefix='musetalk-detach-test-') as root:
            job='a'*32
            script=d.launcher(root,job,"import time,json;time.sleep(.1);print(json.dumps({'result':'synthetic'}),flush=True)")
            first=subprocess.run([__import__('sys').executable,'-B','-c',script],capture_output=True,check=True)
            second=subprocess.run([__import__('sys').executable,'-B','-c',script],capture_output=True,check=True)
            self.assertEqual(json.loads(first.stdout)['state'],'started')
            self.assertEqual(json.loads(second.stdout)['state'],'claimed')
            for _ in range(40):
                path=Path(root)/job/'receipt.json'
                if path.exists(): break
                time.sleep(.05)
            self.assertEqual(json.loads(path.read_text())['result']['result'],'synthetic')
            self.assertFalse(any(p.suffix=='.py' for p in Path(root).rglob('*')))

    def test_transport_loss_reads_same_job_instead_of_resubmitting(self):
        b=types.SimpleNamespace(require=w.require,save=lambda *a:None,event=lambda *a,**kw:None)
        ledger={'deadline_utc':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=10)).isoformat()}
        lost=subprocess.CompletedProcess([],255,b'',b'closed by peer')
        done=subprocess.CompletedProcess([],0,b'{"state":"complete","result":{"result":"build"}}',b'')
        with mock.patch.object(e,'ssh_arguments',return_value=['ssh']), mock.patch.object(e.time,'sleep'), \
             mock.patch.object(e.subprocess,'run',side_effect=[lost,done]) as run:
            self.assertEqual(e.worker(b,{}, {'mode':'build'},100,ledger)['result'],'build')
        self.assertIn('subprocess.Popen',run.call_args_list[0].kwargs['input'].decode())
        self.assertNotIn('subprocess.Popen',run.call_args_list[1].kwargs['input'].decode())

    def test_batch_part_failure_is_durable_and_does_not_abort_other_parts(self):
        c={'parts':[{'part':1},{'part':2}]}
        with mock.patch.object(w,'upload_part',side_effect=[ValueError('private-url'),{'part':2,'etag':'ok'}]), \
             mock.patch.object(w,'save'), mock.patch.object(w,'emit') as emit:
            w.upload_batch(c)
        receipt=emit.call_args.args[0]
        self.assertEqual(receipt['result'],'upload_batch')
        self.assertEqual(receipt['parts'][0]['error_type'],'ValueError')
        self.assertEqual(receipt['parts'][1]['etag'],'ok')
        self.assertNotIn('private-url',json.dumps(receipt))


if __name__ == "__main__":
    unittest.main()
