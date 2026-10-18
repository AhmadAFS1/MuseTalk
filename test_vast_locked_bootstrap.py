"""Source/mock launch tests only. No pip, CUDA, AWS, Git network or real boot."""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'scripts'))
import issue_dependency_read_tickets as issuer


class FakeS3:
    def __init__(self,raw,manifest):
        self.raw=raw;self.signed=[];self.gets=[];self.heads=[]
        self.sizes={a['filename']:a['size_bytes'] for a in manifest['artifacts'] if a['source'].startswith('build:')}
    def get_object(self,**params):
        self.gets.append(params);return {'Body':io.BytesIO(self.raw)}
    def head_object(self,**params):
        self.heads.append(params);return {'ContentLength':self.sizes[params['Key'].rsplit('/',1)[1]]}
    def generate_presigned_url(self,operation,**params):
        self.signed.append((operation,params));return 'https://private.invalid/fixture'


class TicketIssuerTests(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((REPO/'configs/dependencies/rtx3090-cp310-cu121-screening.json').read_text())
        self.raw=(REPO/'docs/dependency_optimization_20261015/wheelhouse-manifest-a2.json').read_bytes()
        self.manifest=json.loads(self.raw);self.s3=FakeS3(self.raw,self.manifest)
        self.identity={'Account':self.contract['owner'],
                       'Arn':'arn:aws:sts::'+self.contract['owner']+':assumed-role/linguaEc2Role/fixture'}
    def test_only_five_owner_bound_gets_no_cloud_mutations(self):
        value=issuer.issue(self.contract,self.s3,self.identity)
        self.assertEqual(set(value),{'manifest','wheels'});self.assertEqual(len(value['wheels']),4)
        self.assertEqual(len(self.s3.signed),5)
        for operation,params in self.s3.signed:
            self.assertEqual(operation,'get_object');self.assertEqual(params['ExpiresIn'],3600)
            self.assertEqual(params['Params']['ExpectedBucketOwner'],self.contract['owner'])
            self.assertTrue(params['Params']['Key'].startswith(self.contract['prefix']))
        self.assertNotIn('Credentials',value)
    def test_manifest_digest_is_independent_and_failure_signs_nothing(self):
        self.s3.raw+=b'corrupt'
        with self.assertRaisesRegex(ValueError,'manifest differs'):
            issuer.issue(self.contract,self.s3,self.identity)
        self.assertFalse(self.s3.signed)
    def test_wrong_role_account_or_unbounded_expiry_refused(self):
        for identity in ({**self.identity,'Account':'other'},{**self.identity,'Arn':'unapproved'}):
            with self.assertRaises(ValueError):issuer.issue(self.contract,self.s3,identity)
        for expiry in (0,299,3601,True):
            with self.assertRaises(ValueError):issuer.issue(self.contract,self.s3,self.identity,expiry)
        self.assertFalse(self.s3.gets)
    def test_wrong_private_wheel_size_refused(self):
        self.s3.sizes={k:1 for k in self.s3.sizes}
        with self.assertRaisesRegex(ValueError,'wheel missing'):
            issuer.issue(self.contract,self.s3,self.identity)


class BootstrapTests(unittest.TestCase):
    def run_shell(self,code,*args):
        # Load definitions only; main() is never invoked by sourcing.
        env={k:v for k,v in os.environ.items() if not k.startswith(('AWS_','MUSETALK_','VENV_PATH'))}
        return subprocess.run(['bash','-s','--',str(REPO/'scripts/vast_bootstrap_locked.sh'),*map(str,args)],
                              input='source "$1"\n'+code,text=True,env=env,capture_output=True,timeout=10)
    def fixture(self,root):
        repo=root/'repo';(repo/'configs/dependencies').mkdir(parents=True);(repo/'scripts').mkdir()
        name='configs/dependencies/rtx3090-cp310-cu121-screening.json'
        (repo/name).write_bytes((REPO/name).read_bytes())
        (repo/'scripts/install_locked_dependencies.py').write_text('# mock only')
        return repo
    def test_pipe_not_environment_and_safe_full_stack_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=self.fixture(root)
            proc=self.run_shell('''REPO_DIR="$2"; MUSETALK_LOCKED_VENV_PATH="$3"; PORT=8000
export MUSETALK_DEPENDENCY_TICKETS_JSON='{"fixture":"ticket"}'
configure_locked_boot
[[ -z "${MUSETALK_DEPENDENCY_TICKETS_JSON+set}" ]]
[[ "$SETUP_CLEAN" == 0 && "$SETUP_FULL_STACK" == 1 && "$MUSETALK_DEPENDENCY_WORKERS" == 4 ]]
[[ "$MUSETALK_DEPENDENCY_INSTALL_MODE" == locked-wheelhouse-uv ]]
line=""; IFS= read -r -u 8 line || [[ -n "$line" ]]
[[ "$line" == '{"fixture":"ticket"}' ]]
exec 8<&-
''',repo,root/'venv')
            self.assertEqual(proc.returncode,0,proc.stderr);self.assertNotIn('ticket',proc.stdout)
    def test_fresh_install_without_ticket_refuses_before_install(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=self.fixture(root)
            proc=self.run_shell('REPO_DIR="$2"; MUSETALK_LOCKED_VENV_PATH="$3"; PORT=8000\nconfigure_locked_boot\n',repo,root/'venv')
            self.assertEqual(proc.returncode,2);self.assertIn('requires EC2-issued',proc.stderr)
            self.assertFalse((root/'venv').exists())
    def test_auth_requires_worker_identity_and_never_logs_values(self):
        proc=self.run_shell('configure_runtime_auth\n')
        self.assertEqual(proc.returncode,2);self.assertIn('Worker AWS read identity missing',proc.stderr)
        proc=self.run_shell('''AWS_ACCESS_KEY_ID=fixture_id; AWS_SECRET_ACCESS_KEY=fixture_secret
MUSETALK_RUNTIME_CONFIG_SOURCE=injected; AVATAR_S3_BUCKET=fixture_avatar; TRT_ARTIFACT_S3_BUCKET=fixture_engine
MUSETALK_AWS_SECRET_ID=old_reference
configure_runtime_auth
[[ -z "$MUSETALK_AWS_SECRET_ID" ]]
''')
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertNotIn('fixture_secret',proc.stdout+proc.stderr)
    def test_unchanged_checkout_does_not_fetch_or_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';(repo/'.git').mkdir(parents=True)
            sentinel=repo/'runtime-sentinel';sentinel.write_text('keep')
            proc=self.run_shell('''REPO_DIR="$2"; REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
MUSETALK_SOURCE_REVISION=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
stat() { id -u; }
git() {
 case "$*" in
  *'remote get-url origin') printf '%s\\n' "$REPO_URL" ;;
  *'rev-parse HEAD') printf '%s\\n' "$MUSETALK_SOURCE_REVISION" ;;
  *'diff --quiet --'|*'diff --cached --quiet --') return 0 ;;
  *) return 71 ;;
 esac
}
checkout_code
''',repo)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(sentinel.read_text(),'keep');self.assertIn('no clone/fetch',proc.stdout)
    def test_canonical_onstart_closes_owned_pipe_only(self):
        source=(REPO/'scripts/vast_onstart.sh').read_text()
        helper=re.search(r'^close_bootstrap_dependency_ticket\(\) \{\n.*?^\}\n',source,re.M|re.S).group()
        proc=self.run_shell(helper+'''exec 8< <(printf fixture)
export MUSETALK_BOOTSTRAP_TICKET_FD=8 MUSETALK_DEPENDENCY_TICKET_FD=8
close_bootstrap_dependency_ticket
[[ -z "${MUSETALK_DEPENDENCY_TICKET_FD+set}" ]]
if { IFS= read -r -u 8 value; } 2>/dev/null; then exit 72; fi
printf 'closed\\n'
''')
        self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(proc.stdout,'closed\n')
        self.assertIn('  run_setup_if_needed\n  close_bootstrap_dependency_ticket\n',source)
        bootstrap=(REPO/'scripts/vast_bootstrap_locked.sh').read_text()
        self.assertIn('bash scripts/vast_onstart.sh 9<&-',bootstrap)

    def test_active_pid_blocks_code_update_without_fetch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';(repo/'.git').mkdir(parents=True)
            pid=root/'api.pid';pid.write_text('fixture')
            proc=self.run_shell('''REPO_DIR="$2"; PID_FILE="$3"; PORT=8000
REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
MUSETALK_SOURCE_REVISION=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
stat() { id -u; }
git() {
 case "$*" in
  *'remote get-url origin') printf '%s\\n' "$REPO_URL" ;;
  *'rev-parse HEAD') printf 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\\n' ;;
  *'diff --quiet --'|*'diff --cached --quiet --') return 0 ;;
  *) return 71 ;;
 esac
}
checkout_code
''',repo,pid)
            self.assertEqual(proc.returncode,2,proc.stderr);self.assertIn('API PID file exists',proc.stderr)
            self.assertEqual(pid.read_text(),'fixture')

    def test_tracked_edits_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';(repo/'.git').mkdir(parents=True)
            sentinel=repo/'modified.py';sentinel.write_text('user edit')
            proc=self.run_shell('''REPO_DIR="$2"; REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
stat() { id -u; }
git() {
 case "$*" in
  *'remote get-url origin') printf '%s\\n' "$REPO_URL" ;;
  *'diff --quiet --') return 1 ;;
  *) return 71 ;;
 esac
}
checkout_code
''',repo)
            self.assertEqual(proc.returncode,2,proc.stderr);self.assertIn('Tracked local edits',proc.stderr)
            self.assertEqual(sentinel.read_text(),'user edit')

    def _update_with_pid(self,pid_text):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';(repo/'.git').mkdir(parents=True)
            pid=root/'api.pid';pid.write_text(pid_text)
            proc=self.run_shell('''REPO_DIR="$2"; PID_FILE="$3"; PORT=8000
REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
MUSETALK_SOURCE_REVISION=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
stat() { id -u; }
head_rev=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
git() {
 case "$*" in
  *'remote get-url origin') printf '%s\\n' "$REPO_URL" ;;
  *'rev-parse HEAD') printf '%s\\n' "$head_rev" ;;
  *'diff --quiet --'|*'diff --cached --quiet --') return 0 ;;
  *'fetch --depth 1 origin '*) printf 'fetched\\n' ;;
  *'checkout --detach --no-overwrite-ignore '*) head_rev=$MUSETALK_SOURCE_REVISION; printf 'checked-out\\n' ;;
  *) return 71 ;;
 esac
}
checkout_code
''',repo,pid)
            return proc,pid.read_text()

    def test_dead_pid_from_previous_container_allows_code_update(self):
        # PID 2**22+7 exceeds Linux pid_max defaults and is never a live process.
        proc,left=self._update_with_pid(str(2**22+7))
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertIn('Stale API PID file',proc.stdout);self.assertIn('fetched',proc.stdout)
        self.assertEqual(left,str(2**22+7))

    def test_live_non_api_pid_is_stale_but_ambiguous_content_blocks(self):
        # PID 1 is alive but is never the MuseTalk API chain.
        proc,_=self._update_with_pid('1')
        self.assertEqual(proc.returncode,0,proc.stderr);self.assertIn('checked-out',proc.stdout)
        proc,_=self._update_with_pid('not-a-pid')
        self.assertEqual(proc.returncode,2);self.assertNotIn('fetched',proc.stdout)

    def test_inherited_generic_venv_path_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=self.fixture(root)
            proc=self.run_shell('''REPO_DIR="$2"; PORT=8000
export VENV_PATH=/workspace/.venvs/musetalk_trt_stagewise
export MUSETALK_DEPENDENCY_TICKETS_JSON='{}'
configure_locked_boot
[[ "$VENV_PATH" == /workspace/.venvs/musetalk_locked_* ]]
exec 8<&-
''',repo)
            self.assertEqual(proc.returncode,0,proc.stderr);self.assertIn('Ignoring inherited VENV_PATH',proc.stdout)

    def test_bootstrap_emits_dated_start_marker(self):
        source=(REPO/'scripts/vast_bootstrap_locked.sh').read_text()
        self.assertRegex(source,r'bootstrap_log "start boot_id=')
        proc=self.run_shell('bootstrap_log probe\n')
        self.assertRegex(proc.stdout,r'^\[bootstrap\] \[\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\] probe$')

    def test_proc1_environ_readers_never_exit_early(self):
        # awk 'exit' closes the pipe early; tr then dies with SIGPIPE and
        # pipefail turns that into exit 141 under set -e (18 KB Vast env).
        for name in ('vast_onstart.sh','vast_server_ctl.sh','run_turnserver_tcp_relay.sh','run_webrtc_relay_api_server.sh'):
            source=(REPO/'scripts'/name).read_text()
            self.assertIn('</proc/1/environ'.replace('<','< '),source,name)
            self.assertNotIn('print; exit}',source,name)
            self.assertIn("'!found && $1 == key {sub(/^[^=]*=/, \"\"); print; found=1}'",source,name)


if __name__=='__main__':unittest.main()
