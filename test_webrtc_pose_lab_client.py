"""Execute the shipped browser JavaScript in Node, with controlled I/O races.

These tests exercise requests and lifecycle behavior, not HTML string matches.
The separate Chromium harness covers real DOM/media/SDP interoperability.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from templates.webrtc_pose_lab import get_webrtc_pose_lab_html

NODE = os.environ.get("NODE_BINARY") or shutil.which("node")
if not NODE:
    candidate = Path("/opt/nvm/versions/node/v22.15.0/bin/node")
    NODE = str(candidate) if candidate.is_file() else None

EXPECTED = {"routing_sha256": "routing", "source_hashes": {
    "neutral_resting": "idle", "speaking_direct": "talk", "light_smile": "smile"}}

HARNESS = r'''
const vm = require("node:vm");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a; reject=b;}); return {promise,resolve,reject}; };
const response = (body, status=200) => ({ok:status<400,status,text:async()=>JSON.stringify(body)});
const clone = (x) => JSON.parse(JSON.stringify(x));
async function until(check) { const start=Date.now(); while(!check()) { if(Date.now()-start>3000) throw new Error("timed out waiting for client state"); await new Promise(setImmediate); } }
function lab() {
  const elements = new Map();
  function element(id="") {
    return { id, textContent:"", value:"", disabled:false, hidden:false, files:[], dataset:{}, className:"", children:[], listeners:{},
      classList:{toggle(){}}, appendChild(child){this.children.push(child);},
      addEventListener(type, callback){this.listeners[type]=callback;},
      click(){if(this.disabled) return; return this.listeners.click?.();},
      play:async()=>{}, scrollTop:0, scrollHeight:0 };
  }
  const document = {
    getElementById(id){if(!elements.has(id)) elements.set(id,element(id)); return elements.get(id);},
    createElement(){return element();},
    querySelectorAll(query) {
      if(query === ".pose-button") return this.getElementById("poseButtons").children;
      if(query === "[data-event]") return events;
      throw new Error("Unexpected selector " + query);
    },
  };
  const events=["user_speech_started","user_speech_ended","assistant_thinking","assistant_turn_aborted"].map(event=>{const e=element();e.dataset.event=event;return e;});
  for(const [id,value] of Object.entries({fps:"20",batchSize:"4",reactionIntent:"none",speechMovement:"talking"})) document.getElementById(id).value=value;
  document.getElementById("audioFile").files=[{name:"tts.wav"}];
  const calls=[]; const peers=[]; let sessionCounter=0;
  const state={session_id:"",active_stream:null,pose_protocol:{last_seq:0,active_turn_id:null,current_pose_id:"neutral_resting"},track_stats:{video:{motion:{enabled:true,settled:true,building:false,bank:{routing_sha256:"routing",sources:{neutral_resting:{sha256:"idle"},speaking_direct:{sha256:"talk"},light_smile:{sha256:"smile"}}}}}}};
  const hooks={};
  class Peer {
    constructor(){peers.push(this);this.connectionState="new";this.iceConnectionState="new";this.transceivers=[];}
    addTransceiver(kind,options){this.transceivers.push({kind,...options});}
    async createOffer(){return {type:"offer",sdp:"mock offer with VP8"};}
    async setLocalDescription(offer){this.localDescription=offer;}
    async setRemoteDescription(answer){this.remoteDescription=answer;this.connectionState="connected";this.onconnectionstatechange?.();}
    close(){this.connectionState="closed";this.onconnectionstatechange?.();}
  }
  class Form { constructor(){this.data=new Map();} append(key,value){this.data.set(key,value);} get(key){return this.data.get(key);} }
  const box={document,URLSearchParams,Date,Promise,JSON,String,Number,Boolean,Object,Array,Error,Math,encodeURIComponent,AbortController,
    setTimeout,clearTimeout,setInterval:()=>1,clearInterval:()=>{},FormData:Form,File:class {constructor(_d,name,opts){this.name=name;this.type=opts.type;}},
    RTCPeerConnection:Peer,MediaStream:class {constructor(){this.tracks=[];}getTracks(){return this.tracks;}addTrack(t){this.tracks.push(t);}},
    window:{location:{origin:"http://worker",assign(url){box.navigation=url;}},addEventListener(){}},
    fetch:async(url,options={})=>{
      const call={url,options};calls.push(call);
      if(hooks.all){const result=hooks.all(call);if(result!==undefined)return await result;}
      if(url.includes("/create?")) {const id=`session-${++sessionCounter}`;state.session_id=id;return response({session_id:id});}
      if(url.endsWith("/status")) return hooks.status ? await hooks.status(call) : response(clone(state));
      if(url.endsWith("/offer")) return response({type:"answer",sdp:"mock answer"});
      if(url.endsWith("/stream")) {
        state.active_stream="request-"+options.body.get("turn_id");state.pose_protocol.active_turn_id=options.body.get("turn_id");
        return hooks.stream ? await hooks.stream(call) : response({status:"streaming"});
      }
      if(url.endsWith("/events")) {
        const payload=JSON.parse(options.body);
        if(hooks.event)return await hooks.event(call);
        if(payload.seq<=state.pose_protocol.last_seq)return response({accepted:false,reason:"stale_seq"});
        if(payload.event==="assistant_turn_aborted" && payload.turn_id!==state.pose_protocol.active_turn_id)return response({accepted:false,reason:"turn_mismatch"});
        if(!state.active_stream)state.pose_protocol.active_turn_id=payload.turn_id;
        state.pose_protocol.last_seq=payload.seq;
        if(payload.event==="assistant_turn_aborted")state.active_stream=null;
        return response({accepted:true});
      }
      if(options.method==="DELETE") return response({deleted:true});
      if(url.endsWith("/pose") || url.endsWith("/ice"))return response({accepted:true});
      if(hooks.audio) return await hooks.audio(call);
      throw new Error("Unexpected fetch "+url);
    }};
  const baseFetch=box.fetch;
  box.fetch=(url,options={})=>{
    if(!options.signal)return baseFetch(url,options);
    if(options.signal.aborted)return Promise.reject(new Error("AbortError"));
    return new Promise((resolve,reject)=>{
      const abort=()=>reject(new Error("AbortError"));
      options.signal.addEventListener("abort",abort,{once:true});
      baseFetch(url,options).then(resolve,reject).finally(()=>options.signal.removeEventListener("abort",abort));
    });
  };
  vm.createContext(box);vm.runInContext(input.script,box);
  return {box,document,events,calls,hooks,state,peers,eval:(code)=>vm.runInContext(code,box),
    async connect(){await this.eval("createAndConnect()");assert.equal(this.eval("ready"),true);},
    streamCalls(){return calls.filter(c=>c.url.endsWith("/stream"));},
    eventCalls(){return calls.filter(c=>c.url.endsWith("/events"));}};
}
const cases={
 async connect_verifies_bank_and_offer() {
   const l=lab();await l.connect();
   const create=l.calls.find(c=>c.url.includes("/create?"));const url=new URL(create.url);
   assert.equal(url.searchParams.get("pose_switch_mode"),"next_boundary");
   assert.equal(l.document.getElementById("bankState").textContent,"verified");
   assert.equal(l.document.getElementById("streamButton").disabled,false);
   assert.deepEqual(l.peers[0].transceivers,[{kind:"video",direction:"recvonly"},{kind:"audio",direction:"recvonly"}]);
   assert.equal(l.document.getElementById("characterSelect").disabled,true);
   assert(l.calls.findIndex(c=>c.url.endsWith("/status")) < l.calls.findIndex(c=>c.url.endsWith("/offer")));
 },
 async mismatched_bank_fails_closed() {
   for(const mutation of [s=>s.track_stats.video.motion.enabled=false,s=>s.track_stats.video.motion.bank.routing_sha256="other",s=>s.track_stats.video.motion.bank.sources.light_smile.sha256="other"]) {
     const l=lab();mutation(l.state);await l.eval("createAndConnect()");
     assert.equal(l.eval("sessionId"),null);assert.equal(l.document.getElementById("streamButton").disabled,true);
     assert.equal(l.peers.length,0);assert.equal(l.calls.filter(c=>c.options.method==="DELETE").length,1);
   }
 },
 async barge_in_while_stream_response_pending() {
   const l=lab();await l.connect();const held=deferred();l.hooks.stream=()=>held.promise;
   const task=l.eval("streamAudio()");await until(()=>l.streamCalls().length===1);
   const reply=l.streamCalls()[0].options.body.get("turn_id");
   await l.eval('serializeProtocol(ctx=>sendEvent("user_speech_started",null,ctx))');const user=l.eval("turnId");assert.notEqual(user,reply);
   await l.eval('serializeProtocol(ctx=>sendEvent("assistant_turn_aborted",null,ctx))');
   assert.equal(l.eventCalls().length,3);assert.equal(JSON.parse(l.eventCalls()[2].options.body).turn_id,reply);
   assert.equal(l.eval("turnId"),user);assert.equal(l.streamCalls()[0].options.signal.aborted,true);
   assert.deepEqual(l.eventCalls().map(c=>JSON.parse(c.options.body).seq),[1,3,4]);
   await task; // The held upload response never resolves; abort still completes.
   assert.equal(l.document.getElementById("streamButton").disabled,false);
 },
 async abort_unreserved_hung_upload_is_not_blocked() {
   const l=lab();await l.connect();const held=deferred();
   l.hooks.stream=()=>{l.state.active_stream=null;return held.promise;};
   const task=l.eval("streamAudio()");await until(()=>l.streamCalls().length===1);
   const reply=l.streamCalls()[0].options.body.get("turn_id");
   await l.eval('serializeProtocol(ctx=>sendEvent("assistant_turn_aborted",null,ctx))');await task;
   assert.equal(l.streamCalls()[0].options.signal.aborted,true);
   assert.equal(JSON.parse(l.eventCalls()[1].options.body).turn_id,reply);
   assert.deepEqual(l.eventCalls().map(c=>JSON.parse(c.options.body).seq),[1,3]);
   assert.equal(l.state.pose_protocol.last_seq,3); // Delayed stream sequence 2 is now stale.
   assert.equal(l.document.getElementById("streamButton").disabled,false);
 },
 async user_b_fences_unreserved_upload_and_abort_preserves_b() {
   const l=lab();await l.connect();const held=deferred();
   l.hooks.stream=()=>{l.state.active_stream=null;return held.promise;};
   const task=l.eval("streamAudio()");await until(()=>l.streamCalls().length===1);
   const reply=l.streamCalls()[0].options.body.get("turn_id");
   await l.eval('serializeProtocol(ctx=>sendEvent("user_speech_started",null,ctx))');const user=l.eval("turnId");
   const result=await l.eval('serializeProtocol(ctx=>sendEvent("assistant_turn_aborted",null,ctx))');await task;
   assert.equal(result.reason,"pending_reply_already_superseded");assert.equal(l.eval("turnId"),user);
   assert.equal(l.state.pose_protocol.active_turn_id,user);assert.notEqual(user,reply);
   assert.equal(l.eventCalls().length,2);assert.equal(l.state.pose_protocol.last_seq,3);
   assert.equal(l.streamCalls()[0].options.signal.aborted,true);
 },
 async completed_reply_releases_fence_before_delayed_http_response() {
   const l=lab();await l.connect();const held=deferred();
   l.hooks.stream=call=>{l.state.active_stream=null;l.state.pose_protocol.active_turn_id=null;l.state.pose_protocol.last_seq=Number(call.options.body.get("seq"));return held.promise;};
   const task=l.eval("streamAudio()");await until(()=>l.streamCalls().length===1);
   await l.eval('serializeProtocol(ctx=>sendEvent("user_speech_started",null,ctx))');
   assert.equal(l.eventCalls().length,2);assert.equal(l.eval("pendingUpload.completed"),false);
   held.resolve(response({status:"streaming"}));await task;
 },
 async repeated_replies_and_explicit_smile_plan() {
   const l=lab();await l.connect();
   await l.eval("streamAudio()");assert.equal(l.document.getElementById("streamButton").disabled,true);
   l.state.active_stream=null;await l.eval("pollStatus()");
   l.document.getElementById("speechMovement").value="talking_smiling";await l.eval("streamAudio()");
   const [one,two]=l.streamCalls().map(c=>c.options.body);assert.notEqual(one.get("turn_id"),two.get("turn_id"));
   assert.deepEqual(JSON.parse(two.get("pose_plan")).segments,[{at_permille:0,pose_id:"speaking_direct"},{at_permille:350,pose_id:"light_smile"},{at_permille:700,pose_id:"speaking_direct"}]);
   assert.equal(two.get("pose_sequence"),undefined);assert.equal(two.get("mouth_mode"),"lip_sync");assert.equal(two.get("audio_start"),"immediate");
   assert.equal(two.get("seq"),"4");
 },
 async abort_during_local_audio_load_prevents_upload() {
   const l=lab();await l.connect();l.document.getElementById("audioFile").files=[];const held=deferred();l.hooks.audio=()=>held.promise;
   const task=l.eval("streamAudio()");await until(()=>l.calls.some(c=>c.url.endsWith("sample-audio")));
   l.eval('turnId="user-B"');l.state.pose_protocol.active_turn_id="user-B";
   l.events.find(e=>e.dataset.event==="assistant_turn_aborted").click();await task;
   await until(()=>!l.eval("cancellingUpload"));
   assert.equal(l.eval("turnId"),"user-B");assert.equal(l.eventCalls().length,0);
   assert.equal(l.streamCalls().length,0);assert.equal(l.document.getElementById("streamButton").disabled,false);
 },
 async cancel_during_preflight_preserves_queued_user_b_without_upload() {
   const l=lab();await l.connect();const held=deferred();
   l.hooks.event=call=>{const p=JSON.parse(call.options.body);l.state.pose_protocol.active_turn_id=p.turn_id;l.state.pose_protocol.last_seq=p.seq;return held.promise;};
   const task=l.eval("streamAudio()");await until(()=>l.eventCalls().length===1);
   const user=l.eval('serializeProtocol(ctx=>sendEvent("user_speech_started",null,ctx))');
   l.events.find(e=>e.dataset.event==="assistant_turn_aborted").click();l.hooks.event=null;held.resolve(response({accepted:true}));
   await task;await user;await until(()=>!l.eval("cancellingUpload"));
   assert.equal(l.streamCalls().length,0);assert.equal(l.eventCalls().length,2);
   assert.equal(l.eval("turnId"),l.state.pose_protocol.active_turn_id);
   assert.match(l.eval("turnId"),/^pose_lab_user_/);
 },
 async abort_stops_delayed_demo_steps() {
   const l=lab();await l.connect();const task=l.document.getElementById("demoButton").click();
   await until(()=>l.eventCalls().length===1);await l.eval('serializeProtocol(ctx=>sendEvent("assistant_turn_aborted",null,ctx))');
   await task;assert.deepEqual(l.eventCalls().map(c=>JSON.parse(c.options.body).event),["user_speech_started","assistant_turn_aborted"]);
   assert.equal(l.eval("turnId"),null);
 },
 async owned_abort_stops_demo_even_when_user_b_is_preserved() {
   const l=lab();await l.connect();const held=deferred();l.hooks.stream=()=>held.promise;
   const stream=l.eval("streamAudio()");await until(()=>l.streamCalls().length===1);
   const task=l.document.getElementById("demoButton").click();await until(()=>l.eventCalls().length===2);
   const user=l.eval("turnId");await l.eval('serializeProtocol(ctx=>sendEvent("assistant_turn_aborted",null,ctx))');
   await stream;await task;assert.equal(l.eval("turnId"),user);
   assert.deepEqual(l.eventCalls().map(c=>JSON.parse(c.options.body).event),["assistant_thinking","user_speech_started","assistant_turn_aborted"]);
 },
 async late_status_and_peer_callbacks_cannot_touch_new_session() {
   const l=lab();await l.connect();const old=l.peers[0];const oldStatus=deferred();l.hooks.status=()=>oldStatus.promise;
   const poll=l.eval("pollStatus()");await l.eval("endSession()");l.hooks.status=null;await l.connect();
   const newId=l.eval("sessionId");old.onconnectionstatechange();old.ontrack({track:{id:"old"}});old.onicecandidate({candidate:{candidate:"old"}});
   const stale=clone(l.state);stale.active_stream="old request";stale.pose_protocol.current_pose_id="light_smile";oldStatus.resolve(response(stale));await poll;
   assert.equal(l.eval("sessionId"),newId);assert.equal(l.document.getElementById("peerState").textContent,"connected");
   assert.equal(l.eval("activeStream"),false);assert.equal(l.calls.filter(c=>c.url.endsWith("/ice")).length,0);
   assert.equal(l.document.getElementById("remoteVideo").srcObject.getTracks().length,0);
 },
 async queued_event_cannot_migrate_to_new_session() {
   const l=lab();await l.connect();const held=deferred();l.box.hold=held.promise;
   const first=l.eval("serializeProtocol(()=>hold)");await Promise.resolve();
   const queued=l.eval('serializeProtocol(ctx=>sendEvent("assistant_thinking",null,ctx))').then(()=>null,error=>error);
   await l.eval("endSession()");await l.connect();held.resolve();await first;
   assert.match(String(await queued),/Session ended or changed/);assert.equal(l.eventCalls().length,0);assert.equal(l.eval("seq"),0);
 },
 async late_create_is_deleted_without_closing_new_session() {
   const l=lab();const held=deferred();l.hooks.all=c=>c.url.includes("/create?")?held.promise:undefined;
   const old=l.eval("createAndConnect()");await l.eval("endSession()");l.hooks.all=null;await l.connect();
   const current=l.eval("sessionId");held.resolve(response({session_id:"late-old"}));await old;
   assert.equal(l.eval("sessionId"),current);assert.equal(l.peers[0].connectionState,"connected");
   assert(l.calls.some(c=>c.url.endsWith("/late-old") && c.options.method==="DELETE"));
 },
 async missing_owner_and_rejected_abort_preserve_user_turn() {
   const l=lab();await l.connect();l.state.active_stream="owner";l.state.pose_protocol.active_turn_id=null;
   await assert.rejects(l.eval('sendEvent("assistant_turn_aborted")'),/no turn ID/);assert.equal(l.eventCalls().length,0);
   l.state.pose_protocol.active_turn_id="assistant-A";await l.eval('sendEvent("user_speech_started")');const user=l.eval("turnId");
   l.hooks.event=()=>response({accepted:false,reason:"turn_mismatch"});await l.eval('sendEvent("assistant_turn_aborted")');assert.equal(l.eval("turnId"),user);
 },
 async changed_bank_during_poll_ends_session() {
   const l=lab();await l.connect();l.state.track_stats.video.motion.bank.routing_sha256="other";await l.eval("pollStatus()");
   assert.equal(l.eval("sessionId"),null);assert.equal(l.document.getElementById("streamButton").disabled,true);
 },
};
(async()=>{await cases[input.case]();process.stdout.write(JSON.stringify({case:input.case,passed:true})+"\n");})().catch(error=>{console.error(error);process.exitCode=1;});
'''


@unittest.skipUnless(NODE, "Node is required to execute the actual browser client")
class PoseLabClientTests(unittest.TestCase):
    def run_client(self, case):
        page = get_webrtc_pose_lab_html(
            characters=[{"id": "japanese", "label": "Japanese"}],
            selected_character="japanese", expected_motion=EXPECTED)
        script = page.split("<script>", 1)[1].split("</script>", 1)[0]
        result = subprocess.run([NODE, "-e", HARNESS], input=json.dumps({"case": case, "script": script}),
                                text=True, capture_output=True, timeout=12)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["passed"])


def _install_case(name):
    def test(self):
        self.run_client(name)
    test.__name__ = "test_" + name
    setattr(PoseLabClientTests, test.__name__, test)


for _case in (
    "connect_verifies_bank_and_offer", "mismatched_bank_fails_closed",
    "barge_in_while_stream_response_pending", "abort_unreserved_hung_upload_is_not_blocked",
    "user_b_fences_unreserved_upload_and_abort_preserves_b",
    "repeated_replies_and_explicit_smile_plan", "abort_during_local_audio_load_prevents_upload",
    "completed_reply_releases_fence_before_delayed_http_response",
    "cancel_during_preflight_preserves_queued_user_b_without_upload",
    "abort_stops_delayed_demo_steps",
    "owned_abort_stops_demo_even_when_user_b_is_preserved",
    "late_status_and_peer_callbacks_cannot_touch_new_session", "queued_event_cannot_migrate_to_new_session",
    "late_create_is_deleted_without_closing_new_session", "missing_owner_and_rejected_abort_preserve_user_turn",
    "changed_bank_during_poll_ends_session",
):
    _install_case(_case)

if __name__ == "__main__":
    unittest.main()
