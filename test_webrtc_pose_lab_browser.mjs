// CPU-only driver/proof regressions: node --test test_webrtc_pose_lab_browser.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseArgs, assertBank, assertCompletedMotion, assertBargeProof, assertLiveBargeCandidate, assertCancelledMotionReturn, assertReceivedMedia, CDP }
  from './scripts/test_webrtc_pose_lab_browser.mjs';
const clone=structuredClone;
const expected={routing_sha256:'route',source_hashes:{neutral_resting:'idle',speaking_direct:'talk',light_smile:'smile'}};
const motion={enabled:true,settled:true,building:false,bank:{routing_sha256:'route',sources:Object.fromEntries(Object.entries(expected.source_hashes).map(([k,v])=>[k,{sha256:v}]))},
 trace:[{mode:'live',pose_id:'neutral_resting',generation_frame:0,output_frame:20},
 {mode:'live',pose_id:'speaking_direct',generation_frame:1,output_frame:21},
 {mode:'live',pose_id:'light_smile',generation_frame:2,output_frame:22}],
 entries:[{generation_id:2,first_live_output_frame:20,first_live_generation_frame:0,status:'completed'}],
 returns:[{source_output_frame:22,status:'completed',total_seconds:0.4}],last_emitted:{pose_id:'neutral_resting',output_frame:30}};
const status={active_stream:null,track_stats:{video:{motion}},pose_protocol:{active_turn_id:null,user_speaking:true}};
const proof={held:{body:{request_id:'request-A'},turnId:'assistant-A',networkId:'network-A',released:false},
 owner:{active_stream:'request-A',pose_protocol:{active_turn_id:'assistant-A',user_speaking:true,assistant_active:true}},
 started:{body:{accepted:true},payload:{event:'user_speech_started',turn_id:'user-B',seq:2}},
 aborted:{body:{accepted:true,pose_status:{user_speaking:true}},payload:{event:'assistant_turn_aborted',turn_id:'assistant-A',seq:3}},
 beforeRelease:status,pendingBeforeAbort:{dispatched:true,completed:false},uiAfterAbort:{pendingUpload:null},networkCancellation:{canceled:true,id:'network-A'},userTurnId:'user-B'};
const rtp=[{type:'codec',id:'codec',mimeType:'video/VP8'},
 {type:'inbound-rtp',kind:'video',codecId:'codec',framesDecoded:15,bytesReceived:123},
 {type:'inbound-rtp',kind:'audio',bytesReceived:99,packetsReceived:12,totalAudioEnergy:0.2}];
const frames=Array.from({length:15},(_,i)=>({mediaTime:i/20}));
test('CLI requires explicit executable and loopback origin, bounded sizes',()=>{
 const basic=['--browser','/tmp/browser','--character','japanese','--audio','/tmp/audio.wav','--output','/dev/shm/test'];
 assert.equal(parseArgs([...basic,'--barge-in']).bargeIn,true);
 for(const url of ['https://example.com','http://127.0.0.1/not-root','http://u:p@localhost:8000'])assert.throws(()=>parseArgs([...basic,'--base-url',url]));
 assert.throws(()=>parseArgs([...basic,'--max-recording-mib','999']));
 assert.throws(()=>parseArgs([...basic,'--character','../avatar']));
 assert.throws(()=>parseArgs(['--audio','--barge-in']));
});
test('motion hash binding rejects wrong source or routing',()=>{
 assertBank(status,expected);
 const wrong=clone(status);wrong.track_stats.video.motion.bank.sources.light_smile.sha256='different';assert.throws(()=>assertBank(wrong,expected));
 const wrongRouting=clone(expected);wrongRouting.routing_sha256='different';assert.throws(()=>assertBank(status,wrongRouting));
});
test('complete reply proves fresh generation, frame0, poses and bounded return',()=>{
 assert.equal(assertCompletedMotion(status,20,1).generationId,2);
 assert.throws(()=>assertCompletedMotion(status,20,2));
 for(const mutate of [m=>m.trace[0].generation_frame=1,m=>m.trace[1].generation_frame=9,
  m=>m.trace[2].pose_id='speaking_direct',m=>m.returns[0].total_seconds=0.501,m=>m.settled=false]){
  const bad=clone(status);mutate(bad.track_stats.video.motion);assert.throws(()=>assertCompletedMotion(bad,20,1));
 }
});
test('barge-in valid ownership while UI promise held',()=>assertBargeProof(proof));
test('barge-in rejects wrong assistant, response release, active server, lost user speech',()=>{
 for(const mutate of [p=>p.aborted.payload.turn_id='user-B',p=>p.pendingBeforeAbort.completed=true,
  p=>p.beforeRelease.active_stream='request-A',p=>p.owner.pose_protocol.active_turn_id='user-B',
  p=>p.aborted.body.pose_status.user_speaking=false,p=>p.started.body.accepted=false,
  p=>p.held.released=true,p=>p.networkCancellation.id='another-request',p=>p.uiAfterAbort.pendingUpload={}]){
  const bad=clone(proof);mutate(bad);assert.throws(()=>assertBargeProof(bad));
 }
});
const bargeRows=Array.from({length:25},(_,generation_frame)=>({mode:'live',pose_id:'speaking_direct',
 generation_frame,source_frame:100+generation_frame,output_frame:200+generation_frame}));
const liveCandidate={status:{fps:20,active_stream:'request-A',pose_protocol:{active_turn_id:'assistant-A'},
 track_stats:{video:{live_generation_id:3,motion:{enabled:true,settled:true,building:false,
 entries:[{generation_id:2,first_live_output_frame:20,status:'completed'},
  {generation_id:3,first_live_output_frame:200,status:'completed'}],
 trace:bargeRows,last_emitted:bargeRows.at(-1),
 returns:[{started_at:100,source_output_frame:50,status:'completed',total_seconds:0.4}]}}}},
 firstOutputFrame:180,previousGeneration:2,requestId:'request-A',turnId:'assistant-A',
 receiverStart:{generationId:3,count:100,mediaTime:5},receiverNow:{count:123,mediaTime:6.15}};
function cancelledFixture(){
 const before=clone(liveCandidate.status),after=clone(before);
 after.active_stream=null;
 after.track_stats.video.motion.returns.push({started_at:200,source_output_frame:224,
  from_pose:'speaking_direct',from_frame:124,status:'completed',total_seconds:0.4});
 after.track_stats.video.motion.last_emitted={pose_id:'neutral_resting',output_frame:232};
 return {before,after,liveProof:assertLiveBargeCandidate(liveCandidate)};
}
test('live barge candidate binds current generation, speech duration and receiver advancement',()=>{
 const p=assertLiveBargeCandidate(liveCandidate);
 assert.equal(p.generationId,3);assert.equal(p.emittedSpeechSeconds,1.2);
 assert.equal(p.lastLive.output_frame,224);
});
test('queued-only or short speech cannot count as a live barge-in',()=>{
 for(const mutate of [c=>c.status.track_stats.video.motion.entries.pop(),
  c=>c.status.track_stats.video.motion.last_emitted.mode='speech_entry_body_bridge',
  c=>c.status.track_stats.video.motion.trace.splice(10),
  c=>c.status.track_stats.video.motion.last_emitted.pose_id='neutral_resting',
  c=>c.status.active_stream=null]){
  const c=clone(liveCandidate);mutate(c);assert.throws(()=>assertLiveBargeCandidate(c));
 }
 const short=clone(liveCandidate);short.status.track_stats.video.motion.trace=short.status.track_stats.video.motion.trace.slice(0,10);
 short.status.track_stats.video.motion.last_emitted=short.status.track_stats.video.motion.trace.at(-1);
 assert.throws(()=>assertLiveBargeCandidate(short),/one second/);
});
test('live barge rejects wrong generation, frame restart and frozen browser presentation',()=>{
 for(const mutate of [c=>c.status.track_stats.video.live_generation_id=2,
  c=>c.previousGeneration=3,c=>c.receiverStart.generationId=2,
  c=>c.status.track_stats.video.motion.trace[12].generation_frame=0,
  c=>c.receiverNow.count=101,c=>c.receiverNow.mediaTime=5.4]){
  const c=clone(liveCandidate);mutate(c);assert.throws(()=>assertLiveBargeCandidate(c));
 }
});
test('cancelled return binds a new bounded return to exact emitted speech anchor',()=>{
 const {before,after,liveProof}=cancelledFixture();
 const p=assertCancelledMotionReturn(before,after,liveProof);
 assert.equal(p.generationId,3);assert.equal(p.source.output_frame,224);assert.equal(p.return.total_seconds,0.4);
});
test('cancelled return rejects no-op cancellation, reused return and wrong generation',()=>{
 for(const mutate of [f=>f.after.track_stats.video.motion.returns.pop(),
  f=>f.before.track_stats.video.motion.returns.push(clone(f.after.track_stats.video.motion.returns.at(-1))),
  f=>f.after.track_stats.video.motion.returns.at(-1).source_output_frame=50,
  f=>f.after.track_stats.video.motion.entries.at(-1).generation_id=4,
  f=>f.after.track_stats.video.motion.entries.push({generation_id:4,first_live_output_frame:225}),
  f=>f.after.active_stream='request-A']){
  const f=cancelledFixture();mutate(f);assert.throws(()=>assertCancelledMotionReturn(f.before,f.after,f.liveProof));
 }
});
test('cancelled return rejects slow, incomplete or incorrectly bound source anchors',()=>{
 for(const mutate of [f=>f.after.track_stats.video.motion.returns.at(-1).total_seconds=0.501,
  f=>f.after.track_stats.video.motion.returns.at(-1).status='building',
  f=>f.after.track_stats.video.motion.returns.at(-1).from_frame=123,
  f=>f.after.track_stats.video.motion.returns.at(-1).from_pose='light_smile',
  f=>f.after.track_stats.video.motion.trace.at(-1).mode='idle',
  f=>f.after.track_stats.video.motion.last_emitted.pose_id='speaking_direct']){
  const f=cancelledFixture();mutate(f);assert.throws(()=>assertCancelledMotionReturn(f.before,f.after,f.liveProof));
 }
});
test('actual inbound VP8/audio/presentation proof rejects H264, silence and frozen time',()=>{
 assert.equal(assertReceivedMedia(rtp,frames).presentedCallbacks,15);
 const h264=clone(rtp);h264[0].mimeType='video/H264';assert.throws(()=>assertReceivedMedia(h264,frames));
 const silence=clone(rtp);silence[2].totalAudioEnergy=0;assert.throws(()=>assertReceivedMedia(silence,frames));
 const frozen=clone(frames);frozen[7].mediaTime=frozen[6].mediaTime;assert.throws(()=>assertReceivedMedia(rtp,frozen));
});
class Socket extends EventTarget {sent=[];send(x){this.sent.push(JSON.parse(x));}close(){this.dispatchEvent(new Event('close'));}receive(value){this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify(value)}));}}
test('CDP correlates out-of-order responses and surfaces protocol errors',async()=>{
 const socket=new Socket(),cdp=new CDP(socket);const a=cdp.send('one'),b=cdp.send('two');
 socket.receive({id:2,result:{ok:2}});socket.receive({id:1,result:{ok:1}});
 assert.deepEqual(await a,{ok:1});assert.deepEqual(await b,{ok:2});
 const error=cdp.send('bad');socket.receive({id:3,error:{message:'unsupported'}});await assert.rejects(error,/unsupported/);cdp.close();
});
test('CDP timeout/close invalidates waiters and ignores late old responses',async()=>{
 const socket=new Socket(),cdp=new CDP(socket);
 await assert.rejects(cdp.send('stalled',{},5),/timed out/);socket.receive({id:1,result:{late:true}});
 const pending=cdp.send('waiting');cdp.close();await assert.rejects(pending,/closed/);assert.equal(cdp.pending.size,0);
});
test('CDP evaluate wraps browser exception as failure',async()=>{
 const socket=new Socket(),cdp=new CDP(socket);const p=cdp.evaluate('throw');
 socket.receive({id:1,result:{exceptionDetails:{text:'badJS'}}});await assert.rejects(p,/badJS/);cdp.close();
});
