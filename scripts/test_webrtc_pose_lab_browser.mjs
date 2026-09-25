#!/usr/bin/env node
/**
 * Real Chromium / pose-lab UI integration, using only Node 22 built-ins.
 *
 * --browser is an explicitly selected executable or launcher (no shell expansion).
 * This script does not install/restore a browser. A root container may require the
 * explicit --no-sandbox flag; obtain any required execution approval first.
 *
 * Example:
 * node scripts/test_webrtc_pose_lab_browser.mjs --browser /path/to/launcher \
 *   --base-url http://127.0.0.1:8000 --character japanese_20260925 \
 *   --audio /path/to/long.wav --output /dev/shm/pose-lab-new-evidence --barge-in
 * Keep evidence in owned shared memory when executable staging leaves little
 * overlay space; copy evidence to /workspace only after that tooling is removed.
 *
 * received.webm is MediaRecorder's browser re-encode of received audio/video.
 * Presentation callbacks and inbound RTP stats are saved separately. This is not
 * a packet-preserving capture, a source-frame identity proof, or visual approval.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createReadStream } from 'node:fs';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SELF = fileURLToPath(import.meta.url);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const motionOf = status => status?.track_stats?.video?.motion;
const settled = status => !status.active_stream && motionOf(status)?.settled && !motionOf(status)?.building;

export function parseArgs(argv) {
  const result = { baseUrl: 'http://127.0.0.1:8000', timeoutSeconds: 180, maxRecordingMiB: 12 };
  const names = { '--browser': 'browser', '--base-url': 'baseUrl', '--character': 'character',
    '--audio': 'audio', '--output': 'output', '--timeout-seconds': 'timeoutSeconds',
    '--max-recording-mib': 'maxRecordingMiB' };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === '--barge-in') result.bargeIn = true;
    else if (flag === '--no-sandbox') result.noSandbox = true;
    else if (flag === '--help') result.help = true;
    else {
      assert(names[flag], `Unknown option: ${flag}`);
      assert(argv[i + 1] && !argv[i + 1].startsWith('--'), `Missing value for ${flag}`);
      result[names[flag]] = argv[++i];
    }
  }
  if (result.help) return result;
  for (const name of ['browser', 'character', 'audio', 'output']) assert(result[name], `--${name} is required`);
  const base = new URL(result.baseUrl);
  assert(['http:', 'https:'].includes(base.protocol) &&
    ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname) &&
    !base.username && !base.password && base.pathname === '/' && !base.search && !base.hash,
  '--base-url must be a loopback HTTP(S) origin');
  result.baseUrl = base.origin;
  assert(/^[A-Za-z0-9_-]{1,128}$/.test(result.character), 'Character must be a registry ID, not a path');
  for (const name of ['browser', 'audio', 'output']) result[name] = path.resolve(result[name]);
  result.timeoutSeconds = Number(result.timeoutSeconds);
  result.maxRecordingMiB = Number(result.maxRecordingMiB);
  assert(Number.isFinite(result.timeoutSeconds) && result.timeoutSeconds >= 15 && result.timeoutSeconds <= 300);
  assert(Number.isFinite(result.maxRecordingMiB) && result.maxRecordingMiB >= 1 && result.maxRecordingMiB <= 32);
  return result;
}

export function assertBank(status, expected) {
  const motion = motionOf(status);
  assert(expected?.routing_sha256 && Object.keys(expected.source_hashes || {}).length === 3,
    'The selected page must declare a three-source motion bank');
  assert(motion?.enabled && motion.bank, 'Session has no enabled motion bank');
  assert.equal(motion.bank.routing_sha256, expected.routing_sha256, 'Routing hash mismatch');
  for (const [pose, hash] of Object.entries(expected.source_hashes)) {
    assert.equal(motion.bank.sources?.[pose]?.sha256, hash, `Source hash mismatch: ${pose}`);
  }
  return motion;
}

export function assertCompletedMotion(status, firstOutputFrame, previousGeneration = null) {
  assert(settled(status), 'Speech did not settle back to idle');
  const motion = motionOf(status);
  const rows = motion.trace.filter(row => row.mode === 'live' && row.output_frame >= firstOutputFrame);
  assert(rows.length > 0, 'No live frames were emitted for this reply');
  assert.equal(rows[0].generation_frame, 0, 'First phoneme frame was skipped');
  for (let index = 1; index < rows.length; index++) {
    assert.equal(rows[index].generation_frame, rows[index - 1].generation_frame + 1,
      'Generated frame sequence skipped or restarted within the reply');
  }
  for (const pose of ['speaking_direct', 'light_smile']) {
    assert(rows.some(row => row.pose_id === pose), `Long reply never displayed ${pose}`);
  }
  assert.equal(motion.last_emitted.pose_id, 'neutral_resting');
  const entries = motion.entries.filter(entry => entry.first_live_output_frame >= firstOutputFrame);
  assert.equal(entries.length, 1, 'Reply must own exactly one successful speech entry');
  const entry = entries[0];
  assert.equal(entry.status, 'completed');
  assert.equal(entry.first_live_generation_frame, 0);
  if (previousGeneration !== null) assert(entry.generation_id > previousGeneration, 'Old generation reused');
  const returns = motion.returns.filter(item => item.source_output_frame >= firstOutputFrame);
  assert(returns.length > 0 && returns.every(item => item.status === 'completed' && item.total_seconds <= 0.5),
    'Return failed or exceeded the existing0.5s bound');
  return { generationId: entry.generation_id, firstLive: rows[0], lastLive: rows.at(-1),
    poses: [...new Set(rows.map(row => row.pose_id))], returns };
}

export function assertBargeProof({ held, owner, started, aborted, beforeRelease, pendingBeforeAbort, uiAfterAbort, networkCancellation, userTurnId }) {
  assert(held.body.request_id && held.turnId, 'Held response does not identify its request and assistant');
  assert.equal(owner.active_stream, held.body.request_id);
  assert.equal(owner.pose_protocol.active_turn_id, held.turnId);
  assert(owner.pose_protocol.user_speaking && owner.pose_protocol.assistant_active);
  assert(started.body.accepted && aborted.body.accepted, 'UI events were rejected');
  assert.equal(started.payload.event, 'user_speech_started');
  assert.equal(started.payload.turn_id, userTurnId);
  assert.notEqual(userTurnId, held.turnId, 'Barge-in did not create a distinct user turn');
  assert.equal(aborted.payload.event, 'assistant_turn_aborted');
  assert.equal(aborted.payload.turn_id, held.turnId, 'Abort targeted the user instead of active assistant');
  assert(aborted.payload.seq > started.payload.seq);
  assert.equal(aborted.body.pose_status?.user_speaking, true, 'Abort cleared ongoing user speech');
  assert(settled(beforeRelease), 'Server did not cancel and settle while POST response remained held');
  assert.equal(beforeRelease.pose_protocol.active_turn_id, null);
  assert.equal(beforeRelease.pose_protocol.user_speaking, true);
  assert(pendingBeforeAbort?.dispatched && !pendingBeforeAbort.completed,
    'Browser upload promise was not pending before abort');
  assert.equal(held.released, false, 'Held response was released before cancellation proof');
  assert.equal(uiAfterAbort.pendingUpload, null, 'AbortController did not settle the pending UI upload');
  assert(networkCancellation?.canceled && networkCancellation.id === held.networkId,
    'No network cancellation for the held browser fetch');
}

// A queued HTTP response alone is insufficient: cancellation must happen after
// this reply has emitted speech and the browser has continued presenting media.
// Receiver callbacks are independent advancement evidence, not exact RTP-to-source mapping.
export function assertLiveBargeCandidate({ status, firstOutputFrame, previousGeneration,
  requestId, turnId, receiverStart, receiverNow }) {
  assert.equal(status.active_stream, requestId, 'Cancelled reply no longer owns the stream');
  assert.equal(status.pose_protocol.active_turn_id, turnId, 'Cancelled reply lost its assistant identity');
  const motion = motionOf(status);
  const entries = motion.entries.filter(entry => entry.first_live_output_frame >= firstOutputFrame);
  assert.equal(entries.length, 1, 'Barge-in must target exactly one new speech generation');
  const entry = entries[0];
  assert.equal(entry.status, 'completed', 'Speech entry has not completed');
  assert(entry.generation_id > previousGeneration, 'Barge-in reused an older generation');
  assert.equal(status.track_stats.video.live_generation_id, entry.generation_id,
    'Displayed speech belongs to a different generation');
  const rows = motion.trace.filter(row => row.mode === 'live' && row.output_frame >= entry.first_live_output_frame);
  assert(rows.length > 0 && rows[0].generation_frame === 0, 'Barge-in has no fresh frame-zero speech');
  for (let index = 1; index < rows.length; index++) {
    assert.equal(rows[index].generation_frame, rows[index - 1].generation_frame + 1,
      'Speech skipped or restarted before barge-in');
  }
  const last = rows.at(-1);
  assert.equal(motion.last_emitted.mode, 'live', 'Barge-in missed active speech');
  assert.equal(motion.last_emitted.output_frame, last.output_frame);
  assert(['speaking_direct', 'light_smile'].includes(last.pose_id), 'Barge-in did not occur on talking/smiling');
  assert(Number.isFinite(status.fps) && status.fps > 0, 'Missing generation frame rate');
  assert(last.generation_frame / status.fps >= 1, 'Less than one second of this reply was emitted');
  assert.equal(receiverStart.generationId, entry.generation_id, 'Receiver baseline belongs to another generation');
  assert(receiverNow.count - receiverStart.count >= 10 &&
    Number.isFinite(receiverStart.mediaTime) && Number.isFinite(receiverNow.mediaTime) &&
    receiverNow.mediaTime - receiverStart.mediaTime >= 1,
  'Browser presentation did not advance for one second during this reply');
  return { generationId: entry.generation_id, firstLive: rows[0], lastLive: last,
    emittedSpeechSeconds: last.generation_frame / status.fps,
    receiverStart, receiverNow, firstOutputFrame };
}

export function assertCancelledMotionReturn(beforeAbort, beforeRelease, liveProof) {
  assert(settled(beforeRelease), 'Cancelled speech did not settle');
  const before = motionOf(beforeAbort), after = motionOf(beforeRelease);
  const entries = after.entries.filter(entry => entry.first_live_output_frame >= liveProof.firstOutputFrame);
  assert.equal(entries.length, 1, 'Cancellation acquired another speech generation');
  assert.equal(entries[0].generation_id, liveProof.generationId, 'Cancellation return belongs to another generation');
  const oldReturnKeys = new Set(before.returns.map(row => `${row.started_at}:${row.source_output_frame}`));
  const returns = after.returns.filter(row =>
    row.source_output_frame >= before.last_emitted.output_frame &&
    !oldReturnKeys.has(`${row.started_at}:${row.source_output_frame}`));
  assert.equal(returns.length, 1, 'Cancellation must create one new return from the currently displayed speech');
  const result = returns[0];
  assert.equal(result.status, 'completed');
  assert(Number.isFinite(result.total_seconds) && result.total_seconds >= 0 && result.total_seconds <= 0.5,
    'Cancelled speech return exceeded the 0.5s bound');
  const source = after.trace.find(row => row.output_frame === result.source_output_frame);
  assert(source?.mode === 'live' && source.output_frame >= liveProof.firstLive.output_frame,
    'Cancellation return did not start from this generation\'s emitted speech');
  assert.equal(result.from_pose, source.pose_id, 'Return pose does not match its emitted anchor');
  assert.equal(result.from_frame, source.source_frame, 'Return source frame does not match its emitted anchor');
  assert(['speaking_direct', 'light_smile'].includes(source.pose_id), 'Cancellation missed the expressive pose');
  assert.equal(after.last_emitted.pose_id, 'neutral_resting');
  return { generationId: liveProof.generationId, source, return: result };
}

export function assertReceivedMedia(stats, callbacks) {
  const incoming = stats.filter(row => row.type === 'inbound-rtp');
  const videos = incoming.filter(row => row.kind === 'video' || row.mediaType === 'video');
  const audios = incoming.filter(row => row.kind === 'audio' || row.mediaType === 'audio');
  assert(videos.some(row => row.framesDecoded > 10 && row.bytesReceived > 0), 'No decoded inbound video');
  for (const row of videos.filter(row => row.bytesReceived > 0)) {
    const codec = stats.find(codec => codec.id === row.codecId);
    assert.equal(codec?.mimeType?.toLowerCase(), 'video/vp8', 'Browser did not negotiate VP8');
  }
  assert(audios.some(row => row.bytesReceived > 0 && row.packetsReceived > 0), 'No received audio packets');
  assert(audios.some(row => row.totalAudioEnergy > 0), 'Received audio stayed silent during the speech test');
  assert(callbacks.length > 10, 'Video element did not present enough frames');
  for (let index = 1; index < callbacks.length; index++) {
    assert(callbacks[index].mediaTime > callbacks[index - 1].mediaTime, 'Presentation media time did not advance');
  }
  return { presentedCallbacks: callbacks.length,
    decodedFrames: videos.reduce((sum, row) => sum + (row.framesDecoded || 0), 0),
    receivedVideoBytes: videos.reduce((sum, row) => sum + (row.bytesReceived || 0), 0),
    receivedAudioBytes: audios.reduce((sum, row) => sum + (row.bytesReceived || 0), 0) };
}

/** A small CDP transport with correlated replies, bounded waits and owned close. */
export class CDP {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 0;
    this.pending = new Map();
    this.listeners = new Map();
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const entry = this.pending.get(message.id);
        if (!entry) return;
        clearTimeout(entry.timer);
        this.pending.delete(message.id);
        if (message.error) entry.reject(new Error(`${entry.method}: ${JSON.stringify(message.error)}`));
        else entry.resolve(message.result);
      } else for (const listener of this.listeners.get(message.method) || []) listener(message.params);
    });
    socket.addEventListener('close', () => this.rejectPending(new Error('CDP socket closed')));
    socket.addEventListener('error', () => this.rejectPending(new Error('CDP socket error')));
  }
  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { socket.close(); reject(new Error('CDP connect timed out')); }, 10000);
      socket.addEventListener('open', () => { clearTimeout(timer); resolve(); }, { once: true });
      socket.addEventListener('error', () => { clearTimeout(timer); reject(new Error('CDP connect failed')); }, { once: true });
    });
    return new CDP(socket);
  }
  on(method, listener) {
    if (!this.listeners.has(method)) this.listeners.set(method, []);
    this.listeners.get(method).push(listener);
  }
  send(method, params = {}, timeoutMs = 10000) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`${method} timed out`)); }, timeoutMs);
      this.pending.set(id, { method, timer, resolve, reject });
      try { this.socket.send(JSON.stringify({ id, method, params })); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }
  async evaluate(expression, timeoutMs = 10000) {
    const response = await this.send('Runtime.evaluate', { expression, awaitPromise: true,
      returnByValue: true, userGesture: true }, timeoutMs);
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
  rejectPending(error) {
    for (const item of this.pending.values()) { clearTimeout(item.timer); item.reject(error); }
    this.pending.clear();
  }
  close() { this.rejectPending(new Error('CDP client closed')); this.socket.close(); }
}

async function sha256(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}
async function jsonFetch(url, options = {}) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(5000) });
  assert(response.ok, `HTTP ${response.status}: ${url}`);
  return response.json();
}
function compact(value) {
  return JSON.parse(JSON.stringify(value, (key, item) =>
    ['trace', 'rendered_pose_trace', 'decoders'].includes(key) ? undefined : item));
}
async function wavInfo(filename) {
  const stat = await fs.stat(filename);
  assert(stat.isFile() && stat.size <= 16 * 1024 * 1024, 'WAV must be a file of at most16MiB');
  const data = await fs.readFile(filename);
  assert.equal(data.toString('ascii', 0, 4), 'RIFF');
  assert.equal(data.toString('ascii', 8, 12), 'WAVE');
  let byteRate, audioBytes;
  for (let offset = 12; offset + 8 <= data.length;) {
    const size = data.readUInt32LE(offset + 4), kind = data.toString('ascii', offset, offset + 4);
    assert(offset + 8 + size <= data.length, 'Truncated WAV chunk');
    if (kind === 'fmt ' && size >= 16) byteRate = data.readUInt32LE(offset + 16);
    if (kind === 'data') audioBytes = size;
    offset += 8 + size + (size & 1);
  }
  const seconds = audioBytes / byteRate;
  assert(seconds >= 8 && seconds <= 30, 'Use an8–30second speech WAV to exercise talking/smiling');
  return { path: filename, bytes: stat.size, sha256: await sha256(filename), seconds };
}

async function main(args) {
  const audio = await wavInfo(args.audio);
  assert((await fs.stat(args.browser)).isFile(), '--browser must explicitly name an executable or launcher');
  await fs.mkdir(args.output); // Fail rather than overwrite or mix evidence from another run.
  const profile = await fs.mkdtemp('/dev/shm/musetalk-pose-lab-cdp-');
  const evidence = { version: 1, startedAt: new Date().toISOString(), success: false,
    args, audio, node: process.version, platform: `${os.platform()} ${os.release()} ${os.arch()}`,
    harnessSha256: await sha256(SELF), launcherSha256: await sha256(args.browser),
    profile, statuses: [], stats: [], network: [], console: [], exceptions: [], checkpoints: [], cleanup: [],
    limitations: ['Browser MediaRecorder re-encodes received media; WebM does not preserve source/RTP timestamps.',
      'Video presentation callbacks and receiver stats are independent diagnostics, not sender-frame identity mapping.',
      'Functional assertions do not constitute normal-speed visual approval or multi-session capacity validation.'] };
  let browser, cdp, browserCdp, sid, held, sessionReady = false, asynchronousError;
  let closing = false, interrupted = false, consoleBytes = 0, recordingBytes = 0, chunkIndex = 0;
  let statusBytes = 0;
  let recordingWrites = Promise.resolve();
  const responseTasks = new Set(), requests = new Map();
  const started = performance.now();
  const elapsed = () => (performance.now() - started) / 1000;
  const save = (name, value) => fs.writeFile(path.join(args.output, name), `${JSON.stringify(value, null, 2)}\n`);
  const failAsync = error => { asynchronousError ||= error; };
  const signal = () => { interrupted = true; };
  process.on('SIGINT', signal); process.on('SIGTERM', signal);
  async function until(label, check, timeoutSeconds = args.timeoutSeconds) {
    const end = performance.now() + timeoutSeconds * 1000;
    while (performance.now() < end) {
      if (asynchronousError) throw asynchronousError;
      if (interrupted) throw new Error('Interrupted');
      if (browser && (browser.exitCode !== null || browser.signalCode !== null)) {
        throw new Error(`Browser exited: ${browser.exitCode ?? browser.signalCode}`);
      }
      const result = await check();
      if (result) return result;
      await sleep(100);
    }
    throw new Error(`${label} timed out after${timeoutSeconds}s`);
  }
  async function click(selector) {
    const point = await cdp.evaluate(`(() => {
      const element = document.querySelector(${JSON.stringify(selector)});
      if (!element || element.disabled) throw new Error('Missing/disabled control: '+${JSON.stringify(selector)});
      element.scrollIntoView({block:'center'}); const r=element.getBoundingClientRect();
      return {x:r.x+r.width/2,y:r.y+r.height/2}; })()`);
    await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', button: 'left', clickCount: 1, ...point });
    await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', button: 'left', clickCount: 1, ...point });
  }
  const ui = () => cdp.evaluate(`({ sessionId, sessionEpoch, turnId, seq,
    pendingUpload: pendingUpload ? {...pendingUpload} : null, activeStream, motionReady, motionSettled,
    peer: pc?.connectionState, streamDisabled: streamButton.disabled,
    video: {readyState:remoteVideo.readyState,paused:remoteVideo.paused,currentTime:remoteVideo.currentTime,
      videoWidth:remoteVideo.videoWidth,videoHeight:remoteVideo.videoHeight},
    bankState:document.getElementById('bankState').textContent })`);
  async function status() {
    const value = await jsonFetch(`${args.baseUrl}/webrtc/sessions/${encodeURIComponent(sid)}/status`);
    if (evidence.expectedMotion) assertBank(value, evidence.expectedMotion);
    const sample = { atSeconds: elapsed(), value: compact(value) };
    statusBytes += Buffer.byteLength(JSON.stringify(sample));
    assert(statusBytes <= 8 * 1024 * 1024, 'Status evidence exceeded8MiB; failing rather than dropping proof');
    evidence.statuses.push(sample);
    return value;
  }
  async function checkpoint(name, fullStatus = null) {
    const stats = await cdp.evaluate(`(async()=>pc ? Array.from((await pc.getStats()).values(), x=>({...x})) : [])()`);
    const item = { name, atSeconds: elapsed(), ui: await ui(), status: fullStatus || await status(), stats };
    evidence.checkpoints.push(item); evidence.stats.push({ atSeconds: elapsed(), value: stats });
    return item;
  }
  async function eventFromButton(name) {
    const start = evidence.network.length;
    await click(`button[data-event="${name}"]`);
    return until(`UI event ${name}`, async () => evidence.network.slice(start).find(row =>
      row.kind === 'response' && row.url.endsWith('/events') && row.payload?.event === name && row.body), 15);
  }
  async function submitAndComplete(label, previousGeneration = null) {
    await until('Stream button enabled', async () => !(await ui()).streamDisabled, 15);
    const before = await status();
    const firstOutputFrame = (motionOf(before).last_emitted?.output_frame ?? -1) + 1;
    const responseStart = evidence.network.length;
    await click('#streamButton');
    const reply = await until(`${label} response`, async () => evidence.network.slice(responseStart).find(row =>
      row.kind === 'response' && row.url.endsWith('/stream') && row.body?.request_id));
    assert.equal(reply.status, 200, 'Stream HTTP request failed');
    const turnId = reply.body.pose_plan?.turn_id || reply.body.pose_sequence?.turn_id;
    assert(turnId, 'Stream response has no assistant turn ID');
    const final = await until(`${label} completed idle`, async () => {
      const current = await status();
      const live = motionOf(current).trace.some(row => row.mode === 'live' && row.output_frame >= firstOutputFrame);
      return live && settled(current) && !(await ui()).pendingUpload ? current : null;
    });
    const proof = assertCompletedMotion(final, firstOutputFrame, previousGeneration);
    await checkpoint(label, final);
    return { label, firstOutputFrame, requestId: reply.body.request_id, turnId, proof };
  }
  const receiverPresentation = () => cdp.evaluate(`(() => {
    const frames = window.__poseLabEvidence.frames;
    return { count: frames.length, mediaTime: frames.at(-1)?.mediaTime ?? null,
      observedDateMs: frames.at(-1)?.observedDateMs ?? null };
  })()`);
  async function heldResponseBargeIn() {
    await until('Stream button enabled before barge-in', async () => !(await ui()).streamDisabled, 15);
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: `${args.baseUrl}/webrtc/sessions/*/stream`, requestStage: 'Response' }] });
    cdp.on('Fetch.requestPaused', event => {
      if (closing) { cdp.send('Fetch.continueResponse', { requestId: event.requestId }).catch(() => {}); return; }
      if (held) { failAsync(new Error('Unexpected second intercepted stream response')); return; }
      held = { requestId: event.requestId, status: event.responseStatusCode,
        networkId: event.networkId, url: event.request.url, atSeconds: elapsed(), released: false };
    });
    const before = await checkpoint('before-held-stream');
    const firstOutputFrame = (motionOf(before.status).last_emitted?.output_frame ?? -1) + 1;
    const previousGeneration = Math.max(0, ...motionOf(before.status).entries.map(entry => entry.generation_id));
    await click('#streamButton');
    await until('Intercepted stream response', async () => held, 30);
    assert.equal(held.status, 200, 'Held stream failed');
    const body = await cdp.send('Fetch.getResponseBody', { requestId: held.requestId });
    held.body = JSON.parse(body.base64Encoded ? Buffer.from(body.body, 'base64').toString() : body.body);
    const pending = (await ui()).pendingUpload;
    assert(pending?.dispatched && !pending.completed, 'UI is not awaiting held POST response');
    held.turnId = pending.turnId;
    const reservation = await status();
    assert.equal(reservation.active_stream, held.body.request_id, 'Response held after request already completed');
    assert.equal(reservation.pose_protocol.active_turn_id, held.turnId);
    let receiverStart;
    const live = await until('One second of live expressive speech while response is held', async () => {
      const current = await status();
      assert.equal(current.active_stream, held.body.request_id, 'Held reply completed before live barge-in');
      assert.equal(current.pose_protocol.active_turn_id, held.turnId);
      const motion = motionOf(current);
      const entries = motion.entries.filter(entry => entry.first_live_output_frame >= firstOutputFrame);
      const emitted = motion.last_emitted;
      if (!entries.length || emitted?.mode !== 'live') return null;
      assert.equal(entries.length, 1, 'Unexpected second generation during held response');
      const receiverNow = await receiverPresentation();
      receiverStart ||= { ...receiverNow, generationId: entries[0].generation_id };
      if (emitted.generation_frame / current.fps < 1 ||
          !['speaking_direct', 'light_smile'].includes(emitted.pose_id) ||
          receiverNow.count - receiverStart.count < 10 ||
          receiverNow.mediaTime - receiverStart.mediaTime < 1) return null;
      const proof = assertLiveBargeCandidate({ status: current, firstOutputFrame, previousGeneration,
        requestId: held.body.request_id, turnId: held.turnId, receiverStart, receiverNow });
      return { checkpoint: await checkpoint('held-response-live-before-barge', current), proof };
    }, 30);
    const startedEvent = await eventFromButton('user_speech_started');
    const owner = await status();
    // The user event must not have cancelled or replaced the assistant. Capture
    // its last emitted anchor immediately before the explicit abort click.
    const liveProof = assertLiveBargeCandidate({ status: owner, firstOutputFrame, previousGeneration,
      requestId: held.body.request_id, turnId: held.turnId, receiverStart,
      receiverNow: await receiverPresentation() });
    const userTurnId = startedEvent.payload.turn_id;
    const pendingBeforeAbort = (await ui()).pendingUpload;
    const abortedEvent = await eventFromButton('assistant_turn_aborted');
    const beforeRelease = await until('Cancellation before releasing stream response', async () => {
      const current = await status(); return settled(current) ? current : null;
    }, 15);
    const uiAfterAbort = await until('AbortController settles held upload', async () => {
      const current = await ui(); return !current.pendingUpload ? current : null;
    }, 10);
    const networkCancellation = await until('Held browser fetch cancellation', async () =>
      evidence.network.find(row => row.kind === 'loading-failed' && row.id === held.networkId && row.canceled), 10);
    assertBargeProof({ held, owner, started: startedEvent, aborted: abortedEvent,
      beforeRelease, pendingBeforeAbort, uiAfterAbort, networkCancellation, userTurnId });
    const cancelledReturn = assertCancelledMotionReturn(owner, beforeRelease, liveProof);
    const oldGeneration = liveProof.generationId;
    evidence.bargeIn = { before, held: { ...held }, reservation, live, liveProof, cancelledReturn,
      started: startedEvent, owner, aborted: abortedEvent, beforeRelease, pendingBeforeAbort,
      uiAfterAbort, networkCancellation, userTurnId, oldGeneration };
    evidence.bargeIn.releaseAttemptAtSeconds = elapsed();
    try {
      await cdp.send('Fetch.continueResponse', { requestId: held.requestId });
      held.released = true;
      evidence.bargeIn.releasedAtSeconds = elapsed();
    } catch (error) {
      // AbortController may invalidate the intercepted response before CDP can
      // release it. Only accept this after proving that exact network request
      // was cancelled and the server independently returned to settled idle.
      held.cancelledByBrowser = true;
      evidence.bargeIn.releaseAfterCancellationError = String(error);
    }
    await cdp.send('Fetch.disable');
    await until('Held UI promise completed', async () => !(await ui()).pendingUpload, 15);
    const ended = await eventFromButton('user_speech_ended');
    assert(ended.body.accepted && ended.payload.turn_id === userTurnId, 'User end lost its turn ID');
    assert(ended.payload.seq > abortedEvent.payload.seq);
    evidence.bargeIn.ended = ended;
    const afterRelease = await status();
    const cancelledOutputFrame = motionOf(beforeRelease).last_emitted.output_frame;
    assert(!motionOf(afterRelease).trace.some(row => row.mode === 'live' && row.output_frame > cancelledOutputFrame),
      'Cancelled speech resumed after the held HTTP response was released');
    evidence.bargeIn.afterRelease = afterRelease;
    const following = await submitAndComplete('following-reply', oldGeneration);
    assert.notEqual(following.requestId, held.body.request_id, 'Cancelled request ID reused');
    assert.notEqual(following.turnId, held.turnId, 'Cancelled assistant turn ID reused');
    assert.notEqual(following.turnId, userTurnId, 'User turn ID reused as reply ID');
    evidence.bargeIn.following = following;
  }
  try {
    const launchArgs = ['--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
      `--user-data-dir=${profile}`, `--disk-cache-dir=${profile}/cache`, '--disable-gpu',
      '--autoplay-policy=no-user-gesture-required', '--no-first-run', '--no-default-browser-check',
      '--window-size=1280,960', ...(args.noSandbox ? ['--no-sandbox'] : []), 'about:blank'];
    evidence.launchArgs = launchArgs;
    browser = spawn(args.browser, launchArgs, { stdio: ['ignore', 'pipe', 'pipe'], detached: true,
      env: { ...process.env, TMPDIR: profile, XDG_CACHE_HOME: `${profile}/cache` } });
    evidence.browserPid = browser.pid;
    browser.on('error', failAsync);
    for (const stream of [browser.stdout, browser.stderr]) stream.on('data', chunk => {
      if (consoleBytes >= 512 * 1024) return;
      const text = chunk.toString().slice(0, 512 * 1024 - consoleBytes);
      consoleBytes += Buffer.byteLength(text);
      evidence.console.push({ source: 'browser-process', atSeconds: elapsed(), text });
    });
    const portText = await until('Browser DevTools readiness', async () => {
      try { return await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8'); }
      catch (error) { if (error.code === 'ENOENT') return null; throw error; }
    }, 20);
    const [port, endpoint] = portText.trim().split('\n');
    assert(/^\d+$/.test(port) && endpoint.startsWith('/devtools/browser/'));
    const devtools = `http://127.0.0.1:${port}`;
    evidence.browserVersion = await jsonFetch(`${devtools}/json/version`);
    browserCdp = await CDP.connect(`ws://127.0.0.1:${port}${endpoint}`);
    evidence.browserProtocolVersion = await browserCdp.send('Browser.getVersion');
    const target = await browserCdp.send('Target.createTarget', { url: 'about:blank' });
    const targets = await jsonFetch(`${devtools}/json/list`);
    cdp = await CDP.connect(targets.find(item => item.id === target.targetId).webSocketDebuggerUrl);
    cdp.on('Runtime.consoleAPICalled', event => {
      const item = { source: 'page', atSeconds: elapsed(), type: event.type,
        args: event.args.map(arg => arg.value ?? arg.description) };
      consoleBytes += Buffer.byteLength(JSON.stringify(item));
      if (consoleBytes <= 512 * 1024) evidence.console.push(item);
      else evidence.consoleTruncated = true;
    });
    cdp.on('Runtime.exceptionThrown', event => evidence.exceptions.push({ atSeconds: elapsed(), ...event }));
    cdp.on('Network.requestWillBeSent', event => {
      const request = event.request;
      if (!request.url.startsWith(args.baseUrl) || request.method === 'GET') return;
      let payload;
      if (request.url.endsWith('/events')) { try { payload = JSON.parse(request.postData); } catch (_) {} }
      const item = { kind: 'request', id: event.requestId, method: request.method, url: request.url,
        atSeconds: elapsed(), payload };
      requests.set(event.requestId, item); evidence.network.push(item);
    });
    cdp.on('Network.responseReceived', event => {
      const request = requests.get(event.requestId);
      if (request) request.status = event.response.status;
    });
    cdp.on('Network.loadingFailed', event => {
      const request = requests.get(event.requestId);
      if (request) evidence.network.push({ ...request, kind: 'loading-failed', atSeconds: elapsed(),
        canceled: event.canceled === true, errorText: event.errorText });
    });
    cdp.on('Network.loadingFinished', event => {
      const request = requests.get(event.requestId);
      if (!request) return;
      const task = (async () => {
        const response = await cdp.send('Network.getResponseBody', { requestId: event.requestId });
        const text = response.base64Encoded ? Buffer.from(response.body, 'base64').toString() : response.body;
        let body; try { body = JSON.parse(text); } catch (_) { body = text.slice(0, 4096); }
        evidence.network.push({ ...request, kind: 'response', atSeconds: elapsed(), body });
      })();
      responseTasks.add(task);
      task.catch(failAsync).finally(() => responseTasks.delete(task));
    });
    const maxBytes = Math.floor(args.maxRecordingMiB * 1024 * 1024);
    cdp.on('Runtime.bindingCalled', event => {
      if (event.name !== '__poseLabRecorderChunk') return;
      recordingWrites = recordingWrites.then(async () => {
        const chunk = JSON.parse(event.payload);
        assert.equal(chunk.index, chunkIndex++, 'MediaRecorder chunks arrived out of order');
        const data = Buffer.from(chunk.base64, 'base64');
        recordingBytes += data.length;
        assert(recordingBytes <= maxBytes, 'Browser recording exceeded the configured byte cap');
        await fs.appendFile(path.join(args.output, 'received.webm'), data);
      });
      recordingWrites.catch(failAsync);
    });
    await cdp.send('Page.enable'); await cdp.send('Runtime.enable'); await cdp.send('Network.enable');
    await cdp.send('Runtime.addBinding', { name: '__poseLabRecorderChunk' });
    const pageUrl = `${args.baseUrl}/webrtc/pose-lab?character=${encodeURIComponent(args.character)}`;
    await cdp.send('Page.navigate', { url: pageUrl });
    await until('Pose lab page ready', async () => cdp.evaluate(`typeof EXPECTED_MOTION !== 'undefined' && !!document.querySelector('#connectButton')`), 20);
    evidence.page = await cdp.evaluate(`({url:location.href,character:SELECTED_CHARACTER,expected:EXPECTED_MOTION,
      poseSet:POSE_SET,html:document.documentElement.outerHTML})`);
    assert.equal(evidence.page.character, args.character);
    evidence.expectedMotion = evidence.page.expected;
    assert(evidence.expectedMotion, 'Legacy unverified pose lab is not a valid test target');
    evidence.page.sha256 = createHash('sha256').update(evidence.page.html).digest('hex');
    await fs.writeFile(path.join(args.output, 'page.html'), evidence.page.html);
    delete evidence.page.html;
    await click('#connectButton');
    await until('Connected and verified browser session', async () => {
      const current = await ui(); sid = current.sessionId || sid;
      return current.peer === 'connected' && current.bankState === 'verified' && !current.streamDisabled &&
        current.video.readyState >= 2 && !current.video.paused && current.video.videoWidth > 0;
    }, 40);
    sessionReady = true; evidence.sessionId = sid;
    await checkpoint('connected');
    const document = await cdp.send('DOM.getDocument');
    const input = await cdp.send('DOM.querySelector', { nodeId: document.root.nodeId, selector: '#audioFile' });
    await cdp.send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [args.audio] });
    await cdp.evaluate(`document.getElementById('speechMovement').value='talking_smiling';
      document.getElementById('speechMovement').dispatchEvent(new Event('change',{bubbles:true}));
      document.getElementById('reactionIntent').value='none';
      if(document.getElementById('audioFile').files.length!==1) throw new Error('Audio file control not populated');`);
    evidence.recording = await cdp.evaluate(`(() => {
      const video=document.getElementById('remoteVideo');
      const media=video.srcObject;
      if(!media?.getAudioTracks().length || !media.getVideoTracks().length) throw new Error('Missing received audio/video track');
      if(!video.requestVideoFrameCallback) throw new Error('Video presentation callbacks unsupported');
      const state=window.__poseLabEvidence={frames:[],frameLimit:20000,errors:[],chunks:0,bytes:0,
        startedDateMs:Date.now(),startedPerformanceMs:performance.now(),chain:Promise.resolve(),stopped:false};
      function observe(now,metadata){
        if(state.stopped)return;
        if(state.frames.length>=state.frameLimit){state.errors.push('presentation evidence limit');return;}
        const row={now,observedDateMs:Date.now()};
        for(const key of ['mediaTime','presentationTime','expectedDisplayTime','width','height','presentedFrames',
          'processingDuration','captureTime','receiveTime','rtpTimestamp']) if(metadata[key]!==undefined) row[key]=metadata[key];
        state.frames.push(row);state.callback=video.requestVideoFrameCallback(observe);
      }
      state.callback=video.requestVideoFrameCallback(observe);
      const mime='video/webm;codecs=vp8,opus';
      if(!MediaRecorder.isTypeSupported(mime))throw new Error('VP8/Opus MediaRecorder unsupported');
      const recorder=state.recorder=new MediaRecorder(media,{mimeType:mime,videoBitsPerSecond:1000000,audioBitsPerSecond:64000});
      recorder.onerror=event=>state.errors.push(String(event.error));
      recorder.ondataavailable=event=>{
        if(!event.data.size)return;
        state.bytes+=event.data.size;
        if(state.bytes>${maxBytes}){state.errors.push('recording byte cap reached');if(recorder.state!=='inactive')recorder.stop();return;}
        state.chain=state.chain.then(async()=>{
          const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(reader.error);reader.readAsDataURL(event.data);});
          window.__poseLabRecorderChunk(JSON.stringify({index:state.chunks++,base64:data.split(',')[1]}));
        }).catch(error=>state.errors.push(String(error)));
      };
      recorder.start(1000);
      return {mimeType:recorder.mimeType,videoBitsPerSecond:recorder.videoBitsPerSecond,
        audioBitsPerSecond:recorder.audioBitsPerSecond,byteCap:${maxBytes},tracks:media.getTracks().map(t=>({kind:t.kind,id:t.id,readyState:t.readyState}))};
    })()`);
    evidence.normal = await submitAndComplete('normal-long-talking-smiling');
    if (args.bargeIn) await heldResponseBargeIn();
    await sleep(500);
    const final = await checkpoint('final');
    const frames = await cdp.evaluate('window.__poseLabEvidence.frames');
    evidence.receivedMedia = assertReceivedMedia(final.stats, frames);
    assert.equal(evidence.exceptions.length, 0, 'Uncaught browser JavaScript exception');
    evidence.success = true;
  } catch (error) {
    evidence.failure = { name: error.name, message: error.message, stack: error.stack };
  } finally {
    closing = true;
    const cleanup = async (name, action) => {
      try { await action(); evidence.cleanup.push({ name, success: true }); }
      catch (error) { evidence.cleanup.push({ name, success: false, error: String(error) }); evidence.success = false; }
    };
    if (cdp) {
      if (held && !held.released && !held.cancelledByBrowser) await cleanup('release-intercepted-response', () => cdp.send('Fetch.continueResponse', { requestId: held.requestId }, 3000));
      await cleanup('capture-browser-evidence', async () => {
        const result = await cdp.evaluate(`(async()=>{
          const state=window.__poseLabEvidence;
          if(state){
            if(state.recorder.state!=='inactive')await new Promise(resolve=>{state.recorder.addEventListener('stop',resolve,{once:true});state.recorder.stop();});
            await state.chain;state.stopped=true;remoteVideo.cancelVideoFrameCallback(state.callback);
          }
          return {frames:state?.frames||[],errors:state?.errors||[],chunks:state?.chunks||0,bytes:state?.bytes||0,
            recorderState:state?.recorder.state,uiLog:document.getElementById('log')?.textContent};
        })()`, 10000);
        await recordingWrites;
        evidence.recordingResult = { ...result, frames: undefined, bytesWritten: recordingBytes };
        await save('presentation-frames.json', result.frames);
        await fs.writeFile(path.join(args.output, 'ui.log'), result.uiLog || '');
        assert.equal(result.errors.length, 0, JSON.stringify(result.errors));
        if (sessionReady) assert(recordingBytes > 0, 'No browser recording saved');
        assert.equal(recordingBytes, result.bytes, 'Recorder chunk evidence is incomplete');
        if (recordingBytes) evidence.recordingSha256 = await sha256(path.join(args.output, 'received.webm'));
      });
      await cleanup('capture-screenshot', async () => {
        const result = await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: 75 }, 5000);
        await fs.writeFile(path.join(args.output, 'final.jpg'), Buffer.from(result.data, 'base64'));
      });
      // Even a connect failure may have created a session; recover its ID before closing the page.
      await cleanup('delete-ui-session', async () => {
        try {
          const current = await ui(); sid ||= current.sessionId;
          if (current.sessionId) await click('#disconnectButton');
        } finally {
          // A prior CDP/recording failure must not prevent owned server cleanup.
          sid ||= evidence.network.find(row => row.kind === 'response' &&
            row.url.includes('/sessions/create?'))?.body?.session_id;
          if (sid) {
            let deletedByUi = false;
            const deadline = performance.now() + 3000;
            while (performance.now() < deadline) {
              try {
                const response = await fetch(`${args.baseUrl}/webrtc/sessions/${encodeURIComponent(sid)}/status`, { signal: AbortSignal.timeout(1000) });
                if (response.status === 404) { deletedByUi = true; break; }
              } catch (_) { break; }
              await sleep(100);
            }
            if (!deletedByUi) {
              const deleted = await fetch(`${args.baseUrl}/webrtc/sessions/${encodeURIComponent(sid)}`, { method: 'DELETE', signal: AbortSignal.timeout(5000) });
              assert(deleted.ok || deleted.status === 404, 'Owned session cleanup failed');
              evidence.cleanup.push({ name: 'owned-session-direct-delete-fallback', success: true });
            }
          }
        }
      });
      await Promise.allSettled([...responseTasks]);
      cdp.close();
    }
    if (browserCdp) {
      // Browser.close commonly closes the socket before a response, so process exit is the final proof.
      await browserCdp.send('Browser.close', {}, 2000).catch(() => {}); browserCdp.close();
    }
    if (browser?.pid) await cleanup('stop-owned-browser-process-group', async () => {
      if (browser.exitCode === null && browser.signalCode === null) {
        try { process.kill(-browser.pid, 'SIGTERM'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
        await Promise.race([once(browser, 'exit'), sleep(2000)]);
      }
      // A launcher can exit before its Chromium children; only signal our unique detached group.
      try { process.kill(-browser.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
      if (browser.exitCode === null && browser.signalCode === null) {
        await Promise.race([once(browser, 'exit'), sleep(2000)]);
        assert(browser.exitCode !== null || browser.signalCode !== null, 'Owned browser PID did not exit');
      }
    });
    await cleanup('remove-owned-profile', async () => {
      assert(profile.startsWith('/dev/shm/musetalk-pose-lab-cdp-'));
      await fs.rm(profile, { recursive: true, force: true });
    });
    if (asynchronousError) { evidence.success = false; evidence.asynchronousError = String(asynchronousError); }
    evidence.finishedAt = new Date().toISOString(); evidence.elapsedSeconds = elapsed();
    await save('evidence.json', evidence);
    process.removeListener('SIGINT', signal); process.removeListener('SIGTERM', signal);
  }
  console.log(JSON.stringify({ success: evidence.success, output: args.output, sessionId: sid,
    receivedMedia: evidence.receivedMedia, failure: evidence.failure?.message }));
  if (!evidence.success) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) console.log('Required: --browser PATH --character ID --audio LONG.wav --output NEW_DIR\nOptional: --base-url LOOPBACK_ORIGIN --barge-in --no-sandbox --timeout-seconds 180 --max-recording-mib 12');
  else await main(args);
}
