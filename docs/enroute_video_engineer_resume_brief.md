# Enroute Mid-Level Video Engineer Resume Tailoring Brief

This Markdown file is designed to be pasted into ChatGPT or another resume writer as source material for creating a new, targeted resume for Ahmad Sarwar for the Enroute Mid-Level Video Engineer role.

Important positioning rule: do not falsely claim production Node.js ownership unless Ahmad has separate Node.js experience not shown here. The strongest truthful angle is: backend/video streaming engineer with deep Python/FastAPI, Java/Spring Boot, TypeScript/React Native, microservices, event-driven architecture, FFmpeg, HLS, WebRTC, GPU media pipelines, S3, Redis/RQ, Kafka, Grafana/Splunk, and distributed systems experience. For the Node.js requirement, position Ahmad as a strong backend engineer who can ramp quickly because the underlying streaming, API, microservice, and async data-flow concepts are already demonstrated.

## Target Role

Company: Enroute  
Role: Mid-Level Video Engineer  
Core need: backend engineer who can build, support, debug, and optimize scalable video streaming workflows using Node.js, APIs, microservices, HLS/adaptive streaming, video segments, FFmpeg/codecs, data streams, event-driven systems, cloud storage, CDNs, Redis/Kafka/RabbitMQ, monitoring, and performance optimization across network, disk I/O, and processing layers.

## Recommended Resume Theme

Use a resume headline like:

`Video Streaming Backend Engineer | HLS, WebRTC, FFmpeg, Real-Time Media Pipelines, Microservices`

Alternative:

`Backend Video Engineer | HLS/WebRTC Streaming, FFmpeg, Distributed Media Systems, Cloud-Native APIs`

Position Ahmad as a backend engineer who built a real-time AI talking-avatar media system, not as a generic AI engineer. The resume should make Lingua/MuseTalk the most role-relevant experience section and compress unrelated AI resume language.

## One-Paragraph Professional Summary

Use or adapt this:

Backend and video streaming engineer with experience building real-time media APIs, HLS and WebRTC delivery paths, FFmpeg-based segmenting/transcoding workflows, and distributed GPU-backed video processing systems for an AI talking-avatar React Native application. Designed FastAPI microservices for session creation, audio upload, video generation, HLS manifest/segment delivery, WebRTC SDP/ICE signaling, TURN relay configuration, S3-backed avatar persistence, Redis/RQ async orchestration, and observability through health/status/load-test metrics. Also brings enterprise microservices experience from JPMorgan Chase building secure Java/Spring Boot APIs, Kafka event workflows, cloud migrations, OAuth/JWT integrations, and production runbooks in a regulated environment.

## Role Requirement Match

| Enroute requirement | Evidence from Ahmad's experience | How to express it |
| --- | --- | --- |
| Node.js backend | Direct Node.js is not shown in the provided resume. Backend depth exists in Java/Spring Boot, Python/FastAPI, TypeScript/React Native. | Say "backend API and microservices experience across Java, Python, and TypeScript; able to apply the same event-streaming and media workflow patterns in Node.js." Do not list Node.js as a skill unless true. |
| APIs/backend services | JPMorgan REST APIs; Lingua/MuseTalk FastAPI APIs for avatars, HLS sessions, WebRTC sessions, streams, status, stats, workers. | "Built REST APIs for video session lifecycle, audio upload, manifest/segment delivery, WebRTC signaling, status, health, and stats." |
| Microservices | JPMorgan cloud-native microservices; Lingua split into React Native app, EC2 control plane, GPU workers, TTS/voice, media generation, S3 persistence. | "Designed microservice-style control plane and GPU worker architecture for media generation and delivery." |
| HLS/adaptive streaming | Implemented HLS session flow with idle VOD playlist, live event playlist, `.m3u8`, fMP4 idle segments, `.ts` live chunks, live-ready polling, player switching. | "Implemented HLS delivery with dual idle/live playlists, FFmpeg-generated segments, event manifests, cache-busting, prebuffer gating, and mobile playback recovery." |
| Segmented video/media delivery | `results/hls/{session_id}/segments`, idle `seg_*.m4s`, live `chunk_*.ts`, manifest endpoints. | "Built segmented media delivery workflow serving manifests, init assets, and per-stream chunk files through API endpoints." |
| FFmpeg/codecs | FFmpeg used for idle HLS generation, per-chunk TS encoding, audio conversion to 48k mono PCM, H.264/AAC compatibility, soxr resampling. | "Used FFmpeg for H.264/AAC HLS segment generation, audio slicing/resampling, and WebRTC audio preparation." |
| Data streams/event-driven systems | Kafka at JPMorgan; async callbacks and Redis/RQ in Lingua; HLS/WebRTC scheduler jobs and stream lifecycle events. | "Built event-driven workflows using Kafka in enterprise systems and Redis/RQ/job schedulers in GPU media processing." |
| Distributed/scalable systems | EC2 control plane, Vast GPU workers, S3 avatar persistence, worker registry/autoscaling plan, health endpoints, load tests. | "Designed distributed worker architecture with session routing, worker health, queue depth, GPU utilization, and autoscaling signals." |
| WebRTC | aiortc server, SDP offer/answer, ICE candidate endpoints, TURN/STUN config, RTP audio/video tracks, strict FIFO A/V sync. | "Built WebRTC delivery using aiortc, SDP/ICE signaling, TURN relay over TCP, recvonly browser clients, synced audio/video tracks, and WebView player integration." |
| Cloud storage/CDN | S3 avatar persistence; CDN not directly implemented in docs. HLS segments are CDN-ready static assets. | "Persisted prepared avatar artifacts in S3 and designed HLS segment layout that can be fronted by CDN." Do not claim CDN deployment unless true. |
| Redis/Kafka/RabbitMQ | Kafka at JPMorgan; Redis/RQ in Lingua; RabbitMQ not shown. | List Kafka and Redis/RQ. Do not list RabbitMQ unless true. |
| Monitoring | Grafana/Splunk resume skills; health/status/stats endpoints; load-test reports; GPU metrics. | "Created operational metrics and load-test reports for live-ready latency, frame interval, strict stalls, GPU memory, and session health." |
| Network/disk/processing optimization | TURN relay debugging, direct vs relay ICE, HLS segment cadence, GPU scheduler batching, TensorRT/INT8, VRAM budgeting, FFmpeg process bottlenecks. | "Optimized network transport, segment generation cadence, GPU batching, VRAM usage, and startup warmup for real-time media workloads." |

## Most Relevant Project: Lingua / MuseTalk Video Streaming System

### Product Context

Lingua is a cross-platform React Native AI companion and language-learning app. A key user-facing feature is a real-time talking-avatar experience where users speak or receive AI-generated voice responses and see an avatar speak in sync with the audio.

The system integrates:

- React Native mobile client and WebView/native-player integration.
- Python FastAPI backend APIs.
- GPU-backed MuseTalk and LivePortrait inference workers.
- FFmpeg-based media processing.
- HLS playback for mobile compatibility, especially iOS.
- WebRTC playback for lower-latency interactive experiences.
- S3 persistence for prepared avatar assets.
- Redis/RQ-style asynchronous job orchestration.
- AWS EC2/API Gateway/IAM/Secrets Manager/Cognito/S3 and Vast.ai GPU workers.
- TensorRT/PyTorch inference optimization.

### High-Level Architecture

Use this architecture language in the resume or interview prep:

`React Native app -> EC2/API backend control plane -> MuseTalk GPU worker pool -> HLS/WebRTC session APIs -> FFmpeg/media pipeline -> S3 avatar cache -> health/stats/load-test telemetry`

The control plane creates media sessions, routes user audio to the correct worker, and tracks worker health. The GPU worker owns stateful session resources such as HLS manifests/segments, WebRTC peer connections, audio/video tracks, avatar caches, and scheduler queues.

## HLS Implementation Details

### API Flow

The implemented React Native HLS flow is:

1. Prepare an avatar through `POST /avatars/prepare`.
2. Create an HLS session with `POST /hls/sessions/create`.
3. Load the idle HLS manifest at `/hls/sessions/{session_id}/index.m3u8`.
4. Upload user/TTS audio with `POST /hls/sessions/{session_id}/stream`.
5. Poll `/hls/sessions/{session_id}/status` until `live_ready=true`.
6. Switch the player to `/hls/sessions/{session_id}/live.m3u8?stream_id={active_stream}`.
7. Return to idle playback after the live event playlist ends.

The HLS API surface includes:

- `POST /hls/sessions/create`
- `GET /hls/sessions/{session_id}/index.m3u8`
- `GET /hls/sessions/{session_id}/live.m3u8`
- `GET /hls/sessions/{session_id}/segments/{segment_name}`
- `GET /hls/sessions/{session_id}/status`
- `POST /hls/sessions/{session_id}/stream`
- `DELETE /hls/sessions/{session_id}`
- `GET /hls/sessions/stats`
- `GET /hls/player/{session_id}`
- Group/wall endpoints for concurrent session testing.

### Dual-Playlist Design

The HLS path is implemented as two separate playlists:

- Idle playlist: static VOD playlist generated once per session from the avatar idle video.
- Live playlist: event playlist generated while user/TTS audio is actively being converted into talking-avatar frames.

Storage layout:

```text
results/hls/{session_id}/
  index.m3u8
  live.m3u8
  init.mp4
  segments/seg_*.m4s
  segments/{request_id}/chunk_*.ts
```

Idle path:

- Uses FFmpeg to generate fMP4 HLS segments.
- Serves `index.m3u8` as the default loop while no live audio is active.
- Supports a separate `idle_video_file` so idle animation can be visually different from the source video used for live lip-sync generation.

Live path:

- Each inference chunk becomes a live `.ts` segment.
- The server appends `#EXTINF` entries to `live.m3u8`.
- `live_ready` flips when the first playable segment exists.
- The server appends `#EXT-X-ENDLIST` when generation completes.

Resume-tailoring language:

`Implemented a mobile-compatible HLS delivery path with session lifecycle APIs, FFmpeg-generated idle VOD manifests, server-authored live event playlists, segmented media storage, live-ready polling, and smooth idle/live transitions for a real-time AI avatar streaming product.`

### Player and Mobile Playback Strategy

The HLS player uses:

- Native HLS where available, especially iOS Safari/WKWebView.
- `hls.js` fallback where native HLS is unavailable.
- Two stacked video elements: one idle, one live.
- A hold-frame canvas overlay to prevent black frames during transitions.
- Cache-busting live manifest URLs with `stream_id`.
- A reveal gate that waits for both decoded live frame availability and prebuffer lead.
- Idle recovery logic for native HLS stalls on mobile.

Key player behaviors:

- Keep the idle layer visible until the live decoder has actually produced a frame.
- Do not switch back to idle only because the backend status becomes idle; let buffered HLS live segments finish.
- On live end, capture the last live frame, prime idle playback, then remove the hold overlay.
- Track idle video progress and seek/resume idle to avoid restarting from frame zero after every utterance.

Resume-tailoring language:

`Designed HLS player transition logic for mobile WebViews using dual video layers, live prebuffer gating, decoded-frame readiness checks, cache-busted event manifests, and hold-frame overlays to eliminate black frames and stalled live-to-idle handoffs.`

### HLS Performance and Load Testing

Load testing measured:

- Completed sessions.
- Failed sessions.
- Average time to live-ready.
- Average and max segment interval.
- Wall-clock completion time.
- GPU utilization.
- Peak GPU memory.

Observed HLS example:

- 8 concurrent HLS streams completed successfully.
- Average time to live-ready: about `2.265s`.
- Average segment interval: about `1.779s`.
- Peak GPU utilization: `100%`.
- Peak GPU memory: about `23,984MB`.

Additional testing showed:

- `30/30` FPS is viable for lower concurrency but too expensive for 8-stream hosted targets.
- `24/24` FPS at 3 streams achieved healthier cadence.
- Scheduler profiles such as `8,16` improved throughput but could worsen startup/tail latency depending on FPS and concurrency.

Resume-tailoring language:

`Built and ran HLS load tests to quantify live-ready latency, segment cadence, GPU utilization, and tail jitter; validated 8 concurrent HLS streams on a tuned GPU worker and used results to tune batch sizes, FPS targets, scheduler buckets, and prebuffer settings.`

## WebRTC Implementation Details

### API Flow

The WebRTC flow is:

1. Create a WebRTC session with `POST /webrtc/sessions/create`.
2. The browser/WebView creates an `RTCPeerConnection`.
3. Client adds recvonly audio/video transceivers.
4. Client creates SDP offer and sends it to `POST /webrtc/sessions/{session_id}/offer`.
5. Server responds with SDP answer.
6. ICE candidates are exchanged through `POST /webrtc/sessions/{session_id}/ice`.
7. User/TTS audio is uploaded through `POST /webrtc/sessions/{session_id}/stream`.
8. Server sends uploaded audio through a synced audio track and generated avatar frames through a switchable video track.

The WebRTC API surface includes:

- `POST /webrtc/sessions/create`
- `POST /webrtc/sessions/{session_id}/offer`
- `POST /webrtc/sessions/{session_id}/ice`
- `POST /webrtc/sessions/{session_id}/stream`
- `GET /webrtc/sessions/{session_id}/status`
- `GET /webrtc/sessions/stats`
- `DELETE /webrtc/sessions/{session_id}`
- `GET /webrtc/player/{session_id}`
- Group/wall endpoints for concurrent peer-connection testing.

Resume-tailoring language:

`Implemented WebRTC session APIs for SDP offer/answer exchange, ICE candidate handling, TURN/STUN configuration, audio upload, live track switching, status reporting, and browser/WebView playback.`

### WebRTC Media Track Design

The WebRTC implementation uses:

- `aiortc` for server-side peer connection handling.
- A switchable video track that starts with idle avatar frames and switches to live generated frames.
- A synced audio track that sends the uploaded/TTS audio as the authoritative audio stream.
- A shared `MediaStream` in the browser so native media element A/V sync can apply.
- Strict FIFO A/V synchronization mode.
- Server-side playout gates so audio does not race ahead of generated video.

Important implementation detail:

The uploaded audio is the authoritative audio track. MuseTalk generates video frames from that audio. HLS muxes generated video with audio into segments; WebRTC sends audio and video as separate RTP tracks but coordinates their release through a shared timing gate.

Resume-tailoring language:

`Built a WebRTC media path with separate RTP audio/video tracks coordinated by a strict FIFO playout gate, preserving lip sync by delaying release until audio preparation and generated video frames were ready.`

### TURN/STUN and Network Debugging

WebRTC network work included:

- Configuring STUN and TURN servers.
- Debugging black-screen playback caused by ICE connectivity failure.
- Differentiating model/generation failures from transport failures using ICE state, RTP byte counters, output frame counters, and player stats.
- Supporting relay-only mode with `WEBRTC_ICE_TRANSPORT_POLICY=relay`.
- Running local `coturn` with public TURN-over-TCP mappings for Vast/cloud environments.
- Separating browser/client TURN URLs from server-to-TURN local URLs.

Specific operational insight:

When the player stayed black but `output_frames_sent=0` and ICE never reached `connected`, the root cause was network transport, not TensorRT, MuseTalk, or decoder quality. Switching to a TCP-mapped TURN relay fixed public-browser session loading.

Resume-tailoring language:

`Troubleshot WebRTC black-screen failures by tracing ICE state, TURN/STUN configuration, RTP counters, and server frame delivery; implemented relay-oriented TURN-over-TCP configuration for mobile/public-cloud connectivity.`

### WebRTC A/V Sync and Playback Smoothing

The system includes:

- `WEBRTC_SYNC_MODE=strict_fifo`.
- `WEBRTC_VIDEO_PREBUFFER_SECONDS`.
- `WEBRTC_AUDIO_PREBUFFER_SECONDS`.
- `WEBRTC_ADAPTIVE_FPS` controls.
- `WEBRTC_AV_START_DELAY_SECONDS`.
- A shared `VideoSyncClock`.
- Drift logging through `WEBRTC_AUDIO_MAX_LEAD_SECONDS` and `WEBRTC_AUDIO_MAX_LAG_SECONDS`.

The sync strategy:

- Audio is prepared first.
- Video frames are generated and buffered.
- A shared playout gate releases both tracks.
- If the next video frame is not ready, audio waits too.
- This preserves lip sync at the cost of buffering or stalls under saturation.

Measured WebRTC results:

- Clean strict `20fps` playback up to about 3 concurrent streams on RTX 5000 Ada.
- Corrected VAE INT8 improved saturated aggregate throughput from about `60-61fps` to about `70-71fps`.
- INT8 reduced peak VRAM from about `24.4GB` to about `17.7GB`.
- At 4+ streams, sessions completed but strict stalls showed the compute path was saturated.

Resume-tailoring language:

`Measured and tuned WebRTC A/V sync under load, using frame interval, strict stall, aggregate FPS, live-ready, and VRAM metrics to identify GPU generation saturation separately from WebRTC signaling or transport limits.`

## FFmpeg, Codecs, and Media Processing

FFmpeg responsibilities in the system:

- Generate HLS idle VOD playlists and fMP4 segments.
- Encode each live inference chunk into MPEG-TS segments.
- Slice uploaded audio according to chunk windows.
- Convert audio to 48kHz mono PCM for WebRTC.
- Use `soxr` resampling where available.
- Target H.264/AAC compatibility for iOS/mobile HLS playback.

Relevant codec/media concepts demonstrated:

- HLS manifests and event playlists.
- fMP4/CMAF-style idle segments.
- MPEG-TS live chunks.
- H.264/AAC compatibility decisions.
- Audio sample-rate conversion.
- Audio/video muxing for HLS.
- Separate RTP media tracks for WebRTC.
- Segment duration and prebuffer tuning.

Resume-tailoring language:

`Used FFmpeg and media-format knowledge to build HLS and WebRTC pipelines, including H.264/AAC-compatible segmenting, audio slicing/resampling, MPEG-TS live chunk generation, fMP4 idle manifests, and 48kHz PCM audio preparation.`

## Distributed Systems and Event-Driven Architecture

### Enterprise Event-Driven Work

At JPMorgan Chase, Ahmad built and maintained Java/Spring Boot microservices and Kafka event-streaming workflows for onboarding, entitlement lookups, approval callbacks, workflow operations, and real-time UI status updates.

Relevant resume language:

`Built approval workflow microservices integrating entitlement lookups, external approval callbacks, Kafka event streaming, OAuth 2.0/JWT security, and real-time UI status updates.`

### Media Event-Driven Work

In Lingua/MuseTalk, the event-driven pattern appears as:

- Audio upload starts a stream job.
- Scheduler queues active HLS/WebRTC jobs.
- HLS job produces segments and appends playlist entries.
- WebRTC job pushes generated frames to callback sinks.
- Status endpoints expose state transitions like idle, streaming, live_ready, active_stream, complete.
- Completion callbacks drain playback before returning a session to idle.
- Redis/RQ-style async workers support background processing.

Resume-tailoring language:

`Designed asynchronous media workflows where audio-upload events enqueue GPU generation jobs, scheduler workers produce video chunks, callbacks update stream state, and status APIs expose live readiness, active stream IDs, completion, and health telemetry.`

## Shared GPU Scheduler and Processing Optimization

The media system uses a shared HLS GPU scheduler that was extended to serve WebRTC.

The scheduler:

- Batches work across active sessions.
- Shares avatar loading, audio feature extraction, Whisper conditioning, GPU batching, and compose workers.
- Emits HLS chunks for HLS jobs.
- Sends composed frames directly to WebRTC callbacks for WebRTC jobs.
- Avoids per-session GPU worker serialization.
- Uses batch buckets such as `4,8,16` or `8,16`.
- Supports load tests to tune concurrency and latency.

Why this matters:

Originally, independent WebRTC streams appeared to "take turns" because each stream had its own generation worker and GPU memory lease. Moving WebRTC jobs onto the shared scheduler allowed concurrent streams to generate together instead of serializing behind independent GPU leases.

Resume-tailoring language:

`Extended a shared GPU media scheduler from HLS to WebRTC, replacing per-session generation workers with cross-session batching and callback-based frame delivery to improve concurrent stream behavior and reduce serialization under load.`

## Network, Disk I/O, and Processing Performance Work

### Network

Work performed:

- Debugged WebRTC connectivity across local, public-cloud, mobile, and browser environments.
- Configured STUN/direct ICE versus relay-only TURN behavior.
- Used TURN-over-TCP to avoid relying on broad public UDP port exposure.
- Distinguished transport failures from video generation failures.
- Designed HLS/WebRTC player URLs and session APIs that work through mapped public ports.

Resume bullet:

`Debugged mobile/public-cloud WebRTC transport failures by isolating ICE, TURN, RTP, and frame-delivery metrics; configured relay-only TURN-over-TCP paths for reliable browser and mobile playback.`

### Disk I/O

Work performed:

- Defined HLS segment storage layout by session and request ID.
- Wrote idle fMP4 and live TS segments under per-session directories.
- Cleaned up session directories through TTL/session delete behavior.
- Identified repeated FFmpeg process spawning and chunk encoding as a bottleneck/future optimization area.
- Persisted prepared avatar artifacts to S3 as tarballs to avoid expensive regeneration on new workers.

Resume bullet:

`Designed session-scoped segment storage and S3 avatar persistence to reduce repeated preprocessing, isolate stream artifacts, support cleanup, and make generated media assets easier to serve, debug, and eventually front with CDN.`

### Processing

Work performed:

- Tuned MuseTalk FPS, playback FPS, batch size, chunk duration, segment duration, prebuffer, scheduler buckets, and TensorRT warmup.
- Benchmarked FP16 vs INT8 TensorRT VAE paths.
- Reduced peak VRAM through INT8 from about `24.4GB` to about `17.7GB`.
- Improved saturated WebRTC aggregate throughput from about `60-61fps` to about `70-71fps`.
- Created VRAM-aware startup profiles and warmup bucket decisions.

Resume bullet:

`Optimized GPU media processing with TensorRT/INT8 experiments, scheduler bucket tuning, FPS/batch-size profiling, and VRAM-aware startup profiles, improving saturated WebRTC throughput by roughly 15-18% and reducing peak VRAM by about 6.7GB in tested profiles.`

## Cloud Storage and Worker Persistence

S3 persistence implementation:

- Prepared avatar directories are archived and uploaded to S3.
- New workers restore prepared avatars lazily by `avatar_id`.
- Objects are versioned under a prefix such as `avatars/v15/{avatar_id}.tar.gz`.
- AWS Secrets Manager stores runtime S3 credentials for Vast workers.
- IAM separation is used between a bootstrap secret-reader and runtime S3 key.
- Retry and timeout settings exist for S3 restore.

Resume language:

`Implemented S3-backed persistence for prepared avatar assets so stateless GPU workers can restore avatars by ID instead of re-running expensive preprocessing; integrated Secrets Manager, IAM-scoped runtime credentials, retry behavior, and lazy restore flows.`

## Observability and Troubleshooting

Existing observability surfaces:

- `/health`
- `/stats`
- `/stats/gpu-live`
- `/sessions/live`
- `/hls/sessions/stats`
- `/webrtc/sessions/stats`
- Session status endpoints.
- Load-test JSON reports.
- Logs for session lifecycle, stream lifecycle, audio processing, frame generation, chunk creation, live-ready, completion, ICE state, RTP counters, GPU memory, strict stalls.

Metrics used:

- Time to live-ready.
- Segment interval.
- Frame interval.
- Max frame gap.
- Strict video stall seconds.
- Initial A/V start delta.
- Aggregate FPS.
- GPU utilization.
- Peak GPU memory.
- Session completion/failure counts.
- ICE connected/failed states.
- RTP bytes and output frames sent.

Resume language:

`Built operational visibility for real-time media sessions through health/status/stats endpoints, structured load-test reports, and stream-level metrics covering live-ready latency, segment cadence, frame intervals, A/V start delta, strict stalls, GPU utilization, VRAM, ICE state, and RTP delivery.`

## JPMorgan Experience to Keep for This Role

The Enroute posting is backend-heavy, so keep these JPMorgan themes:

- Java/Spring Boot REST APIs.
- Microservices.
- Kafka event streaming.
- Cloud migration.
- OAuth 2.0/JWT.
- Debugging/troubleshooting.
- Production runbooks and regulated delivery.
- Data-access modernization and latency improvement.

Best JPMorgan bullets for this target:

- `Built and maintained Java/Spring Boot REST APIs and cloud-native microservices for onboarding, reference data, agreements, messaging, preferences, and workflow operations in a regulated financial-services environment.`
- `Built approval workflow microservices integrating entitlement lookups, external approval callbacks, Kafka event streaming, OAuth 2.0/JWT security, and real-time UI status updates.`
- `Led microservice migrations from Linux on-prem environments to private cloud, AWS, and Kubernetes platforms, balancing scalability, resiliency, compliance, observability, and delivery timelines.`
- `Modernized reference-data ingestion and persistence from stored procedures to JPA/JDBC-backed services, improving retrieval latency by about 40%.`

## Suggested New Resume Structure

### Header

Ahmad Sarwar  
Video Streaming Backend Engineer | HLS, WebRTC, FFmpeg, Distributed Media Systems  
Email | LinkedIn | Phone

### Summary

Use the video/backend summary from above.

### Technical Skills

Recommended skills section:

`Video / Streaming: HLS, WebRTC, FFmpeg, H.264/AAC, MPEG-TS, fMP4/CMAF concepts, SDP/ICE, STUN/TURN, aiortc, PyAV, hls.js, media segmenting, A/V sync, video load testing`

`Backend: Python, FastAPI, Java, Spring Boot, TypeScript, JavaScript, REST APIs, microservices, async job orchestration, Kafka, Redis/RQ, OAuth 2.0/JWT`

`Cloud / DevOps: AWS EC2, S3, API Gateway, IAM, Secrets Manager, Cognito, Kubernetes, Docker, Linux, Vast.ai GPU workers, Jenkins`

`Observability / Performance: Grafana, Splunk, GPU metrics, health/status APIs, load testing, latency profiling, TensorRT, PyTorch, VRAM optimization`

Do not list RabbitMQ, GStreamer, VLC, OpenCV, CDNs, or Node.js unless Ahmad can defend hands-on experience.

### Experience Order

For this specific job, put Lingua first if acceptable because it is the strongest match. If maintaining chronology strictly, keep JPMorgan first but make Lingua the most detailed section.

Recommended order:

1. Lingua - Bootstrap Founder / Software Engineer
2. JPMorgan Chase - Software Engineer
3. Education / Certifications

## Bullet Bank for Lingua / MuseTalk

Use 6-9 of these, depending on resume length:

- `Built real-time talking-avatar streaming infrastructure for a React Native AI companion app, exposing FastAPI session APIs for avatar preparation, audio upload, HLS playback, WebRTC signaling, status polling, health checks, and stats.`
- `Implemented HLS delivery with dual idle/live playlists: FFmpeg-generated idle fMP4 VOD manifests, server-authored live event playlists, per-request MPEG-TS chunks, live-ready state, and mobile-compatible H.264/AAC playback.`
- `Designed the HLS mobile player strategy for iOS/WebView using native HLS, dual video layers, decoded-frame readiness gates, live prebuffering, cache-busted manifests, hold-frame canvas overlays, and idle recovery logic to reduce black frames and stalled transitions.`
- `Implemented WebRTC streaming with aiortc, SDP offer/answer exchange, ICE candidate APIs, STUN/TURN configuration, recvonly browser clients, synced audio/video tracks, and relay-oriented TURN-over-TCP support for public/mobile networks.`
- `Built strict FIFO A/V synchronization for WebRTC, coordinating uploaded audio and generated video frames through server-side playout gates, prebuffer controls, and drift/stall metrics to preserve lip sync under GPU load.`
- `Used FFmpeg for HLS segment generation, live chunk muxing, audio slicing, 48kHz mono PCM conversion, and mobile-compatible media preparation across HLS and WebRTC workflows.`
- `Extended a shared GPU scheduler to support both HLS segment output and WebRTC frame-callback output, batching active sessions across workers to reduce per-session serialization and improve concurrent stream behavior.`
- `Created HLS and WebRTC load-test workflows measuring live-ready latency, segment/frame interval, max frame gap, aggregate FPS, strict video stalls, A/V start delta, GPU utilization, peak VRAM, and session completion rates.`
- `Validated 8 concurrent HLS streams on a tuned GPU worker with about 2.3s average live-ready latency and about 1.8s average segment cadence, then used tail-latency results to tune FPS, batch size, segment duration, and scheduler buckets.`
- `Benchmarked WebRTC throughput on RTX 5000 Ada at 20fps, identifying clean strict playback at 3 concurrent streams and GPU generation saturation beyond that through frame interval, stall, and aggregate FPS analysis.`
- `Improved saturated WebRTC media throughput by roughly 15-18% in tested profiles by moving from FP16 to TensorRT/INT8 VAE decoding, while reducing peak VRAM from about 24.4GB to about 17.7GB.`
- `Implemented S3-backed persistence for prepared avatar artifacts so new GPU workers can lazily restore avatars by ID instead of re-running expensive preprocessing; integrated IAM-scoped credentials, Secrets Manager, retries, and versioned object prefixes.`
- `Designed autoscaling signals for stateful GPU media workers, including active HLS/WebRTC streams, queue depth, health state, free capacity, GPU utilization, VRAM usage, p95 first-chunk latency, and drain readiness.`
- `Debugged real-time media failures across network, encoding, GPU, and playback layers, including ICE/TURN black screens, native HLS idle-loop stalls, live-to-idle transition hangs, FFmpeg encoder bottlenecks, and TensorRT warmup/VRAM constraints.`

## Bullet Bank for JPMorgan

Use 4-6 of these:

- `Built and maintained Java/Spring Boot REST APIs and cloud-native microservices for onboarding, reference data, agreements, messaging, preferences, and workflow operations across enterprise financial platforms.`
- `Built approval workflow services integrating entitlement lookups, external approval callbacks, Kafka event streaming, OAuth 2.0/JWT security, and real-time UI status updates.`
- `Modernized reference-data ingestion and persistence from stored procedures to JPA/JDBC-backed services, improving retrieval latency by about 40%.`
- `Led microservice migrations from Linux on-prem infrastructure to private cloud, AWS, and Kubernetes environments with attention to resiliency, observability, compliance, and delivery timelines.`
- `Designed a user preference service and migration strategy for 6M+ records, including schema design, data mapping, feature-flagged synchronization APIs, rollback planning, and cutover strategy.`
- `Translated business, security, compliance, and platform requirements into service designs, API contracts, runbooks, deployment plans, and production-ready implementations.`

## Keywords to Include

Use these naturally:

- HLS
- adaptive streaming
- WebRTC
- FFmpeg
- H.264
- AAC
- MPEG-TS
- fMP4
- media segments
- manifests
- event playlist
- SDP
- ICE
- STUN
- TURN
- RTP
- A/V sync
- video processing pipeline
- real-time streaming
- backend services
- microservices
- REST APIs
- event-driven architecture
- Kafka
- Redis
- S3
- health checks
- monitoring
- load testing
- latency
- throughput
- GPU utilization
- TensorRT
- PyTorch
- React Native
- TypeScript
- Python
- FastAPI
- Java
- Spring Boot

## Gaps and Honest Framing

### Node.js

The posting strongly prefers Node.js. The provided resume does not show direct Node.js backend work. Do not fabricate it.

Recommended framing:

`Although my recent media backend work is primarily Python/FastAPI and my enterprise backend work is Java/Spring Boot, the core responsibilities of this role map directly to systems I have built: streaming APIs, media session lifecycle management, segmented video delivery, event-driven processing, FFmpeg workflows, WebRTC signaling, cloud storage, and performance debugging. I am comfortable applying those backend patterns in a Node.js service environment.`

### CDN

The docs show S3 persistence and HLS segment delivery, but not a confirmed CDN deployment.

Recommended framing:

`Designed HLS segment and manifest delivery in a CDN-compatible layout and persisted media/avatar assets in S3; direct CDN deployment is an area I can ramp into quickly.`

### GStreamer / OpenCV / VLC

Do not claim these unless Ahmad has experience elsewhere. The stronger tools are FFmpeg, aiortc, PyAV, hls.js, TensorRT, PyTorch.

### RabbitMQ

Do not claim RabbitMQ unless true. Use Kafka and Redis/RQ.

## Suggested Resume Summary Bullets

These can appear in a top "Selected Highlights" section:

- `Built HLS and WebRTC streaming paths for real-time AI talking avatars, including session APIs, FFmpeg segmenting, SDP/ICE signaling, TURN relay configuration, A/V sync controls, mobile WebView playback, and stream status telemetry.`
- `Validated and tuned concurrent video workloads through load tests measuring live-ready latency, frame/segment cadence, strict stalls, aggregate FPS, GPU utilization, and VRAM across FP16 and TensorRT/INT8 profiles.`
- `Bring enterprise backend depth from JPMorgan Chase across Java/Spring Boot microservices, Kafka event streaming, OAuth/JWT security, cloud migration, API design, debugging, runbooks, and production delivery.`

## Possible Tailored Experience Section Draft

### Lingua - AI Friend & Language Learning Partner App

Bootstrap Founder / Software Engineer  
Mar. 2025 - Present

- Built the real-time talking-avatar media backend for a React Native AI companion app, exposing FastAPI APIs for avatar preparation, HLS/WebRTC session lifecycle, audio upload, stream status, health checks, and operational stats.
- Implemented HLS streaming with dual idle/live playlists, FFmpeg-generated idle fMP4 segments, server-authored live event playlists, MPEG-TS live chunks, cache-busted manifests, and mobile-compatible H.264/AAC playback.
- Designed mobile HLS playback behavior for iOS/WebView with dual video layers, decoded-frame readiness gates, prebuffer thresholds, hold-frame canvas overlays, idle-loop recovery, and smooth live-to-idle transitions.
- Implemented WebRTC playback using aiortc with SDP offer/answer APIs, ICE candidate exchange, STUN/TURN configuration, synced RTP audio/video tracks, strict FIFO playout gates, and TURN-over-TCP relay support for public/mobile networks.
- Used FFmpeg for HLS segment generation, live chunk muxing, audio slicing, 48kHz mono PCM conversion, and media preparation across HLS and WebRTC workflows.
- Extended a shared GPU scheduler to support both HLS segment output and WebRTC frame-callback output, batching active sessions across streams to reduce serialization and improve concurrent generation behavior.
- Built load-test and observability workflows for live-ready latency, segment/frame intervals, aggregate FPS, strict video stalls, A/V start delta, GPU utilization, peak VRAM, and session completion/failure rates.
- Improved tested WebRTC saturated throughput by roughly 15-18% and reduced peak VRAM by about 6.7GB through TensorRT/INT8 VAE experiments, scheduler bucket tuning, and FPS/batch-size profiling.
- Implemented S3-backed avatar persistence and lazy restore for GPU workers using versioned object prefixes, IAM-scoped runtime credentials, AWS Secrets Manager, retries, and worker bootstrap validation.

### JPMorgan Chase & Co.

Software Engineer  
Jun. 2021 - Present

- Built and maintained Java/Spring Boot REST APIs and cloud-native microservices for onboarding, reference data, agreements, messaging, preferences, and workflow operations in a regulated financial-services environment.
- Built approval workflow services integrating entitlement lookups, external approval callbacks, Kafka event streaming, OAuth 2.0/JWT security, and real-time UI status updates.
- Modernized reference-data ingestion and persistence from stored procedures to JPA/JDBC-backed services, improving retrieval latency by about 40%.
- Led microservice migrations from Linux on-prem environments to private cloud, AWS, and Kubernetes platforms, balancing resiliency, compliance, observability, data residency, and delivery timelines.
- Designed a user preference service and migration strategy for 6M+ records, including schema design, data mapping, feature-flagged synchronization APIs, rollback planning, and cutover strategy.

## Interview Talking Points

### HLS

Talk about the dual-playlist strategy, why it was chosen for iOS compatibility, and how idle/live switching was handled without black frames. Mention that LL-HLS was considered but not fully implemented because the current per-chunk TS path does not generate LL-HLS parts; this shows judgment and honesty.

### WebRTC

Talk about the difference between transport problems and generation problems. Use the black-screen debugging example: ICE not connected and zero RTP/frame counters meant it was TURN/network, not model quality. Explain how TURN-over-TCP solved the public-cloud/mobile connectivity issue.

### FFmpeg

Talk about FFmpeg as the practical media workhorse: HLS segmenting, TS chunks, H.264/AAC compatibility, audio slicing, PCM conversion, and resampling. Also mention that per-chunk FFmpeg process spawning was identified as a bottleneck and a persistent encoder/segmenter was a future optimization.

### Distributed Systems

Talk about the media worker as stateful: HLS session files, WebRTC peer connections, GPU scheduler queues, avatar caches, and active sessions. Explain why autoscaling must use media-specific signals like active streams, queue depth, p95 first chunk, GPU utilization, and worker drain state.

### Performance

Talk in metrics:

- 8 concurrent HLS streams validated.
- HLS live-ready around 2.3s in a reference 8-stream test.
- WebRTC clean strict 20fps capacity around 3 streams on RTX 5000 Ada.
- INT8 improved saturated aggregate WebRTC throughput from about 60-61fps to 70-71fps.
- INT8 reduced peak VRAM from about 24.4GB to 17.7GB.

## Final Resume Direction

The tailored resume should not lead with RAG/agentic AI. It should lead with video streaming backend engineering. AI can remain relevant as the source of the generated media, but the Enroute role cares more about:

- HLS/WebRTC.
- FFmpeg and codecs.
- Video segments and media delivery.
- Backend APIs.
- Data streams and events.
- Microservices.
- Distributed systems.
- Debugging.
- Performance/reliability.

The strongest narrative is:

`I am a backend engineer who has already built and debugged the hard parts of a real-time video streaming system: session APIs, HLS manifests and segments, WebRTC signaling and TURN, FFmpeg processing, mobile playback edge cases, GPU-backed processing, observability, load testing, and production-oriented cloud worker design.`
