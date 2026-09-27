# Concurrent audio on the WebRTC wall

The wall supports concurrent sound from multiple peers. **All streams audio**
unmutes every tile; **Mute all** silences them. Each tile has independent
**Mute/Unmute** and **Solo** buttons. New streams request audio by default.

The earlier wall selected a single audio leader and muted every other player
on each stats refresh. That helped isolate one stream during A/V sync checks,
but prevented listening to concurrent peers. Audio preferences are now stored
per session and preserved during periodic refreshes and reconnects. These
controls affect browser playback; muting a tile does not reduce its server
inference or network workload.

Some browsers require a click before playing sound. The player falls back to
muted video and displays **Tap to enable audio** when autoplay is blocked.
Clicking **All streams audio** invokes each player in the same user gesture.

Validated with three real Chromium WebRTC peers receiving a Kokoro turn:
all three received nonzero audio energy, with media elements playing,
`muted=false`, and `volume=1`. Independent mute, Solo, all-stream playback,
and mute persistence through stats refresh and reconnect passed. The test
used 20 FPS generation/transport, batch 8, and 0.50-second prebuffer. This
validates audio controls rather than sustained concurrency capacity.

Evidence: [live browser report](../../experiments/chinese_bob_webrtc_20260927/wall_audio_controls/live_browser.json),
[wall screenshot](../../experiments/chinese_bob_webrtc_20260927/wall_audio_controls/live_wall.png),
and [browser regression report](../../experiments/chinese_bob_webrtc_20260927/wall_audio_controls/browser_regression.json).

The optional Playwright regression exercises the real wall/player templates
with synthetic media, without GPU jobs:

```bash
python scripts/test_webrtc_wall_audio_browser.py
```

Run it in an environment with Playwright and Chromium installed. The live
test script is retained beside the reports. The wall server must reload the
updated templates; users should refresh the page and recreate any groups
cleared by a server restart.
