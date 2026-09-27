#!/usr/bin/env python3
"""Browser regression for wall audio controls; requires optional Playwright.

Uses real wall/player HTML with synthetic media and stubbed signaling. No GPU,
TTS requests or production sessions are needed.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates.webrtc_player import get_webrtc_player_html
from templates.webrtc_wall import get_webrtc_wall_html


SYNTHETIC_RTC = """
if (location.pathname.startsWith('/webrtc/player/')) {
  window.forceAudioBlock = true;
  window.audioUnlocked = false;
  const realPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function() {
    if (!this.muted) {
      if (window.forceAudioBlock || (!window.audioUnlocked && !navigator.userActivation.isActive)) {
        return Promise.reject(new DOMException('Click required', 'NotAllowedError'));
      }
      window.audioUnlocked = true;
    }
    return realPlay.call(this);
  };
  window.RTCPeerConnection = class {
    addTransceiver() {}
    async createOffer() { return {type:'offer', sdp:'synthetic'}; }
    async setLocalDescription(value) { this.localDescription = value; }
    async setRemoteDescription() {
      const canvas = document.createElement('canvas');
      canvas.width = 64; canvas.height = 64;
      const ctx = canvas.getContext('2d');
      this.timer = setInterval(() => {
        ctx.fillStyle = '#698'; ctx.fillRect(0, 0, 64, 64);
        ctx.fillStyle = '#fff'; ctx.fillText(String(Date.now()), 0, 30);
      }, 100);
      this.stream = canvas.captureStream(10);
      for (const track of this.stream.getTracks()) this.ontrack({track});
      this.connectionState = this.iceConnectionState = 'connected';
      this.onconnectionstatechange();
    }
    async getStats() { return new Map(); }
    close() { clearInterval(this.timer); this.stream?.getTracks().forEach(t=>t.stop()); }
  };
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    sessions = [dict(session_id=f'audio_{i}', player_url=f'/webrtc/player/audio_{i}',
                     status='connected') for i in range(3)]
    group = dict(group_id='audio_browser_test', sessions=sessions,
                 wall_url='/webrtc/groups/audio_browser_test/wall', config={})
    refreshes = 0
    errors = []

    def route_request(route):
        nonlocal refreshes
        path = urlparse(route.request.url).path
        if path == '/webrtc/wall':
            route.fulfill(content_type='text/html', body=get_webrtc_wall_html())
        elif path.startswith('/webrtc/player/'):
            session = SimpleNamespace(session_id=path.rsplit('/', 1)[-1],
                                      avatar_id='synthetic', ice_servers=[], fps=20)
            route.fulfill(content_type='text/html', body=get_webrtc_player_html(session))
        elif path.endswith('/offer'):
            route.fulfill(json={'type': 'answer', 'sdp': 'synthetic'})
        elif path == '/webrtc/groups/create':
            route.fulfill(json=group)
        elif path == '/webrtc/groups/audio_browser_test':
            refreshes += 1
            route.fulfill(json=group)
        else:
            route.fulfill(json={})

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context()
        context.add_init_script(SYNTHETIC_RTC)
        context.route('http://127.0.0.1:8331/**', route_request)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8331/webrtc/wall')
        page.click('#createBtn')
        page.wait_for_function("document.querySelectorAll('iframe').length === 3")
        assert 'Create a group to populate the wall.' not in page.locator('#wall').inner_text()
        for frame in page.frames[1:]:
            frame.wait_for_selector('#remoteVideo')
            frame.wait_for_function("!document.querySelector('#remoteVideo').paused")
            frame.wait_for_function("document.querySelector('#remoteVideo').muted")
            frame.wait_for_function("document.querySelector('#statusOverlay').textContent === 'Tap to enable audio' && !document.querySelector('#statusOverlay').classList.contains('hidden')")
            frame.evaluate('window.forceAudioBlock = false')

        def assert_audio(expected):
            page.wait_for_function("expected => [...document.querySelectorAll('iframe')].every((f,i) => {const v=f.contentDocument.querySelector('video'); return v && v.muted===!expected[i] && v.volume===(expected[i]?1:0) && !v.paused;})", arg=expected)
            assert page.locator('.card.audible').count() == sum(expected)

        page.click('#allAudioBtn')
        assert_audio([True, True, True])
        page.locator('[data-role="audio"]').nth(1).click()
        assert_audio([True, False, True])
        original_frames = page.evaluate("window.originalWallFrames = [...document.querySelectorAll('iframe')]; originalWallFrames.length")
        assert original_frames == 3
        before = refreshes
        page.wait_for_timeout(4200)
        assert refreshes > before, 'Periodic refresh did not run'
        assert_audio([True, False, True])
        assert page.evaluate("[...document.querySelectorAll('iframe')].every((f,i)=>f===originalWallFrames[i])")
        page.click('#reconnectBtn')
        assert_audio([True, False, True])
        page.locator('[data-role="solo"]').nth(2).click()
        assert_audio([False, False, True])
        page.locator('[data-role="audio"]').nth(0).click()
        assert_audio([True, False, True])
        page.click('#muteAllBtn')
        assert_audio([False, False, False])
        page.click('#allAudioBtn')
        assert_audio([True, True, True])
        # A player reload must recover the wall's most recent mute choice.
        page.locator('[data-role="audio"]').nth(1).click()
        frame = page.frames[2]
        with frame.expect_navigation():
            frame.evaluate('location.reload()')
        frame.wait_for_function("document.querySelector('#remoteVideo') && !document.querySelector('#remoteVideo').paused")
        assert_audio([True, False, True])
        assert not errors, errors
        result = dict(passed=True, peers=3, periodic_refreshes=refreshes,
                      checks=['autoplay falls back to muted video',
                              'one click enables all players', 'independent mute/unmute',
                              'refresh preserves iframe and mute state',
                              'reconnect preserves mute state', 'solo then multiple audio',
                              'mute all', 'player reload restores mute preference'])
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
