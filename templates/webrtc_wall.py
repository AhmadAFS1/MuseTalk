from html import escape

from templates.hls_wall import get_hls_wall_html


def get_webrtc_wall_html(default_avatar_id: str = "test_avatar") -> str:
    html = get_hls_wall_html()
    safe_default_avatar_id = escape(default_avatar_id, quote=True)
    replacements = {
        "<title>HLS Wall</title>": "<title>WebRTC Wall</title>",
        "HLS Session Wall": "WebRTC Session Wall",
        "Create a batch of HLS sessions, view them together, and start them all with one upload.": (
            "Create a batch of WebRTC sessions, view them together, and start them all with one upload."
        ),
        "HLS Jobs": "WebRTC Streams",
        "No live scheduler jobs yet.": "No live WebRTC streams yet.",
        "Status API: GET ${location.origin}/hls/groups/${currentGroup.group_id}": (
            "Status API: GET ${location.origin}/webrtc/groups/${currentGroup.group_id}"
        ),
    }
    for old, new in replacements.items():
        html = html.replace(old, new)

    html = html.replace("/hls/", "/webrtc/")
    html = html.replace('/hls/sessions/stats', '/webrtc/sessions/stats')
    html = html.replace('id="avatarId" value="test_avatar"', f'id="avatarId" value="{safe_default_avatar_id}"')
    html = html.replace('const initialGroupMatch = window.location.pathname.match(/^\\/hls\\/groups\\/([^/]+)\\/wall$/);',
                        'const initialGroupMatch = window.location.pathname.match(/^\\/webrtc\\/groups\\/([^/]+)\\/wall$/);')
    html = html.replace('frame.allow = "autoplay";', 'frame.allow = "autoplay; fullscreen";')
    return html
