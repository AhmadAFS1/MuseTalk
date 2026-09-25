"""Keep FFmpeg's worker-thread diagnostics out of the Python GIL.

Torchvision enables PyAV's Python logging callback when its video modules are
imported. In PyAV 16, codec finalization can hold the GIL while joining decoder
workers. Those workers can enter the logging callback and wait for that same
GIL, even for messages below the configured logging threshold. Native FFmpeg
logging avoids that lock inversion and still reports codec errors to stderr.
"""


def configure_native_ffmpeg_logging(process_label: str = "process") -> dict:
    """Apply after dependency/model imports and before starting media workers.

    Safe to repeat after lazy initialization; intentionally do not cache an
    'already configured' flag because a later import may replace the callback.
    This changes process-wide FFmpeg logging, not decoder thread counts.
    """
    import av

    previous = av.logging.get_level()
    av.logging.set_level(None)
    av.logging.set_libav_level(av.logging.ERROR)
    av.logging.restore_default_callback()
    summary = {
        "previous_python_level": previous,
        "python_level": av.logging.get_level(),
        "native_level": av.logging.ERROR,
        "callback": "ffmpeg_native",
    }
    print(
        f"[{process_label}] FFmpeg logging callback=ffmpeg_native "
        f"native_level=ERROR previous_python_level={previous!r} "
        "python_callback=disabled",
        flush=True,
    )
    return summary
