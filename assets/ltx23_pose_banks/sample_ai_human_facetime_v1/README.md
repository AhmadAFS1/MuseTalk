# Sample AI Human — LTX 2.3 pose bank

Use the six MP4s in `certified/` for MuseTalk testing:

- `neutral_resting.mp4`
- `active_listening.mp4`
- `speaking_direct.mp4`
- `nod_agree.mp4`
- `empathetic_head_tilt.mp4`
- `light_smile.mp4`

They are silent 480×832 H.264 videos at 24 fps. All six opening frames and
all six closing frames of every certified clip decode to the same canonical
RGB image. The shared decoded-frame SHA-256 is:

`47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33`

The six-frame canonical handles are followed or preceded by six-frame blends
so motion stays in the middle of each clip. `manifest.json` contains the full
generation and validation record. `raw_ltx/` is retained for provenance and
should not be used as the runtime pose bank.
