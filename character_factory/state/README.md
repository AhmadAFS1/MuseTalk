# state/

Working directory for a factory run. Everything here except `roster.json` is reproducible.

| Path | Contents | Keep? |
|---|---|---|
| `roster.json` | the 332-character roster | **yes** — casting decisions live here |
| `ledger.json` | per-character progress, hashes, seeds, prompt IDs | **yes** — makes runs resumable |
| `portrait_prompts/` | ChatGPT prompt files + `queue.json` | regenerable |
| `portrait_inbox/` | PNGs saved out of ChatGPT | regenerable, expensively |
| `portraits/` | canonical 480x832 portraits | regenerable from the inbox |
| `renders/<id>/` | raw LTX clips | regenerable, expensively |
| `certified/<id>/` | handle-certified clips + validation report | regenerable from renders |
| `timings.jsonl` | append-only render wall clocks | keep; it is the only real throughput data |

`portrait_prompts/ja_01.md`, `ja_02.md`, and `ja_03.md` are committed as format examples.

Packaged output does **not** live here. It goes to `assets/ltx23_pose_banks/` and
`configs/pose_test/` in the MuseTalk tree, which is what the runtime loads.
