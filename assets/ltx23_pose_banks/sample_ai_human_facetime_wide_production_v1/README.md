# Sample AI human — LTX 2.3 wide production bank v1

This is the non-destructive full-shoulder successor to the original close-framed FaceTime pose bank. It contains six unique certified videos for the six semantic MuseTalk poses: neutral and active listening intentionally share one physical file, while direct speech has two renderer-owned variants.

All certified files are 480×832, 24 fps, silent H.264 MP4s. Their first six and final six decoded frames are byte-equivalent in RGB space and share boundary hash `df2b493201dbbff467a53fd32286c00a28173c1a89290336447abb40c558d9bd`.

The source portrait is in `source/`. The complete motion provenance and exact content hashes are in `manifest.json`; machine validation is in `validation_report.json`. Rejected LTX rerolls remain in `/workspace/LTX-2.3/musetalk_pose_banks` and are intentionally not deleted.

`review/midpoint_contact_sheet.jpg` provides a labeled midpoint framing check across all six physical videos; both shoulders remain within frame in every sample.

All six content-hashed MuseTalk caches have been prepared without force replacement. A live six-pose WebRTC cycle and a separate two-turn direct-speaking V14-to-V15 rotation both passed on 2026-09-01. Receiver recordings and compact proof metadata are under `generated/webrtc_pose_showcase/2026-09-01/ltx23_full_shoulders_production_v1/`.

Runtime configuration:

`configs/pose_test/sample_ai_human_ltx23_facetime_wide_production_v1.json`

Rollback requires only restoring the previous default manifest pointer. No prior assets or prepared avatar directories are overwritten by this bank.
