# Student character release QA app

This separate iPhone app uses a snapshot of Lingua's shipping HarnessCore. Its bundle identifier is `com.lingua.musetalkstudentqa`; installation preserves the original `com.lingua.musetalkbench` app and its experimental models.

The app fetches an immutable shared model pack and original 24 fps character bundle from S3. Pass `--render-fps 25` to use shipping timing conversion: 240 source frames span 250 render ticks, preserving the source animation speed. The approved profile is `student-r2-a05d2d98-v1`, UNet `unet_bk_tiny_r2_selected_int8w_b1`, with two Neural Engine inference workers. No character-specific model conversion or training is needed.

Build the Xcode project in `App/MuseTalkBench.xcodeproj`, scheme `MuseTalkBench`, for a connected development iPhone. Embedded resources contain no model weights. For playback pass `--lipsync --exit --unet unet_bk_tiny_r2_selected_int8w_b1 --infer-workers 2 --render-fps 25`, plus `--models-manifest-url`, `--model-files-json` (a JSON filename-to-signed-URL map), `--avatar-url`, `--avatar-sha256`, `--avatar-id`, and `--wav`. Keep signed arguments and console logs private.

`LIPSYNC_RESULT` reports generated/displayed frames and fps, skips, stalls, late frames, audio playback, and timing. `MEM_RESULT` records memory pressure and thermal state. This is a focused native renderer harness, not the complete React Native call screen. The original benchmark remains in `ios/MuseTalkBench`; its experimental GPU options and Mac executable are separate.

The app's HarnessCore files must remain byte-identical to the shipping source revision under test. Store their hashes, the compiled app hash, the source commit and immutable model/character hashes with each physical test receipt. Approved-student execution checks use its own quantized Core ML reference, never a teacher quality comparison.
