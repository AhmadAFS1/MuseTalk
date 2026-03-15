#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <label> [concurrency_csv]"
  echo "Example: $0 baseline 1,4,8"
  exit 1
fi

LABEL="$1"
CONCURRENCY_CSV="${2:-1,4,8}"

BASE_URL="${BASE_URL:-http://localhost:8000}"
AVATAR_ID="${AVATAR_ID:-test_avatar}"
AUDIO_FILE="${AUDIO_FILE:-./data/audio/ai-assistant.mpga}"
PLAYBACK_FPS="${PLAYBACK_FPS:-24}"
MUSETALK_FPS="${MUSETALK_FPS:-12}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEGMENT_DURATION="${SEGMENT_DURATION:-1.0}"
HOLD_SECONDS="${HOLD_SECONDS:-5}"

RESULT_DIR="results/compile_matrix"
mkdir -p "${RESULT_DIR}"

IFS=',' read -r -a CONCURRENCIES <<< "${CONCURRENCY_CSV}"

echo "Running compile matrix label=${LABEL}"
echo "Server target: ${BASE_URL}"
echo "Profile: playback_fps=${PLAYBACK_FPS}, musetalk_fps=${MUSETALK_FPS}, batch_size=${BATCH_SIZE}, segment_duration=${SEGMENT_DURATION}"
echo "Concurrencies: ${CONCURRENCY_CSV}"

for concurrency in "${CONCURRENCIES[@]}"; do
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  outfile="${RESULT_DIR}/${LABEL}_c${concurrency}_${timestamp}.json"

  echo
  echo "=== Running concurrency=${concurrency} ==="
  python load_test.py \
    --base-url "${BASE_URL}" \
    --avatar-id "${AVATAR_ID}" \
    --audio-file "${AUDIO_FILE}" \
    --concurrency "${concurrency}" \
    --segment-duration "${SEGMENT_DURATION}" \
    --playback-fps "${PLAYBACK_FPS}" \
    --musetalk-fps "${MUSETALK_FPS}" \
    --batch-size "${BATCH_SIZE}" \
    --hold-seconds "${HOLD_SECONDS}"

  cp load_test_report.json "${outfile}"
  echo "Saved ${outfile}"
done

echo
echo "Done. Results are in ${RESULT_DIR}"
