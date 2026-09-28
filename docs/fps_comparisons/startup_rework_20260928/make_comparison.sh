#!/usr/bin/env bash
# Labelled side-by-side comparison of recorded WebRTC sessions (startup rework A/B).
# Each arm: full 480x832 frame with its label, plus a 2.67x mouth zoom underneath.
#
#   make_comparison.sh OUT.mp4 CLIP1 "LABEL 1" CLIP2 "LABEL 2" [CLIP3 "LABEL 3" ...]
#
# Audio: the session TTS (data/audio/ai-assistant.mpga) delayed by the recorder's
# first live audio target (~1.06 s), so lips can be judged against sound.
set -euo pipefail

OUT=${1:?output}; shift
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
AUDIO=${COMPARISON_AUDIO:-/workspace/MuseTalk/data/audio/ai-assistant.mpga}
AUDIO_DELAY_MS=${COMPARISON_AUDIO_DELAY_MS:-1060}
# Mouth crop for indian_realtime_talking_20f9845543 at 480x832 (mouth centre ~238,395).
CROP=${COMPARISON_MOUTH_CROP:-180:150:148:338}

inputs=() filters="" stack="" n=0 min_dur=""
while (( $# >= 2 )); do
    clip=$1 label=$2; shift 2
    inputs+=(-i "$clip")
    dur=$(ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 "$clip")
    if [[ -z "$min_dur" ]] || awk -v a="$dur" -v b="$min_dur" 'BEGIN{exit !(a<b)}'; then min_dur=$dur; fi
    esc=${label//:/\\:}
    filters+="[$n:v]split=2[f$n][m$n];"
    filters+="[f$n]scale=480:832,drawtext=fontfile=$FONT:text='$esc':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8[fl$n];"
    filters+="[m$n]crop=$CROP,scale=480:400:flags=lanczos,drawtext=fontfile=$FONT:text='mouth zoom':x=12:y=12:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6[ml$n];"
    filters+="[fl$n][ml$n]vstack=inputs=2[col$n];"
    stack+="[col$n]"
    n=$((n + 1))
done
(( n >= 2 )) || { echo "need at least two clips" >&2; exit 2; }
filters+="${stack}hstack=inputs=$n:shortest=1[v];"
filters+="[$n:a]adelay=${AUDIO_DELAY_MS}|${AUDIO_DELAY_MS},apad,atrim=0:${min_dur}[a]"

ffmpeg -v error -y "${inputs[@]}" -i "$AUDIO" -filter_complex "$filters" \
    -map "[v]" -map "[a]" -t "$min_dur" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    -c:a aac -b:a 128k -r 20 "$OUT"
echo "$OUT"
