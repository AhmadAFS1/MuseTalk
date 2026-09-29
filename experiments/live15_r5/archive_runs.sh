#!/usr/bin/env bash
# Copy the small, reviewable artefacts of each live15 run dir into the docs folder (tmp/ is git-ignored):
# the config record, driver log, startup/verify/warm lines, trace reports and the load tester's summaries.
#   archive_runs.sh <runs root> <docs runs dir>
SRC=$1; DST=$2
for d in $SRC/*/; do
  n=$(basename $d); [ -f $d/README.txt ] || continue
  mkdir -p $DST/$n
  cp $d/README.txt $d/driver.log $DST/$n/ 2>/dev/null
  cp $d/startup_lines.txt $d/verify.txt $d/warm.txt $DST/$n/ 2>/dev/null
  cp $d/*_trace.md $d/*_trace.json $DST/$n/ 2>/dev/null
  cp $d/lt2/*summary*.json $DST/$n/ 2>/dev/null
done
du -sh $DST | tail -1
