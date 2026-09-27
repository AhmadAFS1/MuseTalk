#!/bin/bash
# usage: status.sh <label>  -> appends load/GPU snapshot to contamination log
D=/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/throughput300/cpu_probe
{
echo "=== $1 @ $(date +%T)"
uptime
nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,memory.used,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
ps -eo pcpu,pid,comm --sort=-pcpu | head -6 | tail -5
} | tee -a $D/contamination_log.txt
