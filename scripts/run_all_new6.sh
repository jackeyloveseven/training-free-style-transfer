#!/usr/bin/env bash
# Run all 5 methods on new 6 diverse styles, then compute metrics.
# Results go to existing d_grand/ subdirs (no filename conflicts).
set -e
cd /home/ubuntu/chengyanli/items/save/attention_20260426_090948

PYTHON=/home/ubuntu/chengyanli/flux/bin/python3
SENSENOVA=/home/ubuntu/miniconda3/envs/sensenova/bin/python
LONGCAT=/home/ubuntu/miniconda3/envs/longcat-video/bin/python

log() { echo "[run_all_new6] $(date '+%H:%M:%S') $*"; }

log "=== Step 1/5: ours hybrid2 (Klein 9B) ==="
$SENSENOVA scripts/eval_all.py --task t2i_new6styles
log "=== Step 1 DONE ==="

log "=== Step 2/5: StyleShot ==="
$LONGCAT scripts/run_new6_styleshot.py
log "=== Step 2 DONE ==="

log "=== Step 3/5: CSGO (fixed params) ==="
$PYTHON scripts/run_new6_csgo.py
log "=== Step 3 DONE ==="

log "=== Step 4/5: USO ==="
$PYTHON scripts/run_new6_uso.py
log "=== Step 4 DONE ==="

log "=== Step 5/5: IP-Adapter ==="
$PYTHON scripts/run_new6_ipadapter.py
log "=== Step 5 DONE ==="

log "=== Computing metrics for all new6 outputs ==="
$SENSENOVA scripts/metrics_new6_all.py
log "=== ALL DONE ==="
