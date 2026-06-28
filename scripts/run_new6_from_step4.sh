#!/usr/bin/env bash
# Continue from step 4 (our method + StyleShot + CSGO already done).
set -e
cd /home/ubuntu/chengyanli/items/save/attention_20260426_090948

PYTHON=/home/ubuntu/chengyanli/flux/bin/python3
SENSENOVA=/home/ubuntu/miniconda3/envs/sensenova/bin/python

log() { echo "[new6] $(date '+%H:%M:%S') $*"; }

log "=== Step 4/5: USO ==="
$PYTHON scripts/run_new6_uso.py
log "=== Step 4 DONE ==="

log "=== Step 5/5: IP-Adapter ==="
$PYTHON scripts/run_new6_ipadapter.py
log "=== Step 5 DONE ==="

log "=== Computing metrics ==="
$SENSENOVA scripts/metrics_new6_all.py
log "=== ALL DONE ==="
