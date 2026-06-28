#!/usr/bin/env bash
# Phase 2: run all 5 methods on 7 iconic style families baselines don't yet cover
# (02 anime, 08 chinese-ink, 12 cyberpunk, 19 graffiti, 34 stained-glass,
#  35 steampunk, 37 ukiyo-e) x 5 kp prompts, then compute CSD/CLIP-T metrics.
# Outputs append to the same d_grand/ subdirs and results/ext_*.csv (no conflicts).
# Mirrors run_all_new6.sh; style set injected via STYLES_OVERRIDE.
cd /home/ubuntu/chengyanli/items/save/attention_20260426_090948

export STYLES_OVERRIDE="02,08,12,19,34,35,37"
PYTHON=/home/ubuntu/chengyanli/flux/bin/python3
SENSENOVA=/home/ubuntu/miniconda3/envs/sensenova/bin/python
LONGCAT=/home/ubuntu/miniconda3/envs/longcat-video/bin/python

log() { echo "[iconic7] $(date '+%H:%M:%S') $*"; }
step() { log "=== $1 ==="; eval "$2"; log "=== $1 exit=$? ==="; }

log "STYLES_OVERRIDE=$STYLES_OVERRIDE"
step "1/6 Ours hybrid2"  "$SENSENOVA scripts/eval_all.py --task t2i_new6styles"
step "2/6 StyleShot"     "$LONGCAT scripts/run_new6_styleshot.py"
step "3/6 CSGO"          "$PYTHON scripts/run_new6_csgo.py"
step "4/6 USO"           "$PYTHON scripts/run_new6_uso.py"
step "5/6 IP-Adapter"    "$PYTHON scripts/run_new6_ipadapter.py"
step "6/6 Metrics"       "$SENSENOVA scripts/metrics_new6_all.py"
log "ALL DONE"
