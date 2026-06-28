#!/bin/bash
# Rerun IP-Adapter (strength 0.8/scale 1.5) + Qwen (stronger instruction), then re-eval.
set -u
cd /home/ubuntu/chengyanli/items/save/attention_20260426_090948
PY=/home/ubuntu/chengyanli/flux/bin/python3
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs
echo "=== rerun start $(date) ===" > logs/i2i_rerun.log
echo "START ipadapter $(date)" >> logs/i2i_rerun.log
$PY scripts/i2i_run_ipadapter.py >> logs/i2i_rerun.log 2>&1
echo "START qwen $(date)" >> logs/i2i_rerun.log
$PY scripts/i2i_run_qwen_edit.py >> logs/i2i_rerun.log 2>&1
echo "=== EVAL $(date) ===" >> logs/i2i_rerun.log
$PY scripts/i2i_eval_all.py ext_ipadapter d_grand/i2i_ext_ipadapter 'ipadapter_cnt{cnt}_sty{sty}.png' >> logs/i2i_rerun.log 2>&1
$PY scripts/i2i_eval_all.py ext_qwen d_grand/i2i_ext_qwen_edit 'qwen_edit_cnt{cnt}_sty{sty}.png' >> logs/i2i_rerun.log 2>&1
echo "RERUN_ALL_DONE $(date)" | tee logs/i2i_rerun_DONE.txt >> logs/i2i_rerun.log
