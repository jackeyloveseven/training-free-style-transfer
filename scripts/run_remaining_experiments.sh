#!/bin/bash
# Wait for eval_all.py to finish, then run remaining experiments sequentially.
# Usage: nohup bash run_remaining_experiments.sh > remaining_exp.log 2>&1 &

PYTHON="/home/ubuntu/miniconda3/envs/sensenova/bin/python"
CWD="/home/ubuntu/chengyanli/items/save/attention_20260426_090948"

echo "$(date): Waiting for eval_all.py (PID 1486037) to finish..."
while kill -0 1486037 2>/dev/null; do
    sleep 60
done
echo "$(date): eval_all.py finished. Starting remaining experiments."

echo ""
echo "========================================================================"
echo "$(date): EXPERIMENT 2 — Iters Sweep + SLT Sweep + Adam-Only + Spatial Smoothing"
echo "========================================================================"
cd "$CWD"
$PYTHON eval_iters_and_smoothing.py --task all 2>&1 | tee results/eval_iters_smoothing.log

echo ""
echo "========================================================================"
echo "$(date): EXPERIMENT 3 — Attention Map PCA Visualization"
echo "========================================================================"
$PYTHON eval_attn_pca.py 2>&1 | tee results/eval_attn_pca.log

echo ""
echo "========================================================================"
echo "$(date): ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
