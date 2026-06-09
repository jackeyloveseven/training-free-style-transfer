#!/bin/bash
# Final resume: pick up remaining images from crashed tasks + spatial viz
PYTHON="/home/ubuntu/miniconda3/envs/sensenova/bin/python"
CWD="/home/ubuntu/chengyanli/items/save/attention_20260426_090948"

echo "$(date): Waiting for current i2i_full to finish..."
while pgrep -f "eval_all.py" > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): Done. Starting final resume."

cd "$CWD"

echo "$(date): t2i_full (resume from 253/456)..."
$PYTHON eval_all.py --task t2i_full 2>&1 | tee results/eval_t2i_full_resume.txt

echo "$(date): i2i_ablation (resume from 13/150)..."
$PYTHON eval_all.py --task i2i_ablation 2>&1 | tee results/eval_i2i_ablation_resume.txt

echo "$(date): i2i_baselines..."
$PYTHON eval_all.py --task i2i_baselines 2>&1 | tee results/eval_i2i_baselines_resume.txt

echo "$(date): i2i_full (resume from 676/722)..."
$PYTHON eval_all.py --task i2i_full 2>&1 | tee results/eval_i2i_full_resume.txt

echo "$(date): Spatial smoothing visualization..."
$PYTHON eval_spatial_viz.py 2>&1 | tee results/eval_spatial_viz.log

echo "$(date): ALL FINAL TASKS COMPLETE"
