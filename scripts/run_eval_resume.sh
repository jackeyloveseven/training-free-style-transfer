#!/bin/bash
# Resume eval_all.py for the parts that failed (t2i_full + i2i)
# Wait for any running python process to finish first

PYTHON="/home/ubuntu/miniconda3/envs/sensenova/bin/python"
CWD="/home/ubuntu/chengyanli/items/save/attention_20260426_090948"

echo "$(date): Waiting for PCA experiment to finish..."
while pgrep -f "eval_attn_pca" > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): PCA done. Starting eval_all resume."

cd "$CWD"

echo "$(date): Running t2i_full..."
$PYTHON eval_all.py --task t2i_full 2>&1 | tee results/eval_t2i_full_log.txt

echo "$(date): Running i2i_ablation..."
$PYTHON eval_all.py --task i2i_ablation 2>&1 | tee results/eval_i2i_ablation_log.txt

echo "$(date): Running i2i_baselines..."
$PYTHON eval_all.py --task i2i_baselines 2>&1 | tee results/eval_i2i_baselines_log.txt

echo "$(date): Running i2i_full..."
$PYTHON eval_all.py --task i2i_full 2>&1 | tee results/eval_i2i_full_log.txt

echo "$(date): ALL RESUME TASKS COMPLETE"
