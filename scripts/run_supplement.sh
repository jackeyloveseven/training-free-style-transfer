#!/bin/bash
PYTHON="/home/ubuntu/miniconda3/envs/sensenova/bin/python"
CWD="/home/ubuntu/chengyanli/items/save/attention_20260426_090948"

echo "$(date): Waiting for all other experiments to finish..."
while pgrep -f "eval_all.py|eval_iters|eval_attn|eval_spatial" > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): GPU free. Starting supplement experiments."

cd "$CWD"
$PYTHON eval_supplement.py --task all 2>&1 | tee results/eval_supplement.log

echo "$(date): SUPPLEMENT COMPLETE"
