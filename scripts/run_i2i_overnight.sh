#!/bin/bash
# Overnight I2I baseline comparison: run remaining baselines sequentially (single
# GPU), then evaluate every method (4 baselines + our 4 seeds) with i2i_eval_all.
# IP-Adapter is already running when this launches; we wait for it first.
set -u
cd /home/ubuntu/chengyanli/items/save/attention_20260426_090948
PY=/home/ubuntu/chengyanli/flux/bin/python3
mkdir -p logs
echo "=== overnight start $(date) ===" > logs/i2i_overnight.log

# 1. wait for the already-running IP-Adapter to finish
while pgrep -f i2i_run_ipadapter.py >/dev/null 2>&1; do sleep 20; done
echo "ipadapter done $(date)" >> logs/i2i_overnight.log

# 2. remaining baselines, sequential (idempotent / skip-if-exists)
for run in qwen_edit csgo uso; do
  echo "START $run $(date)" >> logs/i2i_overnight.log
  $PY scripts/i2i_run_${run}.py > logs/i2i_${run}.log 2>&1
  echo "END   $run $(date) (exit $?)" >> logs/i2i_overnight.log
done

# 3. evaluate everything (per-method summary printed to each eval log)
echo "=== EVAL $(date) ===" >> logs/i2i_overnight.log
$PY scripts/i2i_eval_all.py ext_ipadapter d_grand/i2i_ext_ipadapter 'ipadapter_cnt{cnt}_sty{sty}.png' > logs/eval_ipadapter.log 2>&1
$PY scripts/i2i_eval_all.py ext_qwen      d_grand/i2i_ext_qwen_edit 'qwen_edit_cnt{cnt}_sty{sty}.png' > logs/eval_qwen.log 2>&1
$PY scripts/i2i_eval_all.py ext_csgo      d_grand/i2i_ext_csgo      'csgo_cnt{cnt}_sty{sty}.png'      > logs/eval_csgo.log 2>&1
$PY scripts/i2i_eval_all.py ext_uso       d_grand/i2i_ext_uso       'uso_cnt{cnt}_sty{sty}_0.png'     > logs/eval_uso.log 2>&1
$PY scripts/i2i_eval_all.py ours_s43 d_grand/i2i_ablation     'hybrid2_cnt{cnt}_sty{sty}.png' > logs/eval_ours43.log 2>&1
$PY scripts/i2i_eval_all.py ours_s44 d_grand/i2i_ours_seed44  'hybrid2_cnt{cnt}_sty{sty}.png' > logs/eval_ours44.log 2>&1
$PY scripts/i2i_eval_all.py ours_s45 d_grand/i2i_ours_seed45  'hybrid2_cnt{cnt}_sty{sty}.png' > logs/eval_ours45.log 2>&1
$PY scripts/i2i_eval_all.py ours_s46 d_grand/i2i_ours_seed46  'hybrid2_cnt{cnt}_sty{sty}.png' > logs/eval_ours46.log 2>&1

# 4. collect summaries
echo "=== SUMMARY $(date) ===" >> logs/i2i_overnight.log
grep -h '===' logs/eval_*.log -A1 | grep -E 'CSD|===' >> logs/i2i_overnight.log
echo "ALL DONE $(date)" | tee logs/i2i_overnight_DONE.txt >> logs/i2i_overnight.log
