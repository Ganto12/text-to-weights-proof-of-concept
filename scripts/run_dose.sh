#!/usr/bin/env bash
# Dose-response meta-model run (signal hunt, 2026-11-18 20:25 UTC; stability fix on).
# Usage: run_dose.sh <N teachers or 0 for all> <label, e.g. strict> <tasks file>
# stage 1 (1500 steps) -> stage 2 (1500 steps, lr 2e-5, clamp, format lines; validation + probe every 500
# on 5 benchmarks x 300 POOL rows) -> full test on the same 5 benchmarks for C and R with the
# checkpoint picked by SNI-validation loss. Rows go to the timeline with note "teachers=N; set=...".
set -uo pipefail
ROOT=/workspace/poc
export HF_HOME=$ROOT/hf_cache PYTHONUNBUFFERED=1
N=$1; SET=$2; TASKS=$3; EXTRA=${4:-}
cd $ROOT/scripts
NT=$(python -c "
import json,os
t=json.load(open('$ROOT/$TASKS'))['tasks']
p=[x for x in t if os.path.exists(f'$ROOT/teachers/{x}/metrics.json') and json.load(open(f'$ROOT/teachers/{x}/metrics.json')).get('passed')]
print(min(len(p), $N) if $N else len(p))")
TAG="dose_t${NT}_${SET}${S2_STEPS:+_s2x${S2_STEPS}}${SEED:+_seed${SEED}}"; LBL="teachers=${NT}; set=${SET}${SEED:+; seed=${SEED}}"
OUT=$ROOT/meta/$TAG
BENCH=comve,winogrande,bbh_logical_deduction_three_objects,gsm8k_final_number,arc_challenge
echo "[dose] $TAG start $(date -u +%T)"
python train_meta.py --stage 1 --steps 1500 --seed ${SEED:-0} --tasks-file $TASKS --max-teachers $NT ${EXTRA:+--extra-tasks $EXTRA} \
  --label "$LBL" --out $OUT/stage1.pt 2>&1 | tee $ROOT/logs/${TAG}_s1.log || exit 1
python train_meta.py --stage 2 --seed ${SEED:-0} --steps ${S2_STEPS:-1500} --probe-every 500 --probe-benches $BENCH --probe-n 300 \
  --tasks-file $TASKS --max-teachers $NT ${EXTRA:+--extra-tasks $EXTRA} \
  --label "$LBL" --init $OUT/stage1.pt --out $OUT/stage2 2>&1 | tee $ROOT/logs/${TAG}_s2.log || exit 1
CK=$(python -c "import json;print(json.load(open('$OUT/stage2/selected.json'))['selected'])")
STEP=$(python -c "import json;print(json.load(open('$OUT/stage2/selected.json'))['step'])")
echo "[dose] $TAG test with $CK $(date -u +%T)"
for a in C R; do python eval_arms.py --arm $a --benches $BENCH --tag ${TAG}_s${STEP} --meta $CK 2>&1 | tee -a $ROOT/logs/${TAG}_eval.log; done
python eval_arms.py --arm C --benches $BENCH --tag ${TAG}_stage1only --meta $OUT/stage1.pt 2>&1 | tee -a $ROOT/logs/${TAG}_eval.log
python timeline.py --sync
echo "[dose] $TAG DONE $(date -u +%T)"
