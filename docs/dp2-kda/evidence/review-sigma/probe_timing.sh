set -e
export ROOT=/scratch/users/ericrcwu/agent-runs/dp2-kda-p0
export KDA_PROBES_DIR=$ROOT/probes
export PYTHONPATH=$ROOT/OLMo-core/src
OUT=/scratch/users/ericrcwu/agent-runs/review-sigma
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== A: s5_words 4000 steps R1 ==="
/usr/bin/time -f "WALL %e" $ROOT/venv/bin/python $ROOT/probes/train_probe.py \
  --arm R1 --task s5_words --bundle-id 1101 --steps 4000 \
  --eval-lengths 40 64 128 256 --out $OUT/out/timing_r1.json 2>&1 | tail -12
echo "=== B: mqar_p8 feasibility, train 41-64 ==="
$ROOT/venv/bin/python $ROOT/probes/train_probe.py \
  --arm R1 --task mqar_p8 --bundle-id 1101 --steps 300 \
  --train-min 41 --train-max 64 --eval-lengths 64 96 128 --out $OUT/out/timing_mqar.json 2>&1 | tail -8
echo "=== C: mqar_p16 feasibility, train 65-96 ==="
$ROOT/venv/bin/python $ROOT/probes/train_probe.py \
  --arm R1 --task mqar_p16 --bundle-id 1101 --steps 300 \
  --train-min 65 --train-max 96 --eval-lengths 96 128 --out $OUT/out/timing_mqar16.json 2>&1 | tail -8
