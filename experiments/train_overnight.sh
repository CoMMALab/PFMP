#!/usr/bin/env bash
# Train mjlab learned proposals overnight, sequentially (one GPU).
# Usage:  nohup bash experiments/train_overnight.sh > logs/overnight/run.log 2>&1 &
# In the morning, mjx_runner auto-resolves the newest checkpoint per task.

set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled
# pfmp (JAX) is imported at module load; stop JAX preallocating ~90% of the GPU
# so the PyTorch PPO trainer has room (otherwise the box can OOM/crash).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

TS=$(date +%Y-%m-%d_%H-%M-%S)
LOGDIR="logs/overnight"
mkdir -p "$LOGDIR"

run() {  # name  script  iters
    local name="$1" script="$2" iters="$3"
    echo "==== [$(date +%H:%M:%S)] START $name (iters=$iters) ===="
    python3 "$script" --train --num-envs 1024 --iters "$iters" \
        > "$LOGDIR/${TS}_${name}.log" 2>&1
    echo "==== [$(date +%H:%M:%S)] DONE  $name (exit $?) ===="
}

run cartpole        examples/mjlab_cartpole.py        2000
run double_cartpole examples/mjlab_double_cartpole.py 2000

# Added once the task definitions exist + pass a validation run:
# run halfcheetah    examples/mjlab_halfcheetah.py     3000
# run walker         examples/mjlab_walker.py          3000

echo "==== [$(date +%H:%M:%S)] ALL DONE ===="
