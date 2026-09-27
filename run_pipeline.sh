#!/bin/bash
# run_pipeline.sh - W-JEPA benchmark pipeline: pretrain, fine-tune and evaluate.
#
# For each target dataset, pretrain W-JEPA on PRETRAIN_SOURCES without the target
# (leave-one-target-out), then fine-tune and evaluate on the target with a random
# window split and a disjoint participant split. The "all6" entry pretrains one
# reusable encoder on all six datasets and skips fine-tuning. Targets are queued
# across GPUS, one run per GPU at a time.
#
# Re-running resumes unfinished stages from their checkpoint_latest.pth; finished
# stages resume past their final epoch and only rewrite metrics.json. Use a fresh
# OUT_BASE for a new experiment.
#
# Usage:
#   GPUS="0,1,2,3" ./run_pipeline.sh
#   GPUS="0" DATASETS="dsads uci_har" ./run_pipeline.sh
#
# Environment variables, with defaults:
#   GPUS                  comma-separated GPU ids              (default "0")
#   DATASETS              space-separated targets              (default: all six + all6)
#   PRETRAIN_SOURCES      comma-separated pretraining sources  (default: the six benchmarks)
#   PRETRAIN_EPOCHS       pretraining epochs                   (default 50)
#   FINETUNE_EPOCHS       fine-tuning epochs                   (default 70)
#   BUDGET                pretraining windows per epoch        (default 34000)
#   SAMPLING_RATE         session resampling rate in Hz        (default 100)
#   WINDOW_SIZE           samples per window                   (default 500)
#   WINDOW_STRIDE         samples between windows              (default 500)
#   SPLIT_STRATEGIES      fine-tuning splits                   (default "window participant")
#   DATA_PATH             dataset root                         (default ./datasets)
#   OUT_BASE              checkpoint and metrics root          (default ./checkpoints)
#   PARTICIPANT_OUT_BASE  participant-split outputs            (default $OUT_BASE/finetune_participant)
#   LOG_DIR               per-target log directory             (default ./logs)
#   NUM_WORKERS           dataloader workers per run           (default 6)
#   SEED                  optional training RNG seed           (default unset)
#   SPLIT_SEED            seed for data selection and splits   (default 42)
#   WANDB_MODE            online, offline or disabled          (default disabled)
#   WANDB_PROJECT         W&B project                          (default wavelet-jepa)
#   WANDB_ENTITY          optional W&B account/team
#   WANDB_RUN_GROUP       optional prefix for W&B run names
set -uo pipefail
cd "$(dirname "$0")"

GPUS="${GPUS:-0}"; GPUS="${GPUS// /}"
DATASETS="${DATASETS:-pamap2 dsads mhealth realworld uci_har usc_had all6}"
PRETRAIN_SOURCES="${PRETRAIN_SOURCES:-pamap2,dsads,mhealth,realworld,uci_har,usc_had}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-70}"
BUDGET="${BUDGET:-34000}"
SAMPLING_RATE="${SAMPLING_RATE:-100}"
WINDOW_SIZE="${WINDOW_SIZE:-500}"
WINDOW_STRIDE="${WINDOW_STRIDE:-500}"
SPLIT_STRATEGIES="${SPLIT_STRATEGIES:-window participant}"
DATA_PATH="${DATA_PATH:-./datasets}"
OUT_BASE="${OUT_BASE:-./checkpoints}"
PARTICIPANT_OUT_BASE="${PARTICIPANT_OUT_BASE:-$OUT_BASE/finetune_participant}"
LOG_DIR="${LOG_DIR:-./logs}"
NUM_WORKERS="${NUM_WORKERS:-6}"

# Arguments shared by every pretraining and fine-tuning run.
common_args=(--data_path "$DATA_PATH" --num_workers "$NUM_WORKERS" --device cuda
             --window_size "$WINDOW_SIZE" --window_stride "$WINDOW_STRIDE"
             --split_seed "${SPLIT_SEED:-42}" --wandb_mode "${WANDB_MODE:-disabled}"
             --wandb_project "${WANDB_PROJECT:-wavelet-jepa}")
[ -n "$SAMPLING_RATE" ] && common_args+=(--sampling_rate "$SAMPLING_RATE")
[ -n "${WANDB_ENTITY:-}" ] && common_args+=(--wandb_entity "$WANDB_ENTITY")
if [ -n "${SEED:-}" ]; then
  common_args+=(--seed "$SEED")
  export PYTHONHASHSEED="$SEED"
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
read -ra DS_ARR <<< "$DATASETS"
read -ra SPLIT_ARR <<< "$SPLIT_STRATEGIES"
if [ "${#GPU_ARR[@]}" -eq 0 ]; then
  echo "No GPUs specified. Set GPUS, for example GPUS=\"0\"." >&2
  exit 1
fi
if [ "${#DS_ARR[@]}" -eq 0 ]; then
  echo "No datasets specified. Set DATASETS or use the default." >&2
  exit 1
fi
for split in "${SPLIT_ARR[@]}"; do
  if [ "$split" != "window" ] && [ "$split" != "participant" ]; then
    echo "Unknown split strategy: $split" >&2
    exit 1
  fi
done
mkdir -p "$LOG_DIR" "$OUT_BASE"

# run_stage NAME SCRIPT OUT_DIR ARGS... runs one training stage, resuming from
# OUT_DIR/checkpoint_latest.pth when an earlier invocation left one behind.
run_stage() {
  local name="$1" script="$2" out="$3"
  shift 3
  if [ -f "$out/checkpoint_latest.pth" ]; then
    set -- "$@" --resume "$out/checkpoint_latest.pth"
  fi
  WANDB_NAME="${WANDB_RUN_GROUP:-wjepa}-$name" \
    python "$script" "${common_args[@]}" "$@" --output_dir "$out"
}

# run_one TARGET GPU: leave-one-target-out pretraining, then one fine-tuning run per split.
run_one() {
  local ds="$1" gpu="$2"
  local pretrain_out="$OUT_BASE/pretrain/$ds" exclude="$ds"
  if [ "$ds" = "all6" ]; then
    pretrain_out="$OUT_BASE/pretrain_all"
    exclude=""
  fi
  export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
  {
    echo "=== [$ds] $(date -u +%FT%TZ) GPU $gpu | pretrain epochs=$PRETRAIN_EPOCHS budget=$BUDGET ==="
    echo "sources=$PRETRAIN_SOURCES exclude=$exclude"
    run_stage "pretrain-$ds" main_pretrain.py "$pretrain_out" \
      --whar_pretrain_datasets "$PRETRAIN_SOURCES" --whar_exclude_datasets "$exclude" \
      --epochs "$PRETRAIN_EPOCHS" --budget "$BUDGET" || {
        echo "[$ds] PRETRAIN FAILED"
        return 1
      }
    if [ "$ds" = "all6" ]; then
      echo "[$ds] DONE"
      return 0
    fi

    # Train-only jitter/scale augmentation for every target except MHealth.
    local aug="--finetune_aug"
    [ "$ds" = "mhealth" ] && aug=""
    local split finetune_out
    for split in "${SPLIT_ARR[@]}"; do
      finetune_out="$OUT_BASE/finetune/$ds"
      [ "$split" = "participant" ] && finetune_out="$PARTICIPANT_OUT_BASE/$ds"
      echo "=== [$ds] GPU $gpu | fine-tune split=$split epochs=$FINETUNE_EPOCHS aug=$([ -n "$aug" ] && echo on || echo off) ==="
      run_stage "finetune-$ds-$split" main_finetune.py "$finetune_out" \
        --dataset "$ds" --pretrained "$pretrain_out/checkpoint_latest.pth" $aug \
        --epochs "$FINETUNE_EPOCHS" --split_strategy "$split" || {
          echo "[$ds] FINETUNE $split FAILED"
          return 1
        }
    done
    echo "[$ds] DONE"
  } >> "$LOG_DIR/$ds.log" 2>&1
}

echo "GPUs: ${GPU_ARR[*]}  |  datasets: ${DS_ARR[*]}"
echo "pretrain_sources=$PRETRAIN_SOURCES"
echo "pretrain_epochs=$PRETRAIN_EPOCHS  finetune_epochs=$FINETUNE_EPOCHS  budget=$BUDGET"
echo "sampling_rate=${SAMPLING_RATE:-native}Hz  window=$WINDOW_SIZE samples  stride=$WINDOW_STRIDE"
echo "outputs: $OUT_BASE/{pretrain,pretrain_all,finetune}  $PARTICIPANT_OUT_BASE"
echo "logs:    $LOG_DIR/<dataset>.log"
echo ""

# One queue slot per GPU; a free slot takes the next target.
slot_pid=()
slot_ds=()
next=0
failed=0
while :; do
  busy=0
  for i in "${!GPU_ARR[@]}"; do
    pid="${slot_pid[$i]:-}"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        echo ">> ${slot_ds[$i]} finished on GPU ${GPU_ARR[$i]}"
      else
        echo ">> ${slot_ds[$i]} failed on GPU ${GPU_ARR[$i]}; check $LOG_DIR/${slot_ds[$i]}.log" >&2
        failed=1
      fi
      pid=""
    fi
    if [ -z "$pid" ] && [ "$next" -lt "${#DS_ARR[@]}" ]; then
      slot_ds[$i]="${DS_ARR[$next]}"
      next=$((next + 1))
      echo ">> launching ${slot_ds[$i]} on GPU ${GPU_ARR[$i]}"
      run_one "${slot_ds[$i]}" "${GPU_ARR[$i]}" &
      pid=$!
    fi
    slot_pid[$i]="$pid"
    [ -n "$pid" ] && busy=1
  done
  [ "$busy" -eq 0 ] && break
  sleep 5
done

echo ""
echo "=== final-epoch test metrics ==="
python - "$OUT_BASE" "$PARTICIPANT_OUT_BASE" "$SPLIT_STRATEGIES" "${DS_ARR[@]}" <<'PYTHON'
import json
from pathlib import Path
import sys
out, participant, splits = sys.argv[1:4]
for split in splits.split():
    for ds in sys.argv[4:]:
        if ds == 'all6':
            continue
        root = Path(participant) if split == 'participant' else Path(out) / 'finetune'
        p = root / ds / 'metrics.json'
        m = json.loads(p.read_text()) if p.exists() else {}
        if not m.get('complete'):
            print(f'{split} {ds}: incomplete')
        else:
            latest = m['latest']
            print(f"{split} {ds}: epoch {latest['epoch']}, accuracy {latest['accuracy']:.2f}, macro-F1 {latest['macro_f1']:.2f}")
PYTHON
exit "$failed"
