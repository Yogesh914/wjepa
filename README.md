<p align="center">
    <img src="assets/apple.png" width="100" style="margin-bottom: 0.2;"/>
</p>

<h2 align="center">W-JEPA: Self-Supervised Learning for Wearable Accelerometer Data</h2>

<h5 align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</h5>

[W-JEPA](https://yogesh914.github.io/wjepa/) is a self-supervised model for wearable human activity recognition. This
repository pretrains, fine-tunes, and evaluates it on six benchmarks: PAMAP2, DSADS,
MHealth, RealWorld, UCI-HAR, and USC-HAD.

## How it works

1. **Tokenize.** Each 5-second window (500 samples at 100 Hz) is decomposed channel by
   channel with a level-4 `db4` wavelet transform. Each of the 5 wavelet bands is pooled
   into 2 tokens holding a (mean, energy) pair, giving 10 tokens per channel.
2. **Pretrain.** About 65% of the (band, channel) token pairs are masked. A transformer
   encoder sees the remaining tokens, and a predictor estimates the masked tokens'
   representations from an EMA copy of the encoder (L1 loss + variance hinge).
   Pretraining uses a standard six-channel accelerometer + gyroscope view of each dataset.
3. **Fine-tune and evaluate.** The pretrained encoder and a linear head over all tokens
   are fine-tuned on every channel of a target dataset, then scored on its held-out test
   windows (accuracy and macro-F1).

## Codebase

| File | Purpose |
|---|---|
| `run_pipeline.sh` | Runs the full benchmark pipeline across GPUs. |
| `download_datasets.py` | Downloads and prepares the six benchmark datasets. |
| `main_pretrain.py`, `engine_pretrain.py` | JEPA pretraining. |
| `main_finetune.py`, `engine_finetune.py` | Fine-tuning with test evaluation after every epoch. |
| `inference.py` | Re-evaluates a fine-tuned checkpoint, predicts labels, or extracts embeddings. |
| `models_jepa.py` | Wavelet tokenizer, transformer encoder, and predictor. |
| `config.py` | Model and optimizer defaults, benchmark channel layouts. |
| `util/` | Data loading and splits, masking, augmentation, checkpointing, W&B logging. |

## Setup

```bash
pip install -r requirements.txt
python download_datasets.py        # all six benchmarks into ./datasets
```

`requirements.txt` pins `whar-datasets` to the revision this code was built against.
The downloader parses MHealth and USC-HAD in a fixed file order, so every machine
gets the same splits.

## Run the pipeline

```bash
GPUS="0,1,2,3" ./run_pipeline.sh
```

For each benchmark, the pipeline:

1. pretrains W-JEPA for **50 epochs** on the other five datasets (leave-one-target-out);
2. fine-tunes that encoder for **70 epochs** on the target, once with a stratified random
   window split and once with a disjoint participant split (20% train, seed 42);
3. evaluates on the test split after every epoch and reports the final epoch.

It also pretrains one reusable encoder on all six datasets (`all6`). Targets run in
parallel, one per GPU, and a summary of test accuracy and macro-F1 is printed at the end.

```text
checkpoints/pretrain/<dataset>/               leave-one-target-out encoder
checkpoints/pretrain_all/                     encoder pretrained on all six datasets
checkpoints/finetune/<dataset>/               window-split model
checkpoints/finetune_participant/<dataset>/   participant-split model
logs/<dataset>.log
```

Each directory holds `checkpoint_latest.pth` and a `metrics.json` with the per-epoch
history and final test metrics. Re-running the script resumes unfinished stages from
their checkpoints, so use a new `OUT_BASE` for a new experiment. Common overrides (see
the top of `run_pipeline.sh` for all of them):

```bash
GPUS="0" DATASETS="dsads uci_har" ./run_pipeline.sh   # only some targets
SEED=0 ./run_pipeline.sh                               # seeded training
WANDB_MODE=online ./run_pipeline.sh                    # log to Weights & Biases
OUT_BASE=./runs/exp1 LOG_DIR=./runs/exp1/logs ./run_pipeline.sh
```

## Run a single stage

```bash
# Pretrain on the six benchmarks except DSADS
python main_pretrain.py --sampling_rate 100 --whar_exclude_datasets dsads \
  --output_dir checkpoints/pretrain/dsads

# Fine-tune and evaluate on DSADS (window split; add --split_strategy participant for the other)
python main_finetune.py --dataset dsads --sampling_rate 100 --finetune_aug \
  --pretrained checkpoints/pretrain/dsads/checkpoint_latest.pth \
  --output_dir checkpoints/finetune/dsads
```

The Python entrypoints default to each dataset's native sampling rate, so pass
`--sampling_rate 100` to match the pipeline. The pipeline uses `--finetune_aug` for
every dataset except MHealth. `--resume <checkpoint>` continues an interrupted run.

## Evaluate a checkpoint

```bash
# Re-evaluate on the test split saved in the checkpoint
python inference.py classify --checkpoint checkpoints/finetune/dsads/checkpoint_latest.pth

# Predict labels for raw windows shaped [N, C, T], with channels ordered as in config.py
python inference.py classify --checkpoint checkpoints/finetune/dsads/checkpoint_latest.pth \
  --input windows.npy --output preds.npy

# Extract embeddings with a pretrained encoder
python inference.py embed --checkpoint checkpoints/pretrain_all/checkpoint_latest.pth \
  --input windows.npy --output emb.npy
```

Add `--device cpu` to run without a GPU.

## Default settings

| | |
|---|---|
| Data | 100 Hz, 500-sample windows, stride 500, no normalization |
| Encoder | Transformer, embed dim 384, depth 8, 6 heads |
| Predictor | Embed dim 192, depth 4, 6 heads |
| Pretraining | 50 epochs, batch 256, AdamW (lr 2e-4, 8 warmup epochs, cosine to 1e-6, weight decay 0.05), up to 34,000 windows per epoch, mask ratio 0.65, EMA 0.998 → 1.0, variance-hinge weight 0.0625 |
| Fine-tuning | 70 epochs, batch 64, AdamW (lr 1e-4, 10 warmup epochs, cosine to 1e-6, weight decay 0.05), dropout 0.1, label smoothing 0.1, jitter/scale augmentation except MHealth |
