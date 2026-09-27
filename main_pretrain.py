"""Wavelet W-JEPA pretraining on the standard six-channel view of the source datasets.

Base encoder, db4 level-4 wavelet tokenizer, masking of (band, channel) token pairs,
EMA target encoder, L1 latent prediction + variance hinge, AdamW, raw windows.
"""
import argparse
import copy
import math
import os

import torch
import wandb

from config import (
    BENCHMARK_DATASETS, PRETRAIN_CONFIG, canonical_dataset_name, get_dataset_config,
)
from models_jepa import sensor_jepa_base
from util.datasets import load_whar_combined_dataset
from util.masking import RandomTubeMasking1D
from util.checkpointing import capture_rng_state, restore_rng_state, validate_resume_args, write_json
from util.experiment import start_run, report_progress
from engine_pretrain import (
    train_one_epoch, cosine_scheduler, save_checkpoint, load_checkpoint,
)


def get_args_parser():
    parser = argparse.ArgumentParser('Wavelet W-JEPA Pretraining', add_help=False)

    # Optimization
    parser.add_argument('--batch_size', default=PRETRAIN_CONFIG['batch_size'], type=int)
    parser.add_argument('--epochs', default=PRETRAIN_CONFIG['epochs'], type=int)
    parser.add_argument('--warmup_epochs', default=PRETRAIN_CONFIG['warmup_epochs'], type=int)
    parser.add_argument('--lr', default=PRETRAIN_CONFIG['lr'], type=float)
    parser.add_argument('--min_lr', default=PRETRAIN_CONFIG['min_lr'], type=float)
    parser.add_argument('--weight_decay', default=PRETRAIN_CONFIG['weight_decay'], type=float)
    parser.add_argument('--reg_coeff', default=0.0625, type=float,
                        help='Variance-hinge regularization coefficient')
    parser.add_argument('--mask_ratio', default=0.65, type=float,
                        help='Fraction of wavelet tokens used as prediction targets')
    parser.add_argument('--ema_momentum_start', default=0.998, type=float)
    parser.add_argument('--ema_momentum_end', default=1.0, type=float)

    # Data
    parser.add_argument('--data_path', default='./datasets', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--budget', default=34000, type=int,
                        help='Balanced windows sampled per epoch across source datasets')
    parser.add_argument('--train_ratio', default=1.0, type=float,
                        help='Fraction of each source dataset used for pretraining')
    parser.add_argument('--window_size', default=500, type=int)
    parser.add_argument('--window_stride', default=500, type=int)
    parser.add_argument('--sampling_rate', default=None, type=float,
                        help='Resample each continuous session to this rate in Hz before windowing')
    parser.add_argument('--split_seed', default=42, type=int)
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed Python, NumPy and PyTorch; unset preserves stochastic initialization')
    parser.add_argument('--whar_pretrain_datasets', default='', type=str,
                        help='Comma-separated source datasets; defaults to the six benchmarks')
    parser.add_argument('--whar_exclude_datasets', default='', type=str,
                        help='Comma-separated source datasets to exclude (e.g. leave-one-target-out)')

    # I/O
    parser.add_argument('--output_dir', default='./checkpoints/pretrain', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--wandb_project', default='wavelet-jepa', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_mode', default='disabled', type=str,
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--device', default='cuda', type=str)
    return parser


def main(args):
    selected = [canonical_dataset_name(n.strip())
                for n in (args.whar_pretrain_datasets.split(',') if args.whar_pretrain_datasets
                          else BENCHMARK_DATASETS)
                if n.strip()]
    excluded = {canonical_dataset_name(n.strip())
                for n in args.whar_exclude_datasets.split(',') if n.strip()}
    datasets = [n for n in selected if n not in excluded]
    if not datasets:
        raise ValueError("No WHAR pretraining datasets remain after exclusions")

    print(args)
    print(f"Pretraining datasets ({len(datasets)}): {', '.join(datasets)}")
    if excluded:
        print(f"Excluded: {', '.join(sorted(excluded))}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    start_run(args, 'pretrain', 'pretrain_base')

    dataset_config = get_dataset_config('whar_all', channel_subset='standard')
    model = sensor_jepa_base(**dataset_config).to(device)

    target_encoder = copy.deepcopy(model.context_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_loader, train_dataset = load_whar_combined_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        dataset_names=datasets,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        budget=args.budget,
        window_size=args.window_size,
        window_stride=args.window_stride,
        num_workers=args.num_workers,
        sampling_rate=args.sampling_rate,
    )
    print(f"Train windows per epoch: {len(train_dataset)}")

    num_patches = dataset_config['num_patches']
    masking_fn = RandomTubeMasking1D(num_patches=num_patches, mask_ratio=args.mask_ratio)
    print(f"num_patches={num_patches}, mask_ratio={args.mask_ratio}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, args.warmup_epochs)

    start_epoch, history = 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume_args(checkpoint['args'], args, [
            'batch_size', 'epochs', 'warmup_epochs', 'lr', 'min_lr', 'weight_decay',
            'reg_coeff', 'mask_ratio', 'ema_momentum_start', 'ema_momentum_end',
            'budget', 'train_ratio', 'window_size', 'window_stride', 'sampling_rate',
            'split_seed', 'seed', 'whar_pretrain_datasets', 'whar_exclude_datasets', 'num_workers',
        ])
        if checkpoint.get('selection_criterion') != 'fixed_final_epoch':
            raise ValueError('Resume requires a fixed_final_epoch checkpoint')
        if not all(key in checkpoint for key in ('rng_state', 'optimizer_state_dict', 'epoch_history', 'global_step')):
            raise ValueError('This checkpoint lacks the state required for exact recovery')
        start_epoch = load_checkpoint(args.resume, model, target_encoder, optimizer)
        history = checkpoint['epoch_history']
        train_dataset.epoch = start_epoch
        restore_rng_state(checkpoint['rng_state'])
        del checkpoint

    print("Starting training...")
    global_step = history[-1]['global_step'] if history else 0
    for epoch in range(start_epoch, args.epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]
        if args.budget is not None:
            train_dataset.resample_epoch()

        metrics = train_one_epoch(
            model=model, target_encoder=target_encoder, data_loader=train_loader,
            optimizer=optimizer, device=device, epoch=epoch, masking_fn=masking_fn,
            ema_momentum_start=args.ema_momentum_start, ema_momentum_end=args.ema_momentum_end,
            total_epochs=args.epochs, reg_coeff=args.reg_coeff, global_step_offset=global_step,
        )
        global_step = metrics['global_step']
        if not math.isfinite(metrics['loss']):
            raise ValueError(f'Non-finite pretraining loss at epoch {epoch + 1}')
        history.append({'epoch': epoch + 1, **metrics})

        wandb.log({'epoch': epoch + 1, 'epoch/loss': metrics['loss'],
                   'epoch/feature_loss': metrics['feature_loss'],
                   'epoch/reg_loss': metrics['reg_loss'],
                   'epoch/learning_rate': lr_schedule[epoch]})
        print(f"Epoch {epoch + 1}/{args.epochs}: Loss={metrics['loss']:.4f}, "
              f"FeatureLoss={metrics['feature_loss']:.4f}, RegLoss={metrics['reg_loss']:.4f}, "
              f"GradNorm={metrics['grad_norm']:.4f}")

        state = {
            'epoch': epoch + 1, 'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'target_encoder_state_dict': target_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'], 'args': args,
            'checkpoint_type': 'latest',
            'selection_criterion': 'fixed_final_epoch',
            'epoch_history': history, 'rng_state': capture_rng_state(),
        }
        save_checkpoint(state, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
        save_metrics(args, history, datasets, excluded)

    # Repair metrics if a crash occurred after the final checkpoint was committed.
    if history:
        save_metrics(args, history, datasets, excluded)

    print("Training complete!")
    wandb.finish()


def save_metrics(args, history, datasets, excluded):
    latest = history[-1]
    write_json(os.path.join(args.output_dir, 'metrics.json'), {
        'stage': 'pretrain', 'complete': latest['epoch'] == args.epochs,
        'epochs_completed': latest['epoch'], 'epochs': args.epochs,
        'checkpoint': os.path.join(args.output_dir, 'checkpoint_latest.pth'),
        'checkpoint_type': 'latest', 'loss': latest['loss'],
        'selection_criterion': 'fixed_final_epoch',
        'selection_uses_test_set': False,
        'loss_definition': 'L1 feature loss + reg_coeff * variance-hinge loss',
        'pretraining_sources': datasets, 'excluded_datasets': sorted(excluded),
        'args': vars(args), 'epoch_history': history,
    })
    report_progress(args, latest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Wavelet W-JEPA Pretraining', parents=[get_args_parser()])
    main(parser.parse_args())
