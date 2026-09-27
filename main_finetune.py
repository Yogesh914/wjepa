"""Full fine-tuning of the Wavelet W-JEPA encoder on a WHAR benchmark.

Loads the pretrained context encoder, attaches a flattened-token linear head, and
fine-tunes end-to-end with all configured channels (the target-adapted recipe).
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import wandb

from config import (
    BASE_MODEL_CONFIG, BENCHMARK_DATASETS, FINETUNE_CONFIG, canonical_dataset_name,
    get_dataset_config,
)
from models_jepa import VisionTransformer
from engine_finetune import (
    ClassificationHead, train_one_epoch, evaluate, save_checkpoint,
    load_pretrained_encoder, print_metrics,
)
from util.datasets import get_whar_dataloaders
from util.experiment import start_run, report_progress
from util.checkpointing import (
    capture_rng_state, preserve_rng_state, restore_rng_state,
    validate_resume_args, write_json,
)


def get_args_parser():
    parser = argparse.ArgumentParser('Wavelet W-JEPA Fine-tuning', add_help=False)

    parser.add_argument('--pretrained', default='./checkpoints/pretrain/checkpoint_latest.pth', type=str,
                        help='Pretrained W-JEPA checkpoint (context encoder is loaded)')
    parser.add_argument('--dataset', default='dsads', type=str, choices=BENCHMARK_DATASETS)
    parser.add_argument('--data_path', default='./datasets', type=str)

    parser.add_argument('--batch_size', default=FINETUNE_CONFIG['batch_size'], type=int)
    parser.add_argument('--epochs', default=FINETUNE_CONFIG['epochs'], type=int)
    parser.add_argument('--warmup_epochs', default=FINETUNE_CONFIG['warmup_epochs'], type=int)
    parser.add_argument('--lr', default=FINETUNE_CONFIG['lr'], type=float)
    parser.add_argument('--min_lr', default=FINETUNE_CONFIG['min_lr'], type=float)
    parser.add_argument('--weight_decay', default=FINETUNE_CONFIG['weight_decay'], type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    parser.add_argument('--channel_subset', default='all', choices=['all', 'standard'],
                        help="'all' is the target-adapted recipe; 'standard' uses six channels")
    parser.add_argument('--finetune_aug', action='store_true',
                        help='Train-only jitter/scale augmentation (used for every target except MHealth)')
    parser.add_argument('--train_ratio', default=0.2, type=float)
    parser.add_argument('--split_seed', default=42, type=int)
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed Python, NumPy and PyTorch; unset preserves stochastic initialization')
    parser.add_argument('--split_strategy', default='window', choices=['window', 'participant'],
                        help='Window split, or disjoint participants using the participant_split branch recipe')
    parser.add_argument('--window_size', default=500, type=int)
    parser.add_argument('--window_stride', default=500, type=int)
    parser.add_argument('--sampling_rate', default=None, type=float,
                        help='Resample each continuous session to this rate in Hz before windowing')

    parser.add_argument('--output_dir', default='./checkpoints/finetune', type=str)
    parser.add_argument('--metrics_file', default='', type=str)
    parser.add_argument('--resume', default='', type=str,
                        help='Resume the latest fine-tuning checkpoint with optimizer and RNG state')
    parser.add_argument('--wandb_project', default='wavelet-jepa', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_mode', default='disabled', type=str,
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    return parser


def main(args):
    args.dataset = canonical_dataset_name(args.dataset)
    if args.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {args.epochs}")
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    start_run(args, 'finetune', f'finetune_{args.dataset}')

    dataset_config = get_dataset_config(args.dataset, channel_subset=args.channel_subset)
    encoder = VisionTransformer(
        num_channels=dataset_config['num_channels'],
        embed_dim=BASE_MODEL_CONFIG['embed_dim'],
        depth=BASE_MODEL_CONFIG['depth'],
        num_heads=BASE_MODEL_CONFIG['num_heads'],
        mlp_ratio=BASE_MODEL_CONFIG['mlp_ratio'],
        wavelet=dataset_config['wavelet'],
        wavelet_level=dataset_config['wavelet_level'],
        wavelet_tokens_per_level=dataset_config['wavelet_tokens_per_level'],
    )
    if args.pretrained:
        encoder = load_pretrained_encoder(encoder, args.pretrained)
    else:
        print("No pretrained checkpoint provided; training encoder from scratch.")
    encoder = encoder.to(device)

    train_loader, test_loader, train_dataset, test_dataset = get_whar_dataloaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        return_labels=True,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        window_size=args.window_size,
        window_stride=args.window_stride,
        num_workers=args.num_workers,
        channel_subset=args.channel_subset,
        use_augmentation=args.finetune_aug,
        sampling_rate=args.sampling_rate,
        split_strategy=args.split_strategy,
    )
    num_classes = train_dataset.num_classes
    train_participants = getattr(train_dataset, 'participants', [])
    test_participants = getattr(test_dataset, 'participants', [])
    if args.split_strategy == 'participant' and set(train_participants) & set(test_participants):
        raise ValueError('Participant train/test sets overlap')
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, "
          f"classes: {num_classes}, channels: {dataset_config['num_channels']}")

    classifier = ClassificationHead(
        embed_dim=BASE_MODEL_CONFIG['embed_dim'], num_classes=num_classes,
        num_tokens=encoder.num_patches, dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    from engine_pretrain import cosine_scheduler
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, args.warmup_epochs)

    latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
    start_epoch, history = 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume_args(checkpoint['args'], args, [
            'dataset', 'pretrained', 'batch_size', 'epochs', 'warmup_epochs', 'lr',
            'min_lr', 'weight_decay', 'dropout', 'label_smoothing', 'channel_subset',
            'finetune_aug', 'train_ratio', 'split_seed', 'window_size', 'window_stride',
            'sampling_rate', 'num_workers', 'seed',
            'split_strategy',
        ])
        if checkpoint.get('selection_criterion') != 'fixed_final_epoch':
            raise ValueError('Resume requires a fixed_final_epoch checkpoint')
        if not all(key in checkpoint for key in ('rng_state', 'optimizer_state_dict', 'epoch_history')):
            raise ValueError('This checkpoint lacks the state required for exact recovery')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['epoch_history']
        restore_rng_state(checkpoint['rng_state'])
        del checkpoint

    print(f"Starting fixed-duration fine-tuning for {args.epochs} epochs...")
    print('Checkpoint policy: retain the latest epoch; report the fixed final epoch.')
    for epoch in range(start_epoch, args.epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]

        train_metrics = train_one_epoch(encoder, classifier, train_loader, criterion,
                                        optimizer, device, epoch)
        # Test evaluation consumes DataLoader worker seeds even without augmentation.
        # Preserve RNG so the extra evaluations do not alter the training recipe.
        with preserve_rng_state():
            test_metrics = evaluate(encoder, classifier, test_loader, criterion, device, num_classes)
        test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
        if not all(math.isfinite(test_metrics[name]) for name in ('loss', 'accuracy', 'macro_f1')):
            raise ValueError(f'Non-finite test metrics at epoch {epoch + 1}')
        latest_metrics = {'epoch': epoch + 1, **test_metrics}
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'test': test_metrics})

        wandb.log({'epoch': epoch + 1, 'train/loss': train_metrics['loss'],
                   'train/accuracy': train_metrics['accuracy'],
                   'train/lr': lr_schedule[epoch],
                   'test/loss': test_metrics['loss'],
                   'test/accuracy': test_metrics['accuracy'],
                   'test/macro_f1': test_metrics['macro_f1']})
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Acc={train_metrics['accuracy']:.2f}% | "
              f"Test Acc={test_metrics['accuracy']:.2f}% F1={test_metrics['macro_f1']:.2f}%")

        state = {
            'epoch': epoch + 1, 'checkpoint_type': 'latest',
            'encoder_state_dict': encoder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_metrics['accuracy'], 'train_loss': train_metrics['loss'],
            'dataset': args.dataset, 'channel_subset': args.channel_subset,
            'num_classes': num_classes, 'args': args,
            'split_strategy': args.split_strategy,
            'train_participants': train_participants, 'test_participants': test_participants,
            'test_metrics': latest_metrics,
            'selection_criterion': 'fixed_final_epoch',
            'epoch_history': history, 'rng_state': capture_rng_state(),
        }
        save_checkpoint(state, latest_path)
        save_metrics(args, latest_metrics, history, num_classes,
                     len(train_dataset), len(test_dataset), train_participants, test_participants)

    if history:
        # Repair metrics after a crash between checkpoint and metrics commits.
        save_metrics(args, {'epoch': history[-1]['epoch'], **history[-1]['test']},
                     history, num_classes, len(train_dataset), len(test_dataset),
                     train_participants, test_participants)
        print_metrics(history[-1]['test'], prefix=f"Latest Test (epoch {history[-1]['epoch']})")
    print('\nFine-tuning complete!')
    wandb.finish()


def save_metrics(args, latest, history, num_classes, num_train_samples, num_test_samples,
                 train_participants, test_participants):
    metrics_file = args.metrics_file or os.path.join(args.output_dir, 'metrics.json')
    payload = {
        'dataset': args.dataset,
        'pretrained': args.pretrained,
        'checkpoint': os.path.join(args.output_dir, 'checkpoint_latest.pth'),
        'checkpoint_type': 'latest',
        'complete': latest['epoch'] == args.epochs,
        'epochs_completed': latest['epoch'],
        'epochs': args.epochs,
        'channel_subset': args.channel_subset,
        'finetune_aug': args.finetune_aug,
        'train_ratio': args.train_ratio,
        'split_seed': args.split_seed,
        'split_strategy': args.split_strategy,
        'train_participants': train_participants,
        'test_participants': test_participants,
        'sampling_rate': args.sampling_rate,
        'window_size': args.window_size,
        'window_stride': args.window_stride,
        'num_classes': num_classes,
        'num_train_samples': num_train_samples,
        'num_test_samples': num_test_samples,
        'accuracy': latest['accuracy'],
        'macro_f1': latest['macro_f1'],
        'loss': latest['loss'],
        'per_class_accuracy': latest['per_class_accuracy'],
        'latest': {**latest, 'checkpoint': os.path.join(args.output_dir, 'checkpoint_latest.pth')},
        'selection_criterion': 'fixed_final_epoch',
        'selection_uses_test_set': False,
        'epoch_history': history,
        'args': vars(args),
    }
    write_json(metrics_file, payload)
    report_progress(args, latest)
    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Wavelet W-JEPA Fine-tuning', parents=[get_args_parser()])
    main(parser.parse_args())
