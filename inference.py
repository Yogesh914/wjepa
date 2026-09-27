"""Inference for the Wavelet W-JEPA pipeline.

Two modes:

  classify  Load a fine-tuned checkpoint (encoder + classifier) and predict
            activity labels. Either evaluate the saved dataset's test split, or
            predict labels for raw windows in a .npy file shaped [N, C, T].

  embed     Load a pretrained checkpoint and emit frozen JEPA embeddings for a
            dataset split or for raw windows in a .npy file.

Examples:
  python inference.py classify --checkpoint checkpoints/finetune/dsads/checkpoint_latest.pth
  python inference.py classify --checkpoint checkpoints/finetune/dsads/checkpoint_latest.pth --input windows.npy --output preds.npy
  python inference.py embed --checkpoint checkpoints/pretrain_all/checkpoint_latest.pth --dataset uci_har --output emb.npy
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from config import BASE_MODEL_CONFIG, WAVELET_CONFIG
from models_jepa import VisionTransformer
from engine_finetune import ClassificationHead, evaluate
from util.datasets import get_whar_dataloaders


def _build_encoder(num_channels, device):
    encoder = VisionTransformer(
        num_channels=num_channels,
        embed_dim=BASE_MODEL_CONFIG['embed_dim'],
        depth=BASE_MODEL_CONFIG['depth'],
        num_heads=BASE_MODEL_CONFIG['num_heads'],
        mlp_ratio=BASE_MODEL_CONFIG['mlp_ratio'],
        wavelet=WAVELET_CONFIG['wavelet'],
        wavelet_level=WAVELET_CONFIG['wavelet_level'],
        wavelet_tokens_per_level=WAVELET_CONFIG['wavelet_tokens_per_level'],
    )
    return encoder.to(device).eval()


def _channels_from_encoder_state(state):
    return state['patch_embed.channel_embed'].shape[1]


def _arg_or_saved(args, saved_args, name, default):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return getattr(saved_args, name, default)


@torch.no_grad()
def _encode_array(encoder, array, device, batch_size=256):
    """array: [N, C, T] -> features [N, num_patches, embed_dim]."""
    x = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if x.dim() == 3:
        x = x.unsqueeze(1)  # [N, 1, C, T]
    feats = []
    for start in range(0, x.shape[0], batch_size):
        feats.append(encoder(x[start:start + batch_size].to(device)).cpu())
    return torch.cat(feats, dim=0)


def run_classify(args, device):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder_state = ckpt['encoder_state_dict']
    classifier_state = ckpt['classifier_state_dict']
    num_channels = _channels_from_encoder_state(encoder_state)
    num_classes = classifier_state['head.2.weight'].shape[0]
    input_dim = classifier_state['head.2.weight'].shape[1]
    num_tokens = input_dim // BASE_MODEL_CONFIG['embed_dim']

    encoder = _build_encoder(num_channels, device)
    encoder.load_state_dict(encoder_state)
    classifier = ClassificationHead(
        embed_dim=BASE_MODEL_CONFIG['embed_dim'], num_classes=num_classes,
        num_tokens=num_tokens).to(device).eval()
    classifier.load_state_dict(classifier_state)

    if args.input:
        windows = np.load(args.input)
        feats = _encode_array(encoder, windows, device).to(device)
        with torch.no_grad():
            preds = classifier(feats).argmax(dim=1).cpu().numpy()
        print(f"Predicted {len(preds)} labels: {preds[:20]}{' ...' if len(preds) > 20 else ''}")
        if args.output:
            np.save(args.output, preds)
            print(f"Saved predictions to {args.output}")
        return

    saved_args = ckpt.get('args')
    dataset = args.dataset or ckpt.get('dataset') or getattr(saved_args, 'dataset', None)
    if dataset is None:
        raise ValueError("Checkpoint has no dataset; pass --dataset or --input")
    channel_subset = (ckpt.get('channel_subset')
                      or getattr(saved_args, 'whar_channel_subset', None)
                      or getattr(saved_args, 'channel_subset', None) or 'all')
    if args.channel_subset is not None:
        channel_subset = args.channel_subset
    train_ratio = _arg_or_saved(args, saved_args, 'train_ratio', 0.2)
    split_seed = _arg_or_saved(args, saved_args, 'split_seed', 42)
    split_strategy = _arg_or_saved(args, saved_args, 'split_strategy', 'window')
    window_size = _arg_or_saved(args, saved_args, 'window_size', 500)
    window_stride = _arg_or_saved(args, saved_args, 'window_stride', 500)
    sampling_rate = _arg_or_saved(args, saved_args, 'sampling_rate', None)
    label_smoothing = _arg_or_saved(args, saved_args, 'label_smoothing', 0.0)

    _, test_loader, _, test_dataset = get_whar_dataloaders(
        data_path=args.data_path, dataset_name=dataset, batch_size=args.batch_size,
        return_labels=True, split_seed=split_seed, train_ratio=train_ratio,
        window_size=window_size, window_stride=window_stride, num_workers=args.num_workers,
        channel_subset=channel_subset, sampling_rate=sampling_rate, split_strategy=split_strategy)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    metrics = evaluate(encoder, classifier, test_loader, criterion, device, test_dataset.num_classes)

    payload = {
        'dataset': dataset,
        'checkpoint': args.checkpoint,
        'channel_subset': channel_subset,
        'train_ratio': train_ratio,
        'split_seed': split_seed,
        'split_strategy': split_strategy,
        'test_participants': test_dataset.participants,
        'window_size': window_size,
        'window_stride': window_stride,
        'sampling_rate': sampling_rate,
        'label_smoothing': label_smoothing,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'device': str(device),
        'num_classes': test_dataset.num_classes,
        'num_test_samples': len(test_dataset),
        'loss': metrics['loss'],
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'per_class_accuracy': metrics['per_class_accuracy'],
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
    }
    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file) or '.', exist_ok=True)
        with open(args.metrics_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.metrics_file}")
    print(f"{dataset}: accuracy={metrics['accuracy']:.2f}%  macro_f1={metrics['macro_f1']:.2f}%")
    return payload


def run_embed(args, device):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_state = ckpt['model_state_dict']
    encoder_state = {k.replace('context_encoder.', ''): v
                     for k, v in model_state.items() if k.startswith('context_encoder.')}
    num_channels = _channels_from_encoder_state(encoder_state)
    encoder = _build_encoder(num_channels, device)
    encoder.load_state_dict(encoder_state)

    def pool(feats):
        return feats.mean(dim=1) if args.pooling == 'mean' else feats.reshape(feats.shape[0], -1)

    if args.input:
        feats = _encode_array(encoder, np.load(args.input), device)
        embeddings = pool(feats).numpy()
        labels = None
    else:
        if args.dataset is None:
            raise ValueError("Pass --dataset or --input")
        _, test_loader, _, _ = get_whar_dataloaders(
            data_path=args.data_path, dataset_name=args.dataset, batch_size=args.batch_size,
            return_labels=True, train_ratio=0.2, num_workers=args.num_workers,
            channel_subset='standard')
        emb_list, lab_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                emb_list.append(pool(encoder(x.to(device)).cpu()))
                lab_list.append(y)
        embeddings = torch.cat(emb_list, dim=0).numpy()
        labels = torch.cat(lab_list, dim=0).numpy()

    print(f"Embeddings: {embeddings.shape}")
    if args.output:
        np.save(args.output, embeddings)
        print(f"Saved embeddings to {args.output}")
        if labels is not None:
            label_path = args.output.replace('.npy', '') + '_labels.npy'
            np.save(label_path, labels)
            print(f"Saved labels to {label_path}")


def get_args_parser():
    parser = argparse.ArgumentParser('Wavelet W-JEPA inference')
    sub = parser.add_subparsers(dest='mode', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--checkpoint', required=True)
    common.add_argument('--data_path', default='./datasets')
    common.add_argument('--dataset', default=None)
    common.add_argument('--input', default=None, help='Raw windows .npy of shape [N, C, T]')
    common.add_argument('--output', default=None, help='Where to save predictions/embeddings (.npy)')
    common.add_argument('--batch_size', default=256, type=int)
    common.add_argument('--num_workers', default=0, type=int)
    common.add_argument('--device', default='cuda')

    classify = sub.add_parser(
        'classify', parents=[common], help='Predict activity labels with a fine-tuned model')
    classify.add_argument('--metrics_file', default=None,
                          help='Write evaluation metrics as JSON when no --input is provided')
    classify.add_argument('--channel_subset', default=None, choices=['all', 'standard'],
                          help='Override checkpoint channel subset for dataset evaluation')
    classify.add_argument('--train_ratio', default=None, type=float,
                          help='Override checkpoint train split ratio for dataset evaluation')
    classify.add_argument('--split_seed', default=None, type=int,
                          help='Override checkpoint split seed for dataset evaluation')
    classify.add_argument('--split_strategy', default=None, choices=['window', 'participant'],
                          help='Override checkpoint window/participant split strategy')
    classify.add_argument('--window_size', default=None, type=int,
                          help='Override checkpoint window size for dataset evaluation')
    classify.add_argument('--window_stride', default=None, type=int,
                          help='Override checkpoint window stride for dataset evaluation')
    classify.add_argument('--sampling_rate', default=None, type=float,
                          help='Override checkpoint session resampling rate in Hz')
    classify.add_argument('--label_smoothing', default=None, type=float,
                          help='Override checkpoint label smoothing for reported evaluation loss')
    embed = sub.add_parser('embed', parents=[common], help='Extract frozen JEPA embeddings')
    embed.add_argument('--pooling', default='mean', choices=['mean', 'flatten'])
    return parser


def main():
    args = get_args_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.mode == 'classify':
        run_classify(args, device)
    else:
        run_embed(args, device)


if __name__ == '__main__':
    main()
