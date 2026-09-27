import torch
import torch.nn as nn
from tqdm import tqdm
import wandb


class ClassificationHead(nn.Module):
    """Linear classifier over the flattened encoder tokens."""

    def __init__(self, embed_dim, num_classes, num_tokens, dropout=0.1):
        super().__init__()
        input_dim = embed_dim * num_tokens
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x.reshape(x.shape[0], -1))


def train_one_epoch(
    encoder, classifier, data_loader, criterion, optimizer, device, epoch, log_interval=50
):
    encoder.train()
    classifier.train()

    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = classifier(encoder(x))
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().double()
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum()

        if batch_idx % log_interval == 0:
            loss_value = loss.item()
            acc = 100.0 * correct.item() / total
            wandb.log(
                {'global_step': epoch * len(data_loader) + batch_idx,
                 'step/train_loss': loss_value, 'step/train_accuracy': acc},
            )
            pbar.set_postfix({'loss': loss_value, 'acc': f'{acc:.2f}%'})

    return {
        'loss': total_loss.item() / len(data_loader),
        'accuracy': 100.0 * correct.item() / total,
    }


@torch.no_grad()
def evaluate(encoder, classifier, data_loader, criterion, device, num_classes):
    encoder.eval()
    classifier.eval()

    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for batch in tqdm(data_loader, desc='Evaluating'):
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = classifier(encoder(x))
        total_loss += criterion(logits, y).double()

        _, predicted = logits.max(1)
        total += y.size(0)
        # The class count is known: update fixed-size integer bins directly.
        # CUDA bincount reads its input min/max back to the CPU even when
        # minlength is given, forcing synchronization after every batch.
        confusion.view(-1).scatter_add_(0, y * num_classes + predicted, torch.ones_like(y))

    confusion = confusion.cpu()
    correct = confusion.diag().sum().item()
    per_class_acc = []
    f1_scores = []
    for i in range(num_classes):
        tp = confusion[i, i].item()
        actual = confusion[i, :].sum().item()
        pred = confusion[:, i].sum().item()
        per_class_acc.append(100.0 * tp / actual if actual else 0.0)
        precision = tp / pred if pred else 0.0
        recall = tp / actual if actual else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    return {
        'loss': total_loss.item() / len(data_loader),
        'accuracy': 100.0 * correct / total,
        'macro_f1': 100.0 * sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion,
    }


def save_checkpoint(state, filename='checkpoint_finetune.pth'):
    from util.checkpointing import atomic_torch_save

    atomic_torch_save(state, filename)
    print(f'Checkpoint saved to {filename}')


def load_pretrained_encoder(encoder, pretrain_checkpoint_path):
    """Load the JEPA context-encoder weights into a fresh encoder.

    Channel-dependent parameters (the wavelet channel embedding) are skipped when
    the downstream channel count differs; they are trained from scratch during the
    full fine-tune.
    """
    checkpoint = torch.load(pretrain_checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict']
    encoder_state = {
        key.replace('context_encoder.', ''): value
        for key, value in model_state.items()
        if key.startswith('context_encoder.')
    }

    current_state = encoder.state_dict()
    compatible = {}
    skipped = []
    for key, value in encoder_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    encoder.load_state_dict(compatible, strict=False)
    print(
        f'Loaded {len(compatible)}/{len(current_state)} context-encoder layers '
        f'from {pretrain_checkpoint_path}'
    )
    if skipped:
        print(f'Reinitialized (shape mismatch, trained during fine-tune): {skipped}')
    return encoder


def print_metrics(metrics, prefix=''):
    print(f'\n{prefix} Metrics:')
    print(f'  Loss: {metrics["loss"]:.4f}')
    print(f'  Accuracy: {metrics["accuracy"]:.2f}%')
    print(f'  Macro F1: {metrics["macro_f1"]:.2f}%')
