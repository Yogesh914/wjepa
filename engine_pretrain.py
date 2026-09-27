import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from util.masking import gather_tokens


def prediction_losses(predictions, targets, reg_coeff):
    """L1 and per-sample, per-feature variance hinge (unbiased variance)."""
    feature_loss = (predictions - targets).abs().mean(dim=(1, 2)).mean()
    reg_loss = predictions.new_zeros(())
    if reg_coeff > 0:
        std = torch.sqrt(predictions.var(dim=1) + 1e-4)
        reg_loss = torch.relu(1.0 - std).mean(dim=1).mean()
    return feature_loss, reg_loss


def train_one_epoch(
    model,
    target_encoder,
    data_loader,
    optimizer,
    device,
    epoch,
    masking_fn,
    ema_momentum_start,
    ema_momentum_end,
    total_epochs,
    reg_coeff=0.0625,
    log_interval=50,
    ema_scale_factor=1.25,
    global_step_offset=0,
):
    """One W-JEPA epoch: L1 latent prediction of masked targets + a variance hinge."""
    model.train()
    target_encoder.eval()

    # Match Python's double-precision metric accumulation without synchronizing
    # the host after every batch (or after every parameter's gradient norm).
    totals = torch.zeros(4, dtype=torch.float64, device=device)

    iters_per_epoch = len(data_loader)
    total_iters = int(iters_per_epoch * total_epochs * ema_scale_factor)
    current_epoch_start_iter = epoch * iters_per_epoch

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for batch_idx, data in enumerate(pbar):
        x = data[0] if isinstance(data, (list, tuple)) else data
        x = x.to(device, non_blocking=True)
        batch_size = x.shape[0]

        context_mask, target_mask = masking_fn.sample_indices(batch_size, device=device)

        wavelet_features = model.context_encoder.patch_embed.extract_features(x)
        context_features = model.context_encoder(x, context_mask, wavelet_features=wavelet_features)

        with torch.no_grad():
            target_features = target_encoder(x, wavelet_features=wavelet_features)
            targets = gather_tokens(target_features, target_mask)
            targets = nn.functional.layer_norm(targets, (targets.shape[-1],))

        predictions = model.predictor(context_features, context_mask, target_mask)

        feature_loss, reg_loss = prediction_losses(predictions, targets, reg_coeff)

        loss = feature_loss + reg_coeff * reg_loss

        optimizer.zero_grad()
        loss.backward()

        gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        # This norm is only logged; it never clips gradients or changes updates.
        norms = (
            torch._foreach_norm(gradients, 2)
            if x.is_cuda
            else [gradient.norm(2) for gradient in gradients]
        )
        grad_norm = torch.stack(norms).double().square().sum().sqrt()

        optimizer.step()

        current_iter = current_epoch_start_iter + batch_idx
        ema_momentum = ema_momentum_start + (ema_momentum_end - ema_momentum_start) * (
            current_iter / total_iters
        )
        update_target_encoder(model.context_encoder, target_encoder, ema_momentum)

        values = torch.stack(
            (
                loss.detach().double(),
                feature_loss.detach().double(),
                reg_loss.detach().double(),
                grad_norm,
            )
        )
        totals += values

        if batch_idx % log_interval == 0:
            global_step = global_step_offset + batch_idx
            loss_value, feature_value, reg_value, norm_value = values.tolist()
            wandb.log(
                {
                    'global_step': global_step,
                    'step/loss': loss_value,
                    'step/feature_loss': feature_value,
                    'step/reg_loss': reg_value,
                    'step/ema_momentum': ema_momentum,
                    'step/learning_rate': optimizer.param_groups[0]['lr'],
                    'step/grad_norm': norm_value,
                },
            )
            pbar.set_postfix({'loss': loss_value, 'reg': reg_value, 'ema': ema_momentum})

    num_batches = len(data_loader)
    total_loss, total_feature_loss, total_reg_loss, total_grad_norm = totals.tolist()
    return {
        'loss': total_loss / num_batches,
        'feature_loss': total_feature_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches,
        'grad_norm': total_grad_norm / num_batches,
        'global_step': global_step_offset + num_batches,
    }


def update_target_encoder(context_encoder, target_encoder, momentum):
    with torch.no_grad():
        pairs = list(zip(context_encoder.parameters(), target_encoder.parameters()))
        if pairs and pairs[0][0].is_cuda:
            source, target = map(list, zip(*pairs))
            # Separate multiplies followed by addition preserve the reference
            # arithmetic; a fused lerp/add(alpha=...) can round differently.
            torch._foreach_mul_(target, momentum)
            torch._foreach_add_(target, torch._foreach_mul(source, 1.0 - momentum))
        else:
            for param_q, param_k in pairs:
                param_k.copy_(param_k * momentum + param_q * (1.0 - momentum))


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0):
    warmup_epochs = min(warmup_epochs, epochs)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(final_value, base_value, warmup_epochs + 1)[1:]
    else:
        warmup_schedule = np.array([])

    remaining_epochs = epochs - warmup_epochs
    if remaining_epochs > 0:
        iters = np.arange(remaining_epochs)
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / remaining_epochs)
        )
    else:
        cosine_schedule = np.array([])

    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    assert len(schedule) == epochs, f'Schedule length {len(schedule)} != epochs {epochs}'
    return schedule


def save_checkpoint(state, filename='checkpoint.pth'):
    from util.checkpointing import atomic_torch_save

    atomic_torch_save(state, filename)
    print(f'Checkpoint saved to {filename}')


def load_checkpoint(filename, model, target_encoder, optimizer=None):
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f'Checkpoint loaded from {filename}, epoch {epoch}')
    return epoch
